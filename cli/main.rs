mod docker;
mod fs;
mod opts;
mod prompt;

use std::{
    env,
    ffi::{OsStr, OsString},
    io,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
};

use chimera::{HostFs, MountFlags, Namespace, OverlayFs, Personality, Sandbox, Vfs};
use mimalloc::MiMalloc;

use opts::{Command, Opts, RunCmd};

/// Route every Chimera-side allocation through mimalloc, whose segments are
/// `mmap`-backed and never touch `brk`. This keeps Chimera's heap clear of the
/// guest libc's `brk`-managed `main_arena`, which shares the one process-wide
/// program break.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> ExitCode {
    let opts: Opts = argh::from_env();
    match opts.command {
        Command::Run(cmd) => run(cmd),
        Command::Version(_) => version(),
        Command::Fs(cmd) => fs::command(cmd.action),
    }
}

fn version() -> ExitCode {
    println!(
        "chimera version {} linux x86-64 {}",
        env!("CARGO_PKG_VERSION"),
        if chimera::mpk_enabled() {
            "mpk"
        } else {
            "nompk"
        }
    );
    ExitCode::SUCCESS
}

/// What a run assembles before the sandbox starts: the merged root the guest
/// sees, the filesystem receiving its changes (`None` under `--unsafe`), and
/// the held base image tree when that filesystem branched an image.
type Session = (Arc<dyn Vfs>, Option<fs::Filesystem>, Option<fs::Base>);

fn run(mut cmd: RunCmd) -> ExitCode {
    // Only a shell Chimera starts on its own gets a badged prompt; a program
    // the user named runs exactly as typed.
    let implicit_shell = cmd.argv.is_empty();
    if implicit_shell {
        cmd.argv.push("/bin/bash".into());
    }
    // Route the guest's filesystem syscalls through a userspace VFS mounted
    // at `/`. Every run works in a filesystem: by default it branches one —
    // the live host — so the guest sees what looks like the real tree while
    // every mutation lands in the branch's change-set; `--in` resumes an
    // existing filesystem instead, and `--unsafe` is the refusal to have one
    // at all. The host changes only through `fs apply`. The host root must
    // exist, so its construction cannot fail.
    let host = Arc::new(HostFs::new("/").expect("host root / is a directory"));
    let (root, fsys, base): Session = if cmd.unsafe_ {
        // The explicit flags contradict --unsafe; the ambient environment
        // variable is merely inert, like any other env a script exports.
        if cmd.from.is_some() || cmd.in_.is_some() || cmd.rm {
            eprintln!("chimera: --unsafe runs without a filesystem");
            return ExitCode::FAILURE;
        }
        (host, None, None)
    } else {
        if cmd.from.is_some() && cmd.in_.is_some() {
            eprintln!("chimera: a run either branches with --from or resumes with --in, not both");
            return ExitCode::FAILURE;
        }
        // The environment supplies a default for --in only; an explicit
        // verb always wins over an inherited one.
        let env_in = (cmd.from.is_none() && cmd.in_.is_none())
            .then(|| env::var_os("CHIMERA_FS"))
            .flatten();
        let from_env = env_in.is_some();
        let resume = cmd.in_.clone().map(OsString::from).or(env_in);
        if cmd.rm && resume.is_some() {
            eprintln!(
                "chimera: --rm discards a fresh branch, not a filesystem resumed with --in{}",
                if from_env { " (CHIMERA_FS)" } else { "" },
            );
            return ExitCode::FAILURE;
        }
        let selector = resume
            .clone()
            .or_else(|| cmd.from.clone().map(OsString::from));
        if let Some(sel) = &selector
            && let Some(scheme) = fs::scheme(sel)
        {
            if scheme != "docker" {
                eprintln!(
                    "chimera: unknown filesystem scheme \"{scheme}:\" (a path whose first component contains a colon needs a leading ./)",
                );
                return ExitCode::FAILURE;
            }
            // An image can only ever be a branch point: resuming one would
            // falsify the provenance every branch of it depends on.
            if resume.is_some() {
                eprintln!(
                    "chimera: {} is an image and immutable; branch it with --from{}",
                    Path::new(sel).display(),
                    if from_env { " (CHIMERA_FS)" } else { "" },
                );
                return ExitCode::FAILURE;
            }
        }
        // A locator names a filesystem; the verb decides what happens to
        // it: `--from` branches, leaving the source exactly as it was, and
        // `--in` resumes, accumulating changes into the named filesystem.
        // An id names a kept change-set under the state directory, a path
        // names a change-set directory in place, and `host` is the default
        // branch point spelled out, so what `fs list` prints in its FROM
        // column can be typed straight back; a generated id is 8 hex
        // characters and can never collide with the word.
        let fsys = match &resume {
            Some(sel) if sel.as_bytes() == b"host" => {
                eprintln!(
                    "chimera: --in host mutates the live host; that operation is spelled --unsafe"
                );
                return ExitCode::FAILURE;
            }
            Some(sel) if sel.as_bytes().contains(&b'/') => fs::attach(sel),
            Some(sel) => fs::resume(&sel.to_string_lossy()),
            None => match &cmd.from {
                Some(sel) if sel == "host" => fs::create(&describe(&cmd)),
                // A docker: image resolves to a kept image filesystem —
                // pulled now if this is its first use — and the run branches
                // that, exactly as if its id had been named.
                Some(sel) if fs::scheme(OsStr::new(sel)).as_deref() == Some("docker") => {
                    match docker::pull(sel) {
                        Ok(id) => fs::branch(&id, &describe(&cmd)),
                        Err(err) => {
                            eprintln!("chimera: cannot pull {sel}: {err}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
                Some(sel) => fs::branch(sel, &describe(&cmd)),
                None => fs::create(&describe(&cmd)),
            },
        };
        let fsys = match fsys {
            Ok(fsys) => fsys,
            Err(err) => {
                eprintln!("chimera: cannot open filesystem: {err}");
                return ExitCode::FAILURE;
            }
        };
        // A filesystem the user named is self-evident; one inherited from
        // the environment is not, and a stale exported CHIMERA_FS would
        // otherwise change both what the guest sees and what it mutates
        // with no visible cause.
        if from_env && let Some(sel) = &resume {
            eprintln!("chimera: --in {} (CHIMERA_FS)", Path::new(sel).display());
        }
        // The change-set's lower layer: the live host, or — for a filesystem
        // that branched an image — the image's complete tree, held for the
        // session so removal waits for the run.
        let base = match fs::base_of(&fsys.root).map(|sel| fs::open_base(&sel)) {
            None => None,
            Some(Ok(base)) => Some(base),
            Some(Err(err)) => {
                eprintln!("chimera: {err}");
                return ExitCode::FAILURE;
            }
        };
        let lower: Arc<dyn Vfs> = match &base {
            None => host,
            Some(b) => match HostFs::new(&b.data) {
                Ok(image) => Arc::new(image),
                Err(err) => {
                    eprintln!(
                        "chimera: cannot open image tree {}: {}",
                        b.data.display(),
                        io::Error::from_raw_os_error(err.raw()),
                    );
                    return ExitCode::FAILURE;
                }
            },
        };
        match OverlayFs::new(lower, &fsys.root) {
            Ok(overlay) => (Arc::new(overlay), Some(fsys), base),
            Err(err) => {
                let err = io::Error::from_raw_os_error(err.raw());
                let hint = if err.kind() == io::ErrorKind::Unsupported {
                    " (the delta directory needs a filesystem with user xattrs)"
                } else {
                    ""
                };
                eprintln!(
                    "chimera: cannot open filesystem {}: {err}{hint}",
                    fsys.root.display(),
                );
                return ExitCode::FAILURE;
            }
        }
    };
    // Held for the whole run: dropping the prompt removes the rc file. An
    // image-rooted session gets no badge yet: the rc file lives on the host,
    // and the guest's namespace bottoms out in the image tree, where the
    // file does not exist.
    let _prompt = if implicit_shell && !cmd.no_prompt && base.is_none() {
        match prompt::Prompt::new(fsys.as_ref()) {
            Ok(prompt) => {
                cmd.argv.push("--rcfile".into());
                cmd.argv
                    .push(prompt.rcfile().to_string_lossy().into_owned());
                Some(prompt)
            }
            Err(err) => {
                eprintln!("chimera: cannot configure prompt: {err}");
                None
            }
        }
    } else {
        None
    };
    // Under an image root the host's cwd names nothing in the guest's tree;
    // the session starts at the image's root instead.
    let guest_cwd = if base.is_some() {
        PathBuf::from("/")
    } else {
        env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))
    };
    let (program, args) = cmd.argv.split_first().expect("argv names a program");
    let program = Path::new(program);
    // Resolve the initial executable through the same merged view the
    // guest's own syscalls will see: an inherited change-set may have
    // replaced or deleted the program, its script, or its interpreter, and
    // the session must load the bytes the guest observes, not the lower
    // host's.
    let program = match resolve_program(root.as_ref(), &guest_cwd, program, args) {
        Ok(program) => program,
        Err(err) => {
            eprintln!("chimera: {err}");
            if let Some(fsys) = fsys {
                fsys.finish(cmd.rm);
            }
            return ExitCode::FAILURE;
        }
    };
    let mut sandbox = match Sandbox::new(&program.host_exec) {
        Ok(sandbox) => sandbox,
        Err(err) => {
            eprintln!("chimera: {err}");
            if let Some(fsys) = fsys {
                fsys.finish(cmd.rm);
            }
            return ExitCode::FAILURE;
        }
    };
    // The guest knows the program by its merged-view path; the host_exec
    // backing file is the loader's business, not argv[0]'s.
    sandbox.arg0(&program.exec);
    if let Some(mib) = cmd.code_cache_size {
        sandbox.code_cache_size(mib.saturating_mul(1024 * 1024));
    }

    let mut ns = Namespace::with_root(root, MountFlags::NONE);
    if base.is_some() {
        // The kernel's virtual trees and the device nodes are interfaces to
        // the running host, not content an image tarball could carry; the
        // host serves them over the image tree. A host without one of them
        // has nothing to mount, which is also what the guest should see.
        for point in ["/proc", "/sys", "/dev"] {
            if let Ok(host_tree) = HostFs::new(point) {
                ns.mount(point, Arc::new(host_tree), MountFlags::NONE);
            }
        }
    }
    let personality = Personality::new(ns);
    personality.set_cwd(&guest_cwd);
    personality.set_exe(&program.exec);
    sandbox.system_calls(personality);

    let result = sandbox.args(&program.args).run();
    if let Some(fsys) = fsys {
        fsys.finish(cmd.rm);
    }
    match result {
        Ok(status) => ExitCode::from(status.code() as u8),
        Err(err) => {
            eprintln!("chimera: {err}");
            ExitCode::FAILURE
        }
    }
}

/// The command line a filesystem's provenance records.
fn describe(cmd: &RunCmd) -> String {
    cmd.argv.join(" ")
}

struct Program {
    /// The guest-visible executable path — what `/proc/self/exe` reports.
    exec: PathBuf,
    /// The host file serving `exec` through the merged view; where the ELF
    /// loader actually reads.
    host_exec: PathBuf,
    args: Vec<OsString>,
}

fn resolve_program(
    root: &dyn Vfs,
    cwd: &Path,
    program: &Path,
    args: &[String],
) -> Result<Program, io::Error> {
    let (exec, host_exec) = resolve_path(root, cwd, program)?;
    if let Some((interpreter, interpreter_args)) = read_shebang(&host_exec)? {
        let mut exec_args = interpreter_args;
        // Run shebang scripts through their interpreter so the runtime still
        // only has to execute ELF binaries. The interpreter re-opens the
        // script by its guest-visible name.
        exec_args.push(exec.into_os_string());
        exec_args.extend(args.iter().map(OsString::from));
        let (exec, host_exec) = resolve_path(root, cwd, &interpreter)?;
        return Ok(Program {
            exec,
            host_exec,
            args: exec_args,
        });
    }

    Ok(Program {
        exec,
        host_exec,
        args: args.iter().map(OsString::from).collect(),
    })
}

/// Resolve `program` to its guest-visible absolute path and the host file
/// serving it. A bare name walks `PATH` with each candidate's existence
/// judged in the merged view, so a filesystem whiteout hides a candidate
/// instead of letting the lower host file shadow through.
fn resolve_path(
    root: &dyn Vfs,
    cwd: &Path,
    program: &Path,
) -> Result<(PathBuf, PathBuf), io::Error> {
    if program.is_absolute() || program.components().count() > 1 {
        let exec = absolutize(cwd, program);
        let host = root.host_path(&exec).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("program {:?}: no such file or directory", exec.display()),
            )
        })?;
        return Ok((exec, host));
    }

    let path = env::var_os("PATH").unwrap_or_default();
    for dir in env::split_paths(&path) {
        let candidate = absolutize(cwd, &dir.join(program));
        if let Some(host) = root.host_path(&candidate)
            && host.is_file()
        {
            return Ok((candidate, host));
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("program {:?} not found in PATH", program.display()),
    ))
}

/// The lexically absolute form of `path`, anchored at the guest's initial
/// working directory: the merged view is indexed by absolute guest paths.
fn absolutize(cwd: &Path, path: &Path) -> PathBuf {
    use std::path::Component;

    let mut out = cwd.to_path_buf();
    for c in path.components() {
        match c {
            Component::RootDir => out = PathBuf::from("/"),
            Component::Prefix(_) | Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(n) => out.push(n),
        }
    }
    out
}

fn read_shebang(path: &Path) -> Result<Option<(PathBuf, Vec<OsString>)>, io::Error> {
    let bytes = std::fs::read(path)?;
    if !bytes.starts_with(b"#!") {
        return Ok(None);
    }

    let line_end = bytes
        .iter()
        .position(|&b| b == b'\n')
        .unwrap_or(bytes.len());
    let line = bytes[2..line_end]
        .strip_suffix(b"\r")
        .unwrap_or(&bytes[2..line_end]);
    let line = String::from_utf8_lossy(line);
    let words = shlex::split(&line).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid shebang in {}", path.display()),
        )
    })?;
    let Some((interpreter, args)) = words.split_first() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("empty shebang in {}", path.display()),
        ));
    };

    Ok(Some((
        PathBuf::from(interpreter),
        args.iter().map(OsString::from).collect(),
    )))
}
