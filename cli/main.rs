mod fs;
mod fuser;
mod mount;
mod opts;
mod prompt;

use std::{
    env,
    ffi::OsString,
    io,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
};

use chimera::{Backend, HostFs, MountFlags, Namespace, OverlayFs, Personality, Sandbox, Vfs};
use mimalloc::MiMalloc;

use opts::{Command, Opts, RunCmd};

/// Route every Chimera-side allocation through mimalloc, whose segments are
/// `mmap`-backed and never touch `brk`, keeping Chimera's heap traffic out of
/// the process's `brk` segment. The guest's own `brk` is virtualized by the
/// runtime, so the real break belongs to the host libc alone.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> ExitCode {
    let opts: Opts = argh::from_env();
    match opts.command {
        Command::Run(cmd) => run(cmd),
        Command::Mount(cmd) => mount::command(cmd),
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

fn run(mut cmd: RunCmd) -> ExitCode {
    let backend = match cmd.backend.as_deref() {
        None | Some("dbt") => Backend::Translate,
        Some("sud") => Backend::SyscallUserDispatch,
        Some(other) => {
            eprintln!("chimera: unknown backend {other:?} (expected dbt or sud)");
            return ExitCode::FAILURE;
        }
    };
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
    let (root, fsys): (Arc<dyn Vfs>, Option<fs::Filesystem>) = if cmd.unsafe_ {
        // The explicit flags contradict --unsafe; the ambient environment
        // variable is merely inert, like any other env a script exports.
        if cmd.from.is_some() || cmd.in_.is_some() || cmd.rm {
            eprintln!("chimera: --unsafe runs without a filesystem");
            return ExitCode::FAILURE;
        }
        (host, None)
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
            eprintln!(
                "chimera: unknown filesystem scheme \"{scheme}:\" (a path whose first component contains a colon needs a leading ./)",
            );
            return ExitCode::FAILURE;
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
        match OverlayFs::new(host, &fsys.root) {
            Ok(overlay) => (Arc::new(overlay), Some(fsys)),
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
    // Held for the whole run: dropping the prompt removes the rc file.
    let _prompt = if implicit_shell && !cmd.no_prompt {
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
    let (program, args) = cmd.argv.split_first().expect("argv names a program");
    let program = Path::new(program);
    // Resolve a bare name through `PATH` with existence judged in the same
    // merged view the guest's own syscalls will see, and report a missing
    // program here, before a session starts. The runtime re-resolves the
    // guest path — shebang splice included — when it installs the image.
    let program = match resolve_program(root.as_ref(), program) {
        Ok(program) => program,
        Err(err) => {
            eprintln!("chimera: {err}");
            if let Some(fsys) = fsys {
                fsys.finish(cmd.rm);
            }
            return ExitCode::FAILURE;
        }
    };
    let mut sandbox = match Sandbox::new(&program) {
        Ok(sandbox) => sandbox,
        Err(err) => {
            eprintln!("chimera: {err}");
            if let Some(fsys) = fsys {
                fsys.finish(cmd.rm);
            }
            return ExitCode::FAILURE;
        }
    };
    sandbox.backend(backend);
    if let Some(mib) = cmd.code_cache_size {
        sandbox.code_cache_size(mib.saturating_mul(1024 * 1024));
    }

    let personality = Personality::new(Namespace::with_root(root, MountFlags::NONE));
    sandbox.system_calls(personality);

    let result = sandbox.args(args).run();
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

/// Resolve `program` to its guest-visible absolute path. A bare name walks
/// `PATH` with each candidate's existence judged in the merged view, so a
/// filesystem whiteout hides a candidate instead of letting the lower host
/// file shadow through.
fn resolve_program(root: &dyn Vfs, program: &Path) -> Result<PathBuf, io::Error> {
    if program.is_absolute() || program.components().count() > 1 {
        let exec = absolutize(program)?;
        root.host_path(&exec).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("program {:?}: no such file or directory", exec.display()),
            )
        })?;
        return Ok(exec);
    }

    let path = env::var_os("PATH").unwrap_or_default();
    for dir in env::split_paths(&path) {
        let candidate = absolutize(&dir.join(program))?;
        if let Some(host) = root.host_path(&candidate)
            && host.is_file()
        {
            return Ok(candidate);
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("program {:?} not found in PATH", program.display()),
    ))
}

/// The lexically absolute form of `path`, anchored at the current directory:
/// the merged view is indexed by absolute guest paths.
fn absolutize(path: &Path) -> Result<PathBuf, io::Error> {
    use std::path::Component;

    let mut out = env::current_dir()?;
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
    Ok(out)
}
