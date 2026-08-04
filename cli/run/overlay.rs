//! The `run` path on a host with the copy-on-write filesystem: the guest's
//! file syscalls go through a userspace VFS whose upper layer is the
//! session's change-set, so the run leaves the host untouched.

use std::{
    env,
    ffi::OsString,
    io,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
};

use chimera::{HostFs, MountFlags, Namespace, OverlayFs, Personality, Sandbox, Vfs};

use crate::opts::RunCmd;
use crate::{fs, prompt};

use super::read_shebang;

pub fn run(mut cmd: RunCmd) -> ExitCode {
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
    // Resolve the initial executable through the same merged view the
    // guest's own syscalls will see: an inherited change-set may have
    // replaced or deleted the program, its script, or its interpreter, and
    // the session must load the bytes the guest observes, not the lower
    // host's.
    let program = match resolve_program(root.as_ref(), program, args) {
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
    if let Some(mib) = cmd.code_cache_size {
        sandbox.code_cache_size(mib.saturating_mul(1024 * 1024));
    }

    let personality = Personality::new(Namespace::with_root(root, MountFlags::NONE));
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

fn resolve_program(root: &dyn Vfs, program: &Path, args: &[String]) -> Result<Program, io::Error> {
    let (exec, host_exec) = resolve_path(root, program)?;
    if let Some((interpreter, interpreter_args)) = read_shebang(&host_exec)? {
        let mut exec_args = interpreter_args;
        // Run shebang scripts through their interpreter so the runtime still
        // only has to execute ELF binaries. The interpreter re-opens the
        // script by its guest-visible name.
        exec_args.push(exec.into_os_string());
        exec_args.extend(args.iter().map(OsString::from));
        let (exec, host_exec) = resolve_path(root, &interpreter)?;
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
fn resolve_path(root: &dyn Vfs, program: &Path) -> Result<(PathBuf, PathBuf), io::Error> {
    if program.is_absolute() || program.components().count() > 1 {
        let exec = absolutize(program)?;
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
        let candidate = absolutize(&dir.join(program))?;
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
