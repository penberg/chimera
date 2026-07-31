mod opts;
mod workspace;

use std::{
    env,
    ffi::OsString,
    fs, io,
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
        Command::Workspace(cmd) => workspace::command(cmd.action),
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

fn run(cmd: RunCmd) -> ExitCode {
    // Route the guest's filesystem syscalls through a userspace VFS mounted
    // at `/`. Copy-on-write is the default: every run mounts an overlay
    // whose upper layer is a workspace's delta, so the guest works against
    // what looks like the live host while every mutation lands in the
    // workspace. Only `--unsafe` touches the host directly. The host root
    // must exist, so its construction cannot fail.
    let host = Arc::new(HostFs::new("/").expect("host root / is a directory"));
    let selector = cmd
        .workspace
        .clone()
        .map(OsString::from)
        .or_else(|| env::var_os("CHIMERA_WORKSPACE"));
    let (root, ws): (Arc<dyn Vfs>, Option<workspace::Workspace>) = if cmd.unsafe_ {
        // The explicit flags contradict --unsafe; the ambient environment
        // variable is merely inert, like any other env a script exports.
        if cmd.workspace.is_some() || cmd.rm {
            eprintln!("chimera: --unsafe runs without a workspace");
            return ExitCode::FAILURE;
        }
        (host, None)
    } else {
        let ws = match &selector {
            Some(sel) => workspace::attach(sel),
            None => workspace::create(&describe(&cmd)),
        };
        let ws = match ws {
            Ok(ws) => ws,
            Err(err) => {
                eprintln!("chimera: cannot open workspace: {err}");
                return ExitCode::FAILURE;
            }
        };
        // An attach the user typed is self-evident; one inherited from the
        // environment is not, and a stale exported CHIMERA_WORKSPACE would
        // otherwise resurrect old deletions with no visible cause.
        if cmd.workspace.is_none() && selector.is_some() {
            eprintln!(
                "chimera: attached to workspace {} (CHIMERA_WORKSPACE)",
                ws.root.display(),
            );
        }
        match OverlayFs::new(host, &ws.root) {
            Ok(overlay) => (Arc::new(overlay), Some(ws)),
            Err(err) => {
                let err = io::Error::from_raw_os_error(err.raw());
                let hint = if err.kind() == io::ErrorKind::Unsupported {
                    " (the workspace needs a filesystem with user xattrs)"
                } else {
                    ""
                };
                eprintln!(
                    "chimera: cannot open workspace {}: {err}{hint}",
                    ws.root.display(),
                );
                return ExitCode::FAILURE;
            }
        }
    };
    // Resolve the initial executable through the same merged view the
    // guest's own syscalls will see: an attached workspace may have replaced
    // or deleted the program, its script, or its interpreter, and the session
    // must load the bytes the guest observes, not the lower host's.
    let program = match resolve_program(root.as_ref(), &cmd.program, &cmd.args) {
        Ok(program) => program,
        Err(err) => {
            eprintln!("chimera: {err}");
            if let Some(ws) = ws {
                ws.finish(cmd.rm);
            }
            return ExitCode::FAILURE;
        }
    };
    let mut sandbox = match Sandbox::new(&program.host_exec) {
        Ok(sandbox) => sandbox,
        Err(err) => {
            eprintln!("chimera: {err}");
            if let Some(ws) = ws {
                ws.finish(cmd.rm);
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
    if let Some(ws) = ws {
        ws.finish(cmd.rm);
    }
    match result {
        Ok(status) => ExitCode::from(status.code() as u8),
        Err(err) => {
            eprintln!("chimera: {err}");
            ExitCode::FAILURE
        }
    }
}

/// The command line a workspace's provenance records.
fn describe(cmd: &RunCmd) -> String {
    let mut s = cmd.program.display().to_string();
    for arg in &cmd.args {
        s.push(' ');
        s.push_str(arg);
    }
    s
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
/// judged in the merged view, so a workspace whiteout hides a candidate
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

fn read_shebang(path: &Path) -> Result<Option<(PathBuf, Vec<OsString>)>, io::Error> {
    let bytes = fs::read(path)?;
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
