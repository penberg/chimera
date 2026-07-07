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
    let program = match resolve_program(&cmd.program, &cmd.args) {
        Ok(program) => program,
        Err(err) => {
            eprintln!("chimera: {err}");
            return ExitCode::FAILURE;
        }
    };
    let mut sandbox = match Sandbox::new(&program.exec) {
        Ok(sandbox) => sandbox,
        Err(err) => {
            eprintln!("chimera: {err}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(mib) = cmd.code_cache_size {
        sandbox.code_cache_size(mib.saturating_mul(1024 * 1024));
    }

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
            None => workspace::create(&describe(&program)),
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
fn describe(program: &Program) -> String {
    let mut s = program.exec.display().to_string();
    for arg in &program.args {
        s.push(' ');
        s.push_str(&arg.to_string_lossy());
    }
    s
}

struct Program {
    exec: PathBuf,
    args: Vec<OsString>,
}

fn resolve_program(program: &Path, args: &[String]) -> Result<Program, io::Error> {
    let program = resolve_path(program)?;
    if let Some((interpreter, interpreter_args)) = read_shebang(&program)? {
        let mut exec_args = interpreter_args;
        // Run shebang scripts through their interpreter so the runtime still
        // only has to execute ELF binaries.
        exec_args.push(program.as_os_str().to_os_string());
        exec_args.extend(args.iter().map(OsString::from));
        return Ok(Program {
            exec: resolve_path(&interpreter)?,
            args: exec_args,
        });
    }

    Ok(Program {
        exec: program,
        args: args.iter().map(OsString::from).collect(),
    })
}

fn resolve_path(program: &Path) -> Result<PathBuf, io::Error> {
    if program.is_absolute() || program.components().count() > 1 {
        return Ok(program.to_path_buf());
    }

    let path = env::var_os("PATH").unwrap_or_default();
    for dir in env::split_paths(&path) {
        let candidate = dir.join(program);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("program {:?} not found in PATH", program.display()),
    ))
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
