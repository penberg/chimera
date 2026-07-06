mod opts;

use std::{
    env,
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
};

use chimera::{HostFs, MountFlags, Namespace, Personality, Sandbox};
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
    }
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

    // Route the guest's filesystem syscalls through a userspace VFS: a host
    // passthrough mounted at `/`, so the guest sees the real tree but every
    // operation crosses the Vfs seam. The mount is read-only by default — the
    // guest can read the host but not change it — and `--unsafe` opts into
    // read-write. The root must exist, so this construction cannot fail.
    let flags = if cmd.unsafe_ {
        MountFlags::NONE
    } else {
        MountFlags::RDONLY
    };
    let root = HostFs::new("/").expect("host root / is a directory");
    let personality = Personality::new(Namespace::with_root(Arc::new(root), flags));
    personality.set_exe(&program.exec);
    sandbox.system_calls(personality);

    match sandbox.args(&program.args).run() {
        Ok(status) => ExitCode::from(status.code() as u8),
        Err(err) => {
            eprintln!("chimera: {err}");
            ExitCode::FAILURE
        }
    }
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
