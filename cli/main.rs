mod fs;

#[cfg(target_os = "linux")]
use std::sync::Arc;
mod opts;

use std::{
    env,
    ffi::OsString,
    io,
    path::{Path, PathBuf},
    process::ExitCode,
};

use argh::FromArgs;
use chimera::Sandbox;
#[cfg(target_os = "linux")]
use chimera::{HostFs, MountFlags, Namespace, OverlayFs, Personality, Vfs};
use mimalloc::MiMalloc;

use opts::{Command, Opts, RunCmd};

/// Route every Chimera-side allocation through mimalloc, whose segments are
/// `mmap`-backed and never touch `brk`. This keeps Chimera's heap clear of the
/// guest libc's `brk`-managed `main_arena`, which shares the one process-wide
/// program break.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> ExitCode {
    let args = insert_guest_separator(env::args().collect());
    let strs: Vec<&str> = args.iter().map(String::as_str).collect();
    let cmd_name = Path::new(strs.first().copied().unwrap_or("chimera"))
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("chimera");
    let opts: Opts = match Opts::from_args(&[cmd_name], strs.get(1..).unwrap_or(&[])) {
        Ok(opts) => opts,
        // Mirror `argh::from_env`: help goes to stdout with success, parse
        // errors to stderr with failure.
        Err(early) => {
            return match early.status {
                Ok(()) => {
                    println!("{}", early.output);
                    ExitCode::SUCCESS
                }
                Err(()) => {
                    eprintln!(
                        "{}\nRun {cmd_name} --help for more information.",
                        early.output
                    );
                    ExitCode::FAILURE
                }
            };
        }
    };
    match opts.command {
        Command::Run(cmd) => run(cmd),
        Command::Version(_) => version(),
        Command::Fs(cmd) => fs::command(cmd.action),
    }
}

/// Insert `--` after the `run` subcommand's program path so the guest's own
/// flags (`chimera run /bin/ls -l`) reach it verbatim instead of being parsed as
/// `chimera`'s options. argh's greedy positional cannot do this alone: it only
/// stops option parsing *after* consuming a non-flag argument, so a guest whose
/// first argument is a flag would still error. The program is the first token
/// after `run` that is neither a `chimera` option nor an option value.
fn insert_guest_separator(mut args: Vec<String>) -> Vec<String> {
    let Some(run_idx) = args.iter().position(|a| a == "run") else {
        return args;
    };
    let mut i = run_idx + 1;
    while i < args.len() {
        match args[i].as_str() {
            "--" => return args,               // already separated by the user
            "--code-cache-size" => i += 2,     // `run` option that takes a value
            a if a.starts_with('-') => i += 1, // any other flag (e.g. --help)
            _ => {
                args.insert(i + 1, "--".to_string());
                return args;
            }
        }
    }
    args
}

fn version() -> ExitCode {
    println!(
        "chimera version {} {} {} {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        if chimera::mpk_enabled() {
            "mpk"
        } else {
            "nompk"
        }
    );
    ExitCode::SUCCESS
}

#[cfg(target_os = "linux")]
fn run(cmd: RunCmd) -> ExitCode {
    // Route the guest's filesystem syscalls through a userspace VFS mounted
    // at `/`. Copy-on-write is the default: every run mounts an overlay
    // whose upper layer is a filesystem's delta, so the guest works against
    // what looks like the live host while every mutation lands in the
    // filesystem. Only `--unsafe` touches the host directly. The host root
    // must exist, so its construction cannot fail.
    let host = Arc::new(HostFs::new("/").expect("host root / is a directory"));
    let selector = cmd
        .fs
        .clone()
        .map(OsString::from)
        .or_else(|| env::var_os("CHIMERA_FS"));
    let (root, fsys): (Arc<dyn Vfs>, Option<fs::Filesystem>) = if cmd.unsafe_ {
        // The explicit flags contradict --unsafe; the ambient environment
        // variable is merely inert, like any other env a script exports.
        if cmd.fs.is_some() || cmd.rm {
            eprintln!("chimera: --unsafe runs without a filesystem");
            return ExitCode::FAILURE;
        }
        (host, None)
    } else {
        let fsys = match &selector {
            Some(sel) => fs::attach(sel),
            None => fs::create(&describe(&cmd)),
        };
        let fsys = match fsys {
            Ok(fsys) => fsys,
            Err(err) => {
                eprintln!("chimera: cannot open filesystem: {err}");
                return ExitCode::FAILURE;
            }
        };
        // An attach the user typed is self-evident; one inherited from the
        // environment is not, and a stale exported CHIMERA_FS would
        // otherwise resurrect old deletions with no visible cause.
        if cmd.fs.is_none() && selector.is_some() {
            eprintln!(
                "chimera: attached to filesystem {} (CHIMERA_FS)",
                fsys.root.display(),
            );
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
    // Resolve the initial executable through the same merged view the
    // guest's own syscalls will see: an attached filesystem may have replaced
    // or deleted the program, its script, or its interpreter, and the session
    // must load the bytes the guest observes, not the lower host's.
    let program = match resolve_program(root.as_ref(), &cmd.program, &cmd.args) {
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
#[cfg(target_os = "linux")]
fn describe(cmd: &RunCmd) -> String {
    let mut s = cmd.program.display().to_string();
    for arg in &cmd.args {
        s.push(' ');
        s.push_str(arg);
    }
    s
}

#[cfg(target_os = "linux")]
struct Program {
    /// The guest-visible executable path — what `/proc/self/exe` reports.
    exec: PathBuf,
    /// The host file serving `exec` through the merged view; where the ELF
    /// loader actually reads.
    host_exec: PathBuf,
    args: Vec<OsString>,
}

#[cfg(target_os = "linux")]
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
#[cfg(target_os = "linux")]
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
#[cfg(not(target_os = "linux"))]
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
    match sandbox.args(&program.args).run() {
        Ok(status) => ExitCode::from(status.code() as u8),
        Err(err) => {
            eprintln!("chimera: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "linux"))]
struct Program {
    exec: PathBuf,
    args: Vec<OsString>,
}

#[cfg(not(target_os = "linux"))]
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

#[cfg(not(target_os = "linux"))]
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

#[cfg(target_os = "linux")]
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
