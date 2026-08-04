//! The `run` path on a host without the copy-on-write filesystem: the
//! guest resolves its program against the real host and its writes land
//! there, the way `--unsafe` behaves where the overlay does exist.

use std::{
    env,
    ffi::OsString,
    io,
    path::{Path, PathBuf},
    process::ExitCode,
};

use chimera::Sandbox;

use crate::opts::RunCmd;

use super::read_shebang;

pub fn run(cmd: RunCmd) -> ExitCode {
    let Some((program, args)) = cmd.argv.split_first() else {
        eprintln!("chimera: no program to run");
        return ExitCode::FAILURE;
    };
    let program = match resolve_program(Path::new(program), args) {
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
