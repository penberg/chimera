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
use crate::prompt;

use super::read_shebang;

pub fn run(mut cmd: RunCmd) -> ExitCode {
    // An empty command line starts the user's shell, the way the overlay
    // path starts bash: `$SHELL` when the environment names one, the
    // platform's default shell otherwise.
    let implicit_shell = cmd.argv.is_empty();
    if implicit_shell {
        cmd.argv
            .push(env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".into()));
    }
    // Same contradiction the overlay path rejects, reported the same way.
    if cmd.unsafe_ && (cmd.from.is_some() || cmd.in_.is_some() || cmd.rm) {
        eprintln!("chimera: --unsafe runs without a filesystem");
        return ExitCode::FAILURE;
    }
    // Refuse rather than proceed: this host has no filesystem to put behind
    // these flags, and running the guest straight against the host as though
    // isolation had been arranged is the one outcome the user did not ask for.
    if cmd.from.is_some() || cmd.in_.is_some() || cmd.rm {
        eprintln!(
            "chimera: --from, --in and --rm need the copy-on-write filesystem, which is built for Linux only"
        );
        return ExitCode::FAILURE;
    }
    // Badge the shell Chimera started itself. There is no filesystem here,
    // so every session is what `--unsafe` is on Linux — writes reach the
    // host — and the prompt says so. Held for the whole run: dropping the
    // prompt removes its startup files.
    let mut zdotdir: Option<PathBuf> = None;
    let _prompt = if implicit_shell && !cmd.no_prompt {
        match prompt::shell_kind(Path::new(&cmd.argv[0])) {
            Some(shell) => match prompt::Prompt::new(&shell, None) {
                Ok(prompt) => {
                    match shell {
                        prompt::Shell::Bash => {
                            cmd.argv.push("--rcfile".into());
                            cmd.argv
                                .push(prompt.rcfile().to_string_lossy().into_owned());
                        }
                        prompt::Shell::Zsh => zdotdir = Some(prompt.dir().to_path_buf()),
                    }
                    Some(prompt)
                }
                Err(err) => {
                    eprintln!("chimera: cannot configure prompt: {err}");
                    None
                }
            },
            None => None,
        }
    } else {
        None
    };
    let (program, args) = cmd.argv.split_first().expect("argv names a program");
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
    // zsh finds its startup files where `$ZDOTDIR` points. `Sandbox::env`
    // replaces inheritance, so the host environment is replayed around it.
    if let Some(dir) = &zdotdir {
        for (key, value) in env::vars_os() {
            if key != "ZDOTDIR" {
                sandbox.env(key, value);
            }
        }
        sandbox.env("ZDOTDIR", dir);
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
