mod opts;

use std::process::ExitCode;

use chimera::Sandbox;

use opts::{Command, Opts, RunCmd};

fn main() -> ExitCode {
    let opts: Opts = argh::from_env();
    match opts.command {
        Command::Run(cmd) => run(cmd),
    }
}

fn run(cmd: RunCmd) -> ExitCode {
    let mut sandbox = match Sandbox::new(&cmd.program) {
        Ok(sandbox) => sandbox,
        Err(err) => {
            eprintln!("chimera: {err}");
            return ExitCode::FAILURE;
        }
    };
    match sandbox.args(cmd.args).run() {
        Ok(status) => ExitCode::from(status.code() as u8),
        Err(err) => {
            eprintln!("chimera: {err}");
            ExitCode::FAILURE
        }
    }
}
