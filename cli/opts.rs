//! Command-line option parsing for the `chimera` program.

use std::path::PathBuf;

use argh::FromArgs;

/// Run a command in zero-setup sandbox.
#[derive(FromArgs)]
pub struct Opts {
    #[argh(subcommand)]
    pub command: Command,
}

/// Subcommands `chimera` understands.
#[derive(FromArgs)]
#[argh(subcommand)]
pub enum Command {
    Run(RunCmd),
}

/// Run a program.
#[derive(FromArgs)]
#[argh(subcommand, name = "run")]
pub struct RunCmd {
    /// path to the program
    #[argh(positional)]
    pub program: PathBuf,

    /// arguments to pass to the guest
    #[argh(positional)]
    pub args: Vec<String>,
}
