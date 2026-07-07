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
    Version(VersionCmd),
}

/// Run a program.
#[derive(FromArgs)]
#[argh(subcommand, name = "run")]
pub struct RunCmd {
    /// translated-code cache capacity in MiB (default 256)
    #[argh(option)]
    pub code_cache_size: Option<usize>,

    /// attach to an existing workspace: an id from a kept run, or a path to
    /// a workspace directory (env: CHIMERA_WORKSPACE)
    #[argh(option, short = 'w')]
    pub workspace: Option<String>,

    /// discard the workspace on exit instead of keeping it
    #[argh(switch)]
    pub rm: bool,

    /// bypass the workspace overlay: the guest mutates the host filesystem
    /// directly
    #[argh(switch, long = "unsafe")]
    pub unsafe_: bool,

    /// path to the program
    #[argh(positional)]
    pub program: PathBuf,

    /// arguments to pass to the guest
    #[argh(positional)]
    pub args: Vec<String>,
}

/// Print version information.
#[derive(FromArgs)]
#[argh(subcommand, name = "version")]
pub struct VersionCmd {}
