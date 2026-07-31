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
    Fs(FsCmd),
}

/// Manage workspace filesystems: the change-sets kept runs leave behind.
#[derive(FromArgs)]
#[argh(subcommand, name = "fs")]
pub struct FsCmd {
    #[argh(subcommand)]
    pub action: FsAction,
}

/// `fs` subcommands.
#[derive(FromArgs)]
#[argh(subcommand)]
pub enum FsAction {
    List(FsListCmd),
    Diff(FsDiffCmd),
    Apply(FsApplyCmd),
    Rm(FsRmCmd),
    Prune(FsPruneCmd),
}

/// List kept workspaces.
#[derive(FromArgs)]
#[argh(subcommand, name = "list")]
pub struct FsListCmd {}

/// Show what a workspace changed, relative to the live host: A added,
/// M modified, D deleted.
#[derive(FromArgs)]
#[argh(subcommand, name = "diff")]
pub struct FsDiffCmd {
    /// workspace id or path
    #[argh(positional)]
    pub workspace: String,
}

/// Copy a workspace's changes onto the host — the adopt step. A file whose
/// host copy changed since the workspace copied it up is refused, not
/// clobbered.
#[derive(FromArgs)]
#[argh(subcommand, name = "apply")]
pub struct FsApplyCmd {
    /// workspace id or path
    #[argh(positional)]
    pub workspace: String,
}

/// Remove workspaces.
#[derive(FromArgs)]
#[argh(subcommand, name = "rm")]
pub struct FsRmCmd {
    /// workspace ids or paths
    #[argh(positional)]
    pub workspaces: Vec<String>,
}

/// Remove every workspace no live session is using, after confirmation.
#[derive(FromArgs)]
#[argh(subcommand, name = "prune")]
pub struct FsPruneCmd {
    /// remove without asking for confirmation
    #[argh(switch, short = 'f')]
    pub force: bool,
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
