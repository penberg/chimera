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
    Workspace(WorkspaceCmd),
}

/// Manage workspaces: the change-sets kept runs leave behind.
#[derive(FromArgs)]
#[argh(subcommand, name = "workspace")]
pub struct WorkspaceCmd {
    #[argh(subcommand)]
    pub action: WorkspaceAction,
}

/// Workspace subcommands.
#[derive(FromArgs)]
#[argh(subcommand)]
pub enum WorkspaceAction {
    List(WsListCmd),
    Diff(WsDiffCmd),
    Apply(WsApplyCmd),
    Rm(WsRmCmd),
}

/// List kept workspaces.
#[derive(FromArgs)]
#[argh(subcommand, name = "list")]
pub struct WsListCmd {}

/// Show what a workspace changed, relative to the live host: A added,
/// M modified, D deleted.
#[derive(FromArgs)]
#[argh(subcommand, name = "diff")]
pub struct WsDiffCmd {
    /// workspace id or path
    #[argh(positional)]
    pub workspace: String,
}

/// Copy a workspace's changes onto the host — the adopt step. A file whose
/// host copy changed since the workspace copied it up is refused, not
/// clobbered.
#[derive(FromArgs)]
#[argh(subcommand, name = "apply")]
pub struct WsApplyCmd {
    /// workspace id or path
    #[argh(positional)]
    pub workspace: String,
}

/// Remove workspaces.
#[derive(FromArgs)]
#[argh(subcommand, name = "rm")]
pub struct WsRmCmd {
    /// workspace ids or paths
    #[argh(positional)]
    pub workspaces: Vec<String>,
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
