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
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub enum Command {
    Run(RunCmd),
    Version(VersionCmd),
    Fs(FsCmd),
}

/// Manage filesystems: the change-sets kept runs leave behind.
#[derive(FromArgs)]
#[argh(subcommand, name = "fs")]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct FsCmd {
    #[argh(subcommand)]
    pub action: FsAction,
}

/// `fs` subcommands.
#[derive(FromArgs)]
#[argh(subcommand)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub enum FsAction {
    List(FsListCmd),
    Diff(FsDiffCmd),
    Apply(FsApplyCmd),
    Rm(FsRmCmd),
    Prune(FsPruneCmd),
}

/// List kept filesystems.
#[derive(FromArgs)]
#[argh(subcommand, name = "list")]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct FsListCmd {}

/// Show what a filesystem changed, relative to the live host: A added,
/// M modified, D deleted.
#[derive(FromArgs)]
#[argh(subcommand, name = "diff")]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct FsDiffCmd {
    /// filesystem id or path
    #[argh(positional)]
    pub filesystem: String,
}

/// Copy a filesystem's changes onto the host — the adopt step. A file whose
/// host copy changed since the filesystem copied it up is refused, not
/// clobbered.
#[derive(FromArgs)]
#[argh(subcommand, name = "apply")]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct FsApplyCmd {
    /// filesystem id or path
    #[argh(positional)]
    pub filesystem: String,
}

/// Remove filesystems.
#[derive(FromArgs)]
#[argh(subcommand, name = "rm")]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct FsRmCmd {
    /// filesystem ids or paths
    #[argh(positional)]
    pub filesystems: Vec<String>,
}

/// Remove every filesystem no live session is using, after confirmation.
#[derive(FromArgs)]
#[argh(subcommand, name = "prune")]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
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

    /// attach to an existing filesystem: an id from a kept run, or a path to
    /// a filesystem directory (env: CHIMERA_FS)
    #[argh(option, short = 'f')]
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub fs: Option<String>,

    /// discard the filesystem on exit instead of keeping it
    #[argh(switch)]
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub rm: bool,

    /// bypass the copy-on-write filesystem: the guest mutates the host
    /// directly
    #[argh(switch, long = "unsafe")]
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub unsafe_: bool,

    /// path to the program
    #[argh(positional)]
    pub program: PathBuf,

    /// arguments to pass to the guest. `greedy` so flags meant for the guest
    /// (`chimera run /bin/ls -l`) reach it verbatim instead of being parsed as
    /// `chimera`'s own options; put `chimera` options before the program.
    #[argh(positional, greedy)]
    pub args: Vec<String>,
}

/// Print version information.
#[derive(FromArgs)]
#[argh(subcommand, name = "version")]
pub struct VersionCmd {}
