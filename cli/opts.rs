//! Command-line option parsing for the `chimera` program.

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
    Mount(MountCmd),
    Version(VersionCmd),
    Fs(FsCmd),
}

/// Mount a filesystem's merged view — the live host with the filesystem's
/// changes applied — at a directory, through FUSE. Writes land in the
/// filesystem's change-set, exactly as a run resuming it would leave them.
/// Runs until unmounted (`fusermount -u`, `umount`, or Ctrl-C).
#[derive(FromArgs)]
#[argh(subcommand, name = "mount")]
pub struct MountCmd {
    /// mount read-only: the merged view is browsable but immutable
    #[argh(switch)]
    pub read_only: bool,

    /// filesystem id or path
    #[argh(positional)]
    pub filesystem: String,

    /// directory to mount at
    #[argh(positional)]
    pub mountpoint: String,
}

/// Manage filesystems: the change-sets kept runs leave behind.
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
    Branch(FsBranchCmd),
    Rm(FsRmCmd),
    Prune(FsPruneCmd),
}

/// List kept filesystems.
#[derive(FromArgs)]
#[argh(subcommand, name = "list")]
pub struct FsListCmd {}

/// Show what a filesystem changed, relative to the live host: A added,
/// M modified, D deleted.
#[derive(FromArgs)]
#[argh(subcommand, name = "diff")]
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
pub struct FsApplyCmd {
    /// filesystem id or path
    #[argh(positional)]
    pub filesystem: String,
}

/// Fork a filesystem: a new filesystem seeded with a copy of the source's
/// changes, leaving the source untouched. Prints the new filesystem's id.
#[derive(FromArgs)]
#[argh(subcommand, name = "branch")]
pub struct FsBranchCmd {
    /// source filesystem id or path
    #[argh(positional)]
    pub filesystem: String,
}

/// Remove filesystems.
#[derive(FromArgs)]
#[argh(subcommand, name = "rm")]
pub struct FsRmCmd {
    /// filesystem ids or paths
    #[argh(positional)]
    pub filesystems: Vec<String>,
}

/// Remove every filesystem no live session is using, after confirmation.
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

    /// the branch point: `host` for the live host (the default), a kept
    /// filesystem's id, or a path to a change-set directory — the run stacks
    /// a fresh change-set on it and the source stays untouched
    #[argh(option, short = 'f')]
    pub from: Option<String>,

    /// resume an existing filesystem — an id or a path, never `host` — so
    /// the run's changes accumulate into it (env: CHIMERA_FS)
    #[argh(option, long = "in")]
    pub in_: Option<String>,

    /// discard the branch on exit instead of keeping it; refused with --in
    #[argh(switch)]
    pub rm: bool,

    /// run without a branch: the guest mutates the live host directly
    #[argh(switch, long = "unsafe")]
    pub unsafe_: bool,

    /// leave the default Bash prompt unchanged
    #[argh(switch)]
    pub no_prompt: bool,

    /// the program and its arguments (default: /bin/bash). Greedy: option
    /// parsing stops at the program token, so the guest's own flags need no `--`
    /// separator.
    #[argh(positional, greedy)]
    pub argv: Vec<String>,
}

/// Print version information.
#[derive(FromArgs)]
#[argh(subcommand, name = "version")]
pub struct VersionCmd {}
