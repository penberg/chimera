//! Filesystem lifecycle for `chimera run`.
//!
//! A filesystem is the persistent unit of a sandbox's changes: the delta
//! directory (`data/` + `tmp/`, the format `chimera-runtime` owns) plus an
//! identity and a small provenance file. A session is one `chimera run`
//! working in a filesystem; sessions come and go, the filesystem is what they
//! leave behind. Every run works in a filesystem under one of two verbs:
//! `--from` branches — a fresh change-set seeded from the named filesystem,
//! the live host by default, with the source left untouched — and `--in`
//! resumes an existing one, so changes accumulate into it. Parallel branch
//! runs are isolated candidate change-sets over the same live tree. An id
//! names a kept filesystem under the state directory; a path names a
//! change-set directory directly.
//!
//! The metadata file is informational only: everything correctness depends
//! on is encoded in the delta tree itself, which is also why two sessions
//! (or one session's forked processes) can share a filesystem with no
//! coordination here.

use std::{
    env, fs, io,
    os::fd::{AsRawFd, OwnedFd},
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

// A session hands the runtime a mounted overlay, which is built for Linux
// alone; on any other host `run` works against the real host and never
// constructs one. The tooling below stays available everywhere: inspecting
// and removing filesystems needs no overlay.
#[cfg(target_os = "linux")]
mod session;
#[cfg(target_os = "linux")]
pub use session::{Filesystem, attach, branch, create, resume, scheme};

/// Where filesystems live: `$XDG_STATE_HOME/chimera/fs`, defaulting
/// to `~/.local/state`.
fn filesystems_dir() -> PathBuf {
    let state = match env::var_os("XDG_STATE_HOME") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => match env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(".local/state"),
            // No home to speak of; keep working rather than refuse to run.
            None => env::temp_dir().join("chimera-state"),
        },
    };
    state.join("chimera/fs")
}

/// 8 hex characters of kernel randomness.
pub fn fresh_id() -> io::Result<String> {
    use std::{fmt::Write as _, io::Read};

    let mut bytes = [0u8; 4];
    // /dev/urandom cannot reasonably fail, but a pid+time fallback keeps
    // even a degenerate environment running.
    if fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut bytes))
        .is_err()
    {
        let seed = u64::from(std::process::id()) ^ now_secs().wrapping_mul(0x9e3779b97f4a7c15);
        bytes.copy_from_slice(&seed.to_ne_bytes()[..4]);
    }
    let mut id = String::with_capacity(8);
    for b in bytes {
        let _ = write!(id, "{b:02x}");
    }
    Ok(id)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

use std::path::Path;

// Reading and replaying a delta means reading its whiteout and origin
// markers, which are xattrs the runtime's delta format defines on Linux
// alone. The tooling that is only directory work — `list`, `rm`, `prune`,
// `branch` — is served here on every host.
#[cfg(target_os = "linux")]
mod delta;
#[cfg(not(target_os = "linux"))]
#[path = "delta_unsupported.rs"]
mod delta;

use crate::opts::{FsAction, FsApplyCmd, FsBranchCmd, FsDiffCmd, FsPruneCmd, FsRmCmd};

/// Entry point for `chimera fs <action>`.
pub fn command(action: FsAction) -> std::process::ExitCode {
    let result = match action {
        FsAction::List(_) => list(),
        FsAction::Diff(FsDiffCmd { filesystem }) => delta::diff(&filesystem),
        FsAction::Apply(FsApplyCmd { filesystem }) => delta::apply(&filesystem),
        #[cfg(target_os = "linux")]
        FsAction::Branch(FsBranchCmd { filesystem }) => branch_cmd(&filesystem),
        #[cfg(not(target_os = "linux"))]
        FsAction::Branch(FsBranchCmd { filesystem }) => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "branching {filesystem} needs the copy-on-write filesystem, \
                 which is built for Linux only"
            ),
        )),
        FsAction::Rm(FsRmCmd { filesystems }) => rm(&filesystems),
        FsAction::Prune(FsPruneCmd { force }) => prune(force),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("chimera: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// A filesystem named on the command line: an id under the state directory,
/// or a path to a filesystem directory. Either way it must exist.
fn resolve(selector: &str) -> io::Result<PathBuf> {
    let root = if selector.contains('/') {
        PathBuf::from(selector)
    } else {
        filesystems_dir().join(selector)
    };
    if !root.join("data").is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no filesystem at {}", root.display()),
        ));
    }
    Ok(root)
}

fn list() -> io::Result<()> {
    let base = filesystems_dir();
    let entries = match fs::read_dir(&base) {
        Ok(entries) => entries.filter_map(Result::ok).collect::<Vec<_>>(),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    // Newest first: an id is random hex, so recency comes from the
    // provenance record, with the id as an arbitrary but stable tiebreak.
    let mut rows: Vec<(u64, String, PathBuf)> = entries
        .into_iter()
        .filter(|e| e.path().join("data").is_dir())
        .map(|e| {
            let root = e.path();
            let id = e.file_name().to_string_lossy().into_owned();
            (created(&root), id, root)
        })
        .collect();
    if rows.is_empty() {
        return Ok(());
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    println!(
        "{:<10} {:<10} {:>5} {:>8}  COMMAND",
        "ID", "FROM", "AGE", "SIZE"
    );
    for (_, id, root) in rows {
        println!("{}", row(&id, &root));
    }
    Ok(())
}

/// The `created` timestamp from a filesystem's provenance; 0 when absent,
/// which sorts records from before the field existed to the end.
fn created(root: &Path) -> u64 {
    fs::read_to_string(root.join("meta"))
        .ok()
        .and_then(|meta| {
            meta.lines()
                .find_map(|l| l.strip_prefix("created ="))
                .and_then(|v| v.trim().parse().ok())
        })
        .unwrap_or(0)
}

/// One filesystem's `list`-format line: id, branch point, age, delta size, and
/// the creating command. No recorded parent means the run branched the live
/// host, which is what the absence of the record encodes.
fn row(id: &str, root: &Path) -> String {
    let meta = fs::read_to_string(root.join("meta")).unwrap_or_default();
    let field = |name: &str| {
        meta.lines()
            .find_map(|l| l.strip_prefix(name))
            .unwrap_or("")
            .trim()
            .to_string()
    };
    let age = field("created =")
        .parse::<u64>()
        .map(|created| human_age(now_secs().saturating_sub(created)))
        .unwrap_or_default();
    let parent = match field("parent =") {
        p if p.is_empty() => "host".to_string(),
        p => p,
    };
    format!(
        "{id:<10} {parent:<10} {age:>5} {:>8}  {}",
        human_size(tree_size(&root.join("data"))),
        field("command ="),
    )
}

fn tree_size(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .filter_map(Result::ok)
        .map(|e| match e.metadata() {
            Ok(md) if md.is_dir() => tree_size(&e.path()),
            Ok(md) => md.len(),
            Err(_) => 0,
        })
        .sum()
}

fn human_age(secs: u64) -> String {
    match secs {
        0..60 => format!("{secs}s"),
        60..3600 => format!("{}m", secs / 60),
        3600..86400 => format!("{}h", secs / 3600),
        _ => format!("{}d", secs / 86400),
    }
}

fn human_size(bytes: u64) -> String {
    match bytes {
        0..1024 => format!("{bytes}B"),
        1024..1048576 => format!("{}K", bytes / 1024),
        1048576..1073741824 => format!("{}M", bytes / 1048576),
        _ => format!("{}G", bytes / 1073741824),
    }
}

#[cfg(target_os = "linux")]
/// The id on stdout is the whole interface:
/// `chimera run -f $(chimera fs branch <src>) ...` scripts cleanly.
fn branch_cmd(selector: &str) -> io::Result<()> {
    let fsys = branch(selector, &format!("branch of {selector}"))?;
    println!("{}", fsys.id);
    Ok(())
}

/// Take the disposal guard on `root`: the exclusive lock succeeds only when
/// no live session's tree holds a share, and riding through the removal it
/// keeps a concurrent attach from landing mid-deletion. `None` for a
/// filesystem from before the lock existed — no live holders to protect.
/// `WouldBlock` while any session is using the filesystem.
fn disposal_guard(root: &Path) -> io::Result<Option<OwnedFd>> {
    let Ok(file) = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(root.join("lock"))
    else {
        return Ok(None);
    };
    let fd = OwnedFd::from(file);
    if unsafe { libc::flock(fd.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(io::ErrorKind::WouldBlock.into());
    }
    Ok(Some(fd))
}

fn rm(selectors: &[String]) -> io::Result<()> {
    for selector in selectors {
        let root = resolve(selector)?;
        let _guard = disposal_guard(&root).map_err(|_| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("filesystem {selector} is in use"),
            )
        })?;
        fs::remove_dir_all(&root)?;
    }
    Ok(())
}

/// Remove every filesystem under the state directory that no live session
/// holds. The unapplied change-sets go with them, so the candidates are
/// listed and confirmed first unless forced; a declined prompt (or one fed
/// from a closed stdin) removes nothing.
fn prune(force: bool) -> io::Result<()> {
    let mut entries = match fs::read_dir(filesystems_dir()) {
        Ok(entries) => entries.filter_map(Result::ok).collect::<Vec<_>>(),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(e),
    };
    entries.sort_by_key(|e| e.file_name());
    let mut victims = Vec::new();
    for entry in entries {
        let root = entry.path();
        if !root.join("data").is_dir() {
            continue;
        }
        let Ok(guard) = disposal_guard(&root) else {
            continue;
        };
        let id = entry.file_name().to_string_lossy().into_owned();
        victims.push((id, root, guard));
    }
    if victims.is_empty() {
        println!("nothing to prune");
        return Ok(());
    }
    if !force {
        use std::io::Write as _;
        println!("pruning removes these filesystems and their unapplied changes:");
        for (id, root, _) in &victims {
            println!("  {}", row(id, root));
        }
        print!("remove {} filesystem(s)? [y/N] ", victims.len());
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        if !matches!(answer.trim(), "y" | "Y" | "yes") {
            return Ok(());
        }
    }
    let mut freed = 0;
    for (_, root, _guard) in &victims {
        freed += tree_size(&root.join("data"));
        fs::remove_dir_all(root)?;
    }
    println!(
        "removed {} filesystem(s), freed {}",
        victims.len(),
        human_size(freed),
    );
    Ok(())
}
