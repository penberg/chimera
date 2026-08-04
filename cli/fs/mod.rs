//! Filesystem lifecycle for `chimera run`.
//!
//! A filesystem is the persistent unit of a sandbox's changes: the delta
//! directory (`data/` + `tmp/`, the format `chimera-runtime` owns) plus an
//! identity and a small provenance file. A session is one `chimera run`
//! attached to a filesystem; sessions come and go, the filesystem is what they
//! leave behind. Every plain run creates a fresh filesystem — parallel runs
//! are isolated candidate change-sets over the same live tree — and
//! `-f`/`CHIMERA_FS` attaches a new session to an existing one.
//!
//! The metadata file is informational only: everything correctness depends
//! on is encoded in the delta tree itself, which is also why two sessions
//! (or one session's forked processes) can share a filesystem with no
//! coordination here.

use std::{
    env,
    ffi::OsStr,
    fmt::Write as _,
    fs, io,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::ffi::OsStrExt,
    },
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

/// One filesystem: its short id and the directory holding `data/`, `tmp/`,
/// and the metadata file.
pub struct Filesystem {
    pub id: String,
    pub root: PathBuf,
    /// Created fresh by this run — eligible for empty-delta removal and the
    /// kept notice. An attached filesystem is the user's to manage.
    fresh: bool,
    /// The session root's pid. Guest fork is a host fork, so every guest
    /// child's host process carries a copy of this struct and returns
    /// through the CLI when its guest exits. The root alone speaks for the
    /// session (the kept notice); disposal belongs to whoever releases the
    /// hold last.
    owner: u32,
    /// This process's share of the tree-wide hold: a shared `flock` on
    /// `<root>/lock`, one open file description inherited by every host
    /// process of the guest tree. The lock outlives any single process and
    /// ends only when the last inherited descriptor closes, which is
    /// exactly the lifetime disposal must wait for — a counter in process
    /// memory would tear at the first host fork.
    hold: Option<OwnedFd>,
}

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

/// A fresh filesystem with a collision-free generated id.
pub fn create(command: &str) -> io::Result<Filesystem> {
    let base = filesystems_dir();
    fs::create_dir_all(&base)?;
    loop {
        let id = fresh_id()?;
        let root = base.join(&id);
        match fs::create_dir(&root) {
            Ok(()) => {
                write_meta(&root, &id, command)?;
                let hold = hold(&root)?;
                return Ok(Filesystem {
                    id,
                    root,
                    fresh: true,
                    owner: std::process::id(),
                    hold: Some(hold),
                });
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

/// Attach to an existing filesystem. A selector containing a path separator
/// names the filesystem directory itself (created if missing — the escape
/// hatch scripts and the conformance suite use); anything else is an id
/// under the state directory, which must exist.
pub fn attach(selector: &OsStr) -> io::Result<Filesystem> {
    if selector.as_bytes().contains(&b'/') {
        let root = PathBuf::from(selector);
        fs::create_dir_all(&root)?;
        let hold = hold(&root)?;
        return Ok(Filesystem {
            id: selector.to_string_lossy().into_owned(),
            root,
            fresh: false,
            owner: std::process::id(),
            hold: Some(hold),
        });
    }
    let root = filesystems_dir().join(selector);
    if !root.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "no filesystem {:?} under {}",
                selector,
                filesystems_dir().display()
            ),
        ));
    }
    let hold = hold(&root)?;
    Ok(Filesystem {
        id: selector.to_string_lossy().into_owned(),
        root,
        fresh: false,
        owner: std::process::id(),
        hold: Some(hold),
    })
}

/// Acquire this session's share of the tree-wide hold on `root`. The shared
/// lock never contends with other sessions' shares; only a disposal in
/// progress (which holds it exclusively) refuses, so an attach cannot land
/// on a tree that is being removed. The descriptor sits above the runtime's
/// backing-fd floor, out of reach of guest descriptor operations.
fn hold(root: &std::path::Path) -> io::Result<OwnedFd> {
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join("lock"))?;
    // No host FD_CLOEXEC: Chimera emulates guest execve in-process and
    // sweeps CLOEXEC-marked host descriptors as part of it; the hold must
    // survive every guest exec in the tree.
    let high = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_DUPFD, 512) };
    if high < 0 {
        return Err(io::Error::last_os_error());
    }
    let fd = unsafe { OwnedFd::from_raw_fd(high) };
    if unsafe { libc::flock(fd.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) } != 0 {
        let e = io::Error::last_os_error();
        return Err(if e.kind() == io::ErrorKind::WouldBlock {
            io::Error::new(io::ErrorKind::WouldBlock, "filesystem is being removed")
        } else {
            e
        });
    }
    Ok(fd)
}

impl Filesystem {
    /// End-of-session disposition. `discard` removes the filesystem outright
    /// (`--rm`). Otherwise a fresh filesystem whose delta is empty vanishes
    /// silently — `chimera run ls` leaves no residue — and a kept one prints
    /// its one-line notice. Attached filesystems are left exactly as they
    /// are. Disposal itself belongs to the last process out of the guest
    /// tree (across every attached session), so a backgrounded guest keeps
    /// its filesystem usable after the session root has exited.
    pub fn finish(mut self, discard: bool) {
        let owner = std::process::id() == self.owner;
        // Release this process's share before probing: while any other
        // process still holds one, disposal is eventually theirs, not ours.
        drop(self.hold.take());
        if !discard && !self.fresh {
            return;
        }
        let Some(_disposing) = self.last_out() else {
            // The tree lives on. The root still announces a fresh filesystem
            // it is leaving behind non-empty.
            if owner && !discard && self.fresh && !self.delta_is_empty() {
                self.notice();
            }
            return;
        };
        if discard || self.delta_is_empty() {
            let _ = fs::remove_dir_all(&self.root);
            return;
        }
        if owner {
            self.notice();
        }
    }

    /// The disposal probe: taking the lock exclusively succeeds only when
    /// every share of every session's process tree is gone. The returned
    /// guard is held through the removal so a concurrent attach cannot
    /// acquire a share on the tree mid-deletion.
    fn last_out(&self) -> Option<OwnedFd> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(self.root.join("lock"))
            .ok()?;
        let fd = OwnedFd::from(file);
        (unsafe { libc::flock(fd.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0).then_some(fd)
    }

    fn notice(&self) {
        eprintln!("chimera: filesystem kept; continue with:");
        eprintln!("  chimera run -f {}", self.id);
    }

    /// An empty delta means the guest changed nothing: no upper entries at
    /// all. (Staging leftovers in `tmp/` are garbage, not changes.)
    fn delta_is_empty(&self) -> bool {
        match fs::read_dir(self.root.join("data")) {
            Ok(mut entries) => entries.next().is_none(),
            Err(_) => true,
        }
    }
}

/// 8 hex characters of kernel randomness.
pub fn fresh_id() -> io::Result<String> {
    use std::io::Read;

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

/// The provenance record: which command created the filesystem, from where,
/// and when.
fn write_meta(root: &std::path::Path, id: &str, command: &str) -> io::Result<()> {
    let cwd = env::current_dir()
        .map(|d| d.display().to_string())
        .unwrap_or_default();
    fs::write(
        root.join("meta"),
        format!(
            "id = {id}\ncommand = {command}\ncwd = {cwd}\ncreated = {}\n",
            now_secs(),
        ),
    )
}

// --- `chimera fs` tooling --------------------------------------------------
//
// All of it reads the self-describing on-disk format directly — no daemon,
// no index. The marker predicates and the origin record come from the
// runtime crate, which owns the format.

use std::path::Path;

#[cfg(target_os = "linux")]
mod delta;
#[cfg(not(target_os = "linux"))]
#[path = "delta_unsupported.rs"]
mod delta;

use crate::opts::{FsAction, FsPruneCmd, FsRmCmd};
use crate::opts::{FsApplyCmd, FsDiffCmd};

/// Entry point for `chimera fs <action>`.
pub fn command(action: FsAction) -> std::process::ExitCode {
    let result = match action {
        FsAction::List(_) => list(),
        FsAction::Rm(FsRmCmd { filesystems }) => rm(&filesystems),
        FsAction::Prune(FsPruneCmd { force }) => prune(force),
        // Reading a delta means reading its whiteout and origin markers, which
        // are xattrs; replaying one means restoring them. Listing, removing and
        // pruning are directory work and need no such thing.
        FsAction::Diff(FsDiffCmd { filesystem }) => delta::diff(&filesystem),
        FsAction::Apply(FsApplyCmd { filesystem }) => delta::apply(&filesystem),
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
    let mut entries = match fs::read_dir(&base) {
        Ok(entries) => entries.filter_map(Result::ok).collect::<Vec<_>>(),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    entries.sort_by_key(|e| e.file_name());
    if entries.is_empty() {
        return Ok(());
    }
    println!("{:<10} {:>5} {:>8}  COMMAND", "ID", "AGE", "SIZE");
    for entry in entries {
        let root = entry.path();
        if !root.join("data").is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().into_owned();
        println!("{}", row(&id, &root));
    }
    Ok(())
}

/// One filesystem's `list`-format line: id, age, delta size, creating command.
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
    format!(
        "{id:<10} {age:>5} {:>8}  {}",
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
