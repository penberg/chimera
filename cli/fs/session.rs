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
    fs, io,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::ffi::OsStrExt,
    },
    path::PathBuf,
};

use super::{filesystems_dir, fresh_id, now_secs};

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
