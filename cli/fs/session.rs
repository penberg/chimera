//! The session lifecycle `chimera run` drives: creating, branching,
//! resuming, and finishing a filesystem. Only the overlay `run` path mounts
//! one, so this half exists only where the copy-on-write filesystem does.

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

use super::delta::copy_entry;
use super::{filesystems_dir, fresh_id, now_secs, resolve};

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
    /// The CLI's stderr, duplicated above the guest fd floor before the
    /// guest runs. The initial process's guest stdio is backed by the CLI's
    /// own descriptors, so a guest that closes its stderr at exit (as every
    /// coreutils program does via `close_stdout`) takes the CLI's fd 2 with
    /// it — and the kept notice must outlive that. `None` when stderr was
    /// already closed at startup: the caller asked for silence.
    stderr: Option<OwnedFd>,
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
                    stderr: stash_stderr(),
                });
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

/// Resume a filesystem named by a path: `selector` is the change-set
/// directory itself, created on first use — how `--in <path>` (and
/// `CHIMERA_FS`, its environment default) pins one change-set across
/// invocations; the conformance suite works this way. A path names raw
/// state, the caller's to manage.
pub fn attach(selector: &OsStr) -> io::Result<Filesystem> {
    let root = PathBuf::from(selector);
    fs::create_dir_all(&root)?;
    let hold = hold(&root)?;
    Ok(Filesystem {
        id: selector.to_string_lossy().into_owned(),
        root,
        fresh: false,
        owner: std::process::id(),
        hold: Some(hold),
        stderr: stash_stderr(),
    })
}

/// Resume a kept filesystem by id: the change-set itself becomes the run's
/// working state, so unlike a branch point it must already exist.
pub fn resume(id: &str) -> io::Result<Filesystem> {
    let root = resolve(id)?;
    let hold = hold(&root)?;
    Ok(Filesystem {
        id: id.to_string(),
        root,
        fresh: false,
        owner: std::process::id(),
        hold: Some(hold),
        stderr: stash_stderr(),
    })
}

/// The reserved scheme of a locator: `<word>:...` names a filesystem that
/// does not live on this machine. A scheme is recognized before path or id —
/// only the first path component is inspected — so introducing one later
/// cannot change how an existing path or id is read; a path whose first
/// component contains a colon needs a leading `./`.
pub fn scheme(selector: &OsStr) -> Option<String> {
    let bytes = selector.as_bytes();
    let first = bytes.split(|&b| b == b'/').next().unwrap_or(b"");
    let colon = first.iter().position(|&b| b == b':')?;
    Some(String::from_utf8_lossy(&first[..colon]).into_owned())
}

/// Duplicate stderr above the guest fd floor. No `FD_CLOEXEC`: emulated
/// guest execve sweeps CLOEXEC-marked host descriptors, and this one must
/// last the session.
fn stash_stderr() -> Option<OwnedFd> {
    let fd = unsafe { libc::fcntl(libc::STDERR_FILENO, libc::F_DUPFD, 512) };
    (fd >= 0).then(|| unsafe { OwnedFd::from_raw_fd(fd) })
}

/// A fresh filesystem seeded with a copy of an existing one's delta: the
/// change-set forks, and the source is left exactly as it was. The copy is
/// marker-exact — whiteouts, opaque marks, and origins carry over — so the
/// branch diffs and applies just as its source would have. A shared hold on
/// the source keeps it from being removed mid-copy; a live session still
/// writing into it can tear the snapshot, so branch from a quiesced source.
pub fn branch(selector: &str, command: &str) -> io::Result<Filesystem> {
    let src = resolve(selector)?;
    let _share = hold(&src)?;
    let fsys = create(command)?;
    let seed = copy_entry(&src.join("data"), &fsys.root.join("data"))
        .and_then(|()| record_parent(&fsys.root, selector));
    if let Err(e) = seed {
        let _ = fs::remove_dir_all(&fsys.root);
        return Err(io::Error::new(
            e.kind(),
            format!("branching {selector}: {e}"),
        ));
    }
    Ok(fsys)
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
    /// End-of-session disposition. `discard` (`--rm`) removes the
    /// filesystem, and is the only thing that does: a fresh filesystem is
    /// kept even when its change-set is empty — the badged prompt advertised
    /// its id for the whole session, and a branch may exist precisely to be
    /// somewhere to stand. A kept fresh filesystem prints the one-line
    /// notice; a resumed one is the user's and is left without comment.
    /// Disposal itself belongs to the last process out of the guest tree
    /// (across every session), so a backgrounded guest keeps its filesystem
    /// usable after the session root has exited.
    pub fn finish(mut self, discard: bool) {
        let owner = std::process::id() == self.owner;
        // Release this process's share before probing: while any other
        // process still holds one, disposal is eventually theirs, not ours.
        drop(self.hold.take());
        if discard {
            if let Some(_disposing) = self.last_out() {
                let _ = fs::remove_dir_all(&self.root);
            }
            return;
        }
        if owner && self.fresh {
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
        let Some(stderr) = &self.stderr else {
            return;
        };
        // No program: `run` starts the shell by itself, so the pasted line
        // lands the user straight back in the filesystem they just left.
        let no_changes = if self.delta_is_empty() {
            " (no changes)"
        } else {
            ""
        };
        let msg = format!(
            "chimera: filesystem kept{no_changes}; continue with:\n  chimera run --in {}\n",
            self.id,
        );
        let _ = unsafe {
            libc::write(
                stderr.as_raw_fd(),
                msg.as_ptr() as *const libc::c_void,
                msg.len(),
            )
        };
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

/// Extend a branch's provenance with the filesystem it forked from.
fn record_parent(root: &std::path::Path, parent: &str) -> io::Result<()> {
    use std::io::Write as _;

    let mut meta = fs::OpenOptions::new()
        .append(true)
        .open(root.join("meta"))?;
    writeln!(meta, "parent = {parent}")
}

// --- `chimera fs` tooling --------------------------------------------------
//
// All of it reads the self-describing on-disk format directly — no daemon,
// no index. The marker predicates and the origin record come from the
// runtime crate, which owns the format.
