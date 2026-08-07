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
    /// The CLI's stderr, duplicated above the guest fd floor before the
    /// guest runs. The initial process's guest stdio is backed by the CLI's
    /// own descriptors, so a guest that closes its stderr at exit (as every
    /// coreutils program does via `close_stdout`) takes the CLI's fd 2 with
    /// it — and the kept notice must outlive that. `None` when stderr was
    /// already closed at startup: the caller asked for silence.
    stderr: Option<OwnedFd>,
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

use std::path::Path;

use chimera::delta::{
    Origin, is_applied, is_opaque, is_whiteout, mark_applied, origin, record_origin,
};

use crate::opts::{FsAction, FsApplyCmd, FsBranchCmd, FsDiffCmd, FsPruneCmd, FsRmCmd};

/// Entry point for `chimera fs <action>`.
pub fn command(action: FsAction) -> std::process::ExitCode {
    let result = match action {
        FsAction::List(_) => list(),
        FsAction::Diff(FsDiffCmd { filesystem }) => diff(&filesystem),
        FsAction::Apply(FsApplyCmd { filesystem }) => apply(&filesystem),
        FsAction::Branch(FsBranchCmd { filesystem }) => branch_cmd(&filesystem),
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

/// One entry of a filesystem's change-set, keyed by the guest-visible path.
struct Change {
    kind: Kind,
    /// The absolute path the guest saw (and the host path the change is
    /// about — the mount covers `/`).
    path: PathBuf,
    /// The upper file backing the change; empty for a deletion.
    upper: PathBuf,
}

#[derive(PartialEq, Copy, Clone)]
enum Kind {
    Added,
    Modified,
    Deleted,
}

impl Kind {
    fn letter(self) -> char {
        match self {
            Kind::Added => 'A',
            Kind::Modified => 'M',
            Kind::Deleted => 'D',
        }
    }
}

/// Walk a delta's `data/` tree into the change list, parents before
/// children. Directories themselves are listed only when opaque (they
/// replace the lower directory wholesale); an ordinary upper directory is
/// just the scaffolding under its children.
fn changes(root: &Path) -> io::Result<Vec<Change>> {
    let data = root.join("data");
    let mut out = Vec::new();
    walk(&data, Path::new("/"), &mut out)?;
    Ok(out)
}

fn walk(upper_dir: &Path, guest_dir: &Path, out: &mut Vec<Change>) -> io::Result<()> {
    let mut entries: Vec<_> = fs::read_dir(upper_dir)?.filter_map(Result::ok).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let upper = entry.path();
        let guest = guest_dir.join(entry.file_name());
        let md = fs::symlink_metadata(&upper)?;
        if is_whiteout(&upper).map_err(errno_to_io)? {
            out.push(Change {
                kind: Kind::Deleted,
                path: guest,
                upper,
            });
            continue;
        }
        if md.is_dir() {
            if is_opaque(&upper).map_err(errno_to_io)? {
                out.push(Change {
                    kind: kind_against_host(&guest),
                    path: guest.clone(),
                    upper: upper.clone(),
                });
            }
            walk(&upper, &guest, out)?;
            continue;
        }
        out.push(Change {
            kind: kind_against_host(&guest),
            path: guest,
            upper,
        });
    }
    Ok(())
}

/// Added or Modified, judged against the live host.
fn kind_against_host(guest: &Path) -> Kind {
    if fs::symlink_metadata(guest).is_ok() {
        Kind::Modified
    } else {
        Kind::Added
    }
}

fn errno_to_io(e: chimera::Errno) -> io::Error {
    io::Error::from_raw_os_error(e.raw())
}

fn diff(selector: &str) -> io::Result<()> {
    let root = resolve(selector)?;
    for change in changes(&root)? {
        println!("{} {}", change.kind.letter(), change.path.display());
    }
    Ok(())
}

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

/// Copy the filesystem's changes onto the host. A modified file whose host
/// copy no longer matches the origin recorded at copy-up is refused rather
/// than clobbered; everything else applies, and any conflict makes the whole
/// command report failure.
fn apply(selector: &str) -> io::Result<()> {
    let root = resolve(selector)?;
    let mut conflicts = 0u32;
    for change in changes(&root)? {
        match apply_change(&change) {
            Ok(()) => println!("{} {}", change.kind.letter(), change.path.display()),
            Err(ApplyError::Conflict) => {
                conflicts += 1;
                eprintln!(
                    "chimera: conflict: {} changed on the host since the filesystem change (skipped)",
                    change.path.display(),
                );
            }
            Err(ApplyError::Io(e)) => {
                return Err(io::Error::new(
                    e.kind(),
                    format!("applying {}: {e}", change.path.display()),
                ));
            }
        }
    }
    if conflicts > 0 {
        return Err(io::Error::other(format!(
            "{conflicts} conflict(s); resolve on the host and re-run apply"
        )));
    }
    Ok(())
}

enum ApplyError {
    Conflict,
    Io(io::Error),
}

impl From<io::Error> for ApplyError {
    fn from(e: io::Error) -> Self {
        ApplyError::Io(e)
    }
}

/// `CString` from a path's bytes; an embedded NUL is invalid input.
fn cpath(path: &std::path::Path) -> io::Result<std::ffi::CString> {
    std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))
}

/// The xattrs apply replays onto the host: the guest-visible user namespace
/// and the POSIX ACL attributes. Chimera's reserved `user.chimera.*`
/// bookkeeping is delta state and never reaches the host.
fn replayable_xattr(name: &[u8]) -> bool {
    (name.starts_with(b"user.") && !name.starts_with(b"user.chimera."))
        || name == b"system.posix_acl_access"
        || name == b"system.posix_acl_default"
}

fn copy_xattrs(
    upper: &std::ffi::CStr,
    host: &std::ffi::CStr,
    keep: fn(&[u8]) -> bool,
) -> io::Result<()> {
    let mut names = vec![0u8; 1024];
    let len = loop {
        let n = unsafe {
            libc::llistxattr(
                upper.as_ptr(),
                names.as_mut_ptr() as *mut libc::c_char,
                names.len(),
            )
        };
        if n >= 0 {
            break n as usize;
        }
        match io::Error::last_os_error() {
            e if e.raw_os_error() == Some(libc::ERANGE) => names.resize(names.len() * 2, 0),
            // A delta filesystem without xattrs holds none to replay.
            e if e.raw_os_error() == Some(libc::ENOTSUP) => return Ok(()),
            e => return Err(e),
        }
    };
    for name in names[..len].split(|&b| b == 0).filter(|n| !n.is_empty()) {
        if !keep(name) {
            continue;
        }
        let cname =
            std::ffi::CString::new(name).map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;
        let mut value = vec![0u8; 256];
        let vlen = loop {
            let n = unsafe {
                libc::lgetxattr(
                    upper.as_ptr(),
                    cname.as_ptr(),
                    value.as_mut_ptr() as *mut libc::c_void,
                    value.len(),
                )
            };
            if n >= 0 {
                break n as usize;
            }
            match io::Error::last_os_error() {
                e if e.raw_os_error() == Some(libc::ERANGE) => value.resize(value.len() * 2, 0),
                e => return Err(e),
            }
        };
        if unsafe {
            libc::lsetxattr(
                host.as_ptr(),
                cname.as_ptr(),
                value.as_ptr() as *const libc::c_void,
                vlen,
                0,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Reproduce the guest-visible metadata of the upper entry on the object at
/// `host`: ownership first (`chown` may clear set-id bits), then the
/// permission bits, then the xattrs `keep` selects — apply replays only the
/// guest-visible set, a branch copies the `user.chimera.*` bookkeeping too —
/// and timestamps last because every earlier step can disturb them. Symlinks
/// take the no-follow subset — ownership and timestamps. `md` is the upper's
/// metadata captured before the content transfer: reading the upper for the
/// copy already disturbs its atime. A failure here is the command's failure;
/// a partial apply never reads as success.
fn replay_metadata(
    md: &fs::Metadata,
    upper: &std::path::Path,
    host: &std::path::Path,
    keep: fn(&[u8]) -> bool,
) -> io::Result<()> {
    use std::os::unix::fs::MetadataExt;

    let cupper = cpath(upper)?;
    let chost = cpath(host)?;
    if unsafe { libc::lchown(chost.as_ptr(), md.uid(), md.gid()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let symlink = md.file_type().is_symlink();
    if !symlink {
        if unsafe { libc::chmod(chost.as_ptr(), (md.mode() & 0o7777) as libc::mode_t) } != 0 {
            return Err(io::Error::last_os_error());
        }
        copy_xattrs(&cupper, &chost, keep)?;
    }
    let times = [
        libc::timespec {
            tv_sec: md.atime(),
            tv_nsec: md.atime_nsec(),
        },
        libc::timespec {
            tv_sec: md.mtime(),
            tv_nsec: md.mtime_nsec(),
        },
    ];
    if unsafe {
        libc::utimensat(
            libc::AT_FDCWD,
            chost.as_ptr(),
            times.as_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Duplicate one delta entry (a directory recursively) for `branch`. The
/// copy carries everything the format encodes — type, content, ownership,
/// mode, timestamps, and every xattr, the `user.chimera.*` bookkeeping
/// included, since in a branch the bookkeeping is the payload. A directory's
/// metadata lands after its children, whose creation would otherwise disturb
/// its timestamps.
fn copy_entry(from: &Path, to: &Path) -> io::Result<()> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};

    let md = fs::symlink_metadata(from)?;
    let ft = md.file_type();
    if ft.is_dir() {
        fs::create_dir(to)?;
        for entry in fs::read_dir(from)?.filter_map(Result::ok) {
            copy_entry(&entry.path(), &to.join(entry.file_name()))?;
        }
    } else if ft.is_file() {
        fs::copy(from, to)?;
    } else if ft.is_symlink() {
        std::os::unix::fs::symlink(fs::read_link(from)?, to)?;
    } else if ft.is_fifo() {
        let cto = cpath(to)?;
        if unsafe { libc::mkfifo(cto.as_ptr(), 0o600) } != 0 {
            return Err(io::Error::last_os_error());
        }
    } else {
        // A socket, or a device node from a privileged guest; recreating a
        // device needs the same privilege the original creation did.
        let cto = cpath(to)?;
        if unsafe {
            libc::mknod(
                cto.as_ptr(),
                md.mode() as libc::mode_t,
                md.rdev() as libc::dev_t,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    replay_metadata(&md, from, to, |_| true)
}

/// Remove whatever the host has at `path` so a replacement (or deletion) can
/// land. The removal call is selected from the existing target, never from
/// the incoming type — and never through a symlink: a symlink is unlinked
/// itself, only a real directory is removed recursively. An absent target is
/// already removed. A target that changes type between inspection and
/// removal surfaces as the resulting I/O error, not a mis-typed deletion.
fn remove_host(path: &std::path::Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(md) if md.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

fn apply_change(change: &Change) -> Result<(), ApplyError> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};

    let host = &change.path;
    if change.kind == Kind::Deleted {
        // Once this whiteout's removal has been applied, the name belongs
        // to the host again: an entry there now is the host's own and a
        // rerun must refuse it, never delete it twice.
        if is_applied(&change.upper).map_err(errno_to_io)? {
            return match fs::symlink_metadata(host) {
                Ok(_) => Err(ApplyError::Conflict),
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(e.into()),
            };
        }
        // The applied mark lands before the removal: a failure in between
        // surfaces as a conflict on the rerun — the safe direction, since
        // the mark is what guarantees no later host entry gets clobbered.
        mark_applied(&change.upper).map_err(errno_to_io)?;
        remove_host(host)?;
        return Ok(());
    }

    let md = fs::symlink_metadata(&change.upper)?;
    if md.is_dir() {
        // An opaque directory replaces the lower entry wholesale, whatever
        // its type; its children follow in the walk.
        remove_host(host)?;
        fs::create_dir_all(host)?;
        replay_metadata(&md, &change.upper, host, replayable_xattr)?;
        return Ok(());
    }

    if md.is_file() {
        // The origin check, both ways around. A copied-up file knows which
        // lower it shadowed: that lower must still exist and match, or the
        // host has moved on — deletion is as much a host-side change as an
        // edit, and recreating the file would overrule it. A guest-created
        // file has no origin and lands only where the host still has
        // nothing; a host entry that appeared since is equally a conflict.
        // Lookup failures other than NotFound are I/O errors, never
        // permission to apply.
        let host_md = match fs::symlink_metadata(host) {
            Ok(md) => Some(md),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        match (origin(&change.upper).map_err(errno_to_io)?, host_md) {
            (Some(o), Some(host_md)) => {
                let current = Origin {
                    dev: host_md.dev(),
                    ino: host_md.ino(),
                    size: host_md.size(),
                    mtime_sec: host_md.mtime(),
                    mtime_nsec: host_md.mtime_nsec(),
                };
                if current != o {
                    return Err(ApplyError::Conflict);
                }
            }
            (Some(_), None) | (None, Some(_)) => return Err(ApplyError::Conflict),
            (None, None) => {}
        }
        if let Some(parent) = host.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&change.upper, host)?;
        replay_metadata(&md, &change.upper, host, replayable_xattr)?;
        // Advance the origin to the exact host identity this apply produced
        // (after the metadata replay, whose timestamps are part of it): the
        // rerun then recognizes its own work as applied, and anything the
        // host does to the file afterward as a conflict. A failure between
        // the copy and here leaves the stale origin, which the rerun
        // reports as a conflict — never a silent overwrite.
        let applied = fs::symlink_metadata(host)?;
        record_origin(
            &change.upper,
            &Origin {
                dev: applied.dev(),
                ino: applied.ino(),
                size: applied.size(),
                mtime_sec: applied.mtime(),
                mtime_nsec: applied.mtime_nsec(),
            },
        )
        .map_err(errno_to_io)?;
        return Ok(());
    }

    if md.file_type().is_symlink() {
        let target = fs::read_link(&change.upper)?;
        remove_host(host)?;
        std::os::unix::fs::symlink(target, host)?;
        replay_metadata(&md, &change.upper, host, replayable_xattr)?;
        return Ok(());
    }

    if md.file_type().is_fifo() {
        let chost = cpath(host)?;
        remove_host(host)?;
        if unsafe { libc::mkfifo(chost.as_ptr(), (md.mode() & 0o7777) as libc::mode_t) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        replay_metadata(&md, &change.upper, host, replayable_xattr)?;
        return Ok(());
    }

    // Sockets and device nodes have no meaningful "apply".
    eprintln!("chimera: skipping {} (special file)", change.path.display());
    Ok(())
}
