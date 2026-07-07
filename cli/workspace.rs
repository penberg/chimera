//! Workspace lifecycle for `chimera run`.
//!
//! A workspace is the persistent unit of a sandbox's changes: the delta
//! directory (`data/` + `tmp/`, the format `chimera-runtime` owns) plus an
//! identity and a small provenance file. A session is one `chimera run`
//! attached to a workspace; sessions come and go, the workspace is what they
//! leave behind. Every plain run creates a fresh workspace — parallel runs
//! are isolated candidate change-sets over the same live tree — and
//! `-w`/`CHIMERA_WORKSPACE` attaches a new session to an existing one.
//!
//! The metadata file is informational only: everything correctness depends
//! on is encoded in the delta tree itself, which is also why two sessions
//! (or one session's forked processes) can share a workspace with no
//! coordination here.

use std::{
    env,
    ffi::OsStr,
    fmt::Write as _,
    fs, io,
    os::unix::ffi::OsStrExt,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

/// One workspace: its short id and the directory holding `data/`, `tmp/`,
/// and the metadata file.
pub struct Workspace {
    pub id: String,
    pub root: PathBuf,
    /// Created fresh by this run — eligible for empty-delta removal and the
    /// kept notice. An attached workspace is the user's to manage.
    fresh: bool,
    /// The session root's pid. Guest fork is a host fork, so every guest
    /// child's host process carries a copy of this struct and returns
    /// through the CLI when its guest exits — but end-of-session
    /// disposition belongs to the root alone, or the first short-lived
    /// child would garbage-collect the live workspace out from under the
    /// session.
    owner: u32,
}

/// Where workspaces live: `$XDG_STATE_HOME/chimera/workspaces`, defaulting
/// to `~/.local/state`.
fn workspaces_dir() -> PathBuf {
    let state = match env::var_os("XDG_STATE_HOME") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => match env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(".local/state"),
            // No home to speak of; keep working rather than refuse to run.
            None => env::temp_dir().join("chimera-state"),
        },
    };
    state.join("chimera/workspaces")
}

/// A fresh workspace with a collision-free generated id.
pub fn create(command: &str) -> io::Result<Workspace> {
    let base = workspaces_dir();
    fs::create_dir_all(&base)?;
    loop {
        let id = fresh_id()?;
        let root = base.join(&id);
        match fs::create_dir(&root) {
            Ok(()) => {
                write_meta(&root, &id, command)?;
                return Ok(Workspace {
                    id,
                    root,
                    fresh: true,
                    owner: std::process::id(),
                });
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

/// Attach to an existing workspace. A selector containing a path separator
/// names the workspace directory itself (created if missing — the escape
/// hatch scripts and the conformance suite use); anything else is an id
/// under the state directory, which must exist.
pub fn attach(selector: &OsStr) -> io::Result<Workspace> {
    if selector.as_bytes().contains(&b'/') {
        let root = PathBuf::from(selector);
        fs::create_dir_all(&root)?;
        return Ok(Workspace {
            id: selector.to_string_lossy().into_owned(),
            root,
            fresh: false,
            owner: std::process::id(),
        });
    }
    let root = workspaces_dir().join(selector);
    if !root.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "no workspace {:?} under {}",
                selector,
                workspaces_dir().display()
            ),
        ));
    }
    Ok(Workspace {
        id: selector.to_string_lossy().into_owned(),
        root,
        fresh: false,
        owner: std::process::id(),
    })
}

impl Workspace {
    /// End-of-session disposition. `discard` removes the workspace outright
    /// (`--rm`). Otherwise a fresh workspace whose delta is empty vanishes
    /// silently — `chimera run ls` leaves no residue — and a kept one prints
    /// its one-line notice. Attached workspaces are left exactly as they
    /// are.
    pub fn finish(self, discard: bool) {
        if std::process::id() != self.owner {
            return;
        }
        if discard {
            let _ = fs::remove_dir_all(&self.root);
            return;
        }
        if !self.fresh {
            return;
        }
        if self.delta_is_empty() {
            let _ = fs::remove_dir_all(&self.root);
            return;
        }
        eprintln!(
            "chimera: workspace {} kept at {} (reattach: chimera run -w {} …)",
            self.id,
            self.root.display(),
            self.id,
        );
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
fn fresh_id() -> io::Result<String> {
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

/// The provenance record: which command created the workspace, from where,
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

// --- `chimera workspace` tooling -------------------------------------------
//
// All of it reads the self-describing on-disk format directly — no daemon,
// no index. The marker predicates and the origin record come from the
// runtime crate, which owns the format.

use std::path::Path;

use chimera::delta::{Origin, is_opaque, is_whiteout, origin};

use crate::opts::{WorkspaceAction, WsApplyCmd, WsDiffCmd, WsRmCmd};

/// Entry point for `chimera workspace <action>`.
pub fn command(action: WorkspaceAction) -> std::process::ExitCode {
    let result = match action {
        WorkspaceAction::List(_) => list(),
        WorkspaceAction::Diff(WsDiffCmd { workspace }) => diff(&workspace),
        WorkspaceAction::Apply(WsApplyCmd { workspace }) => apply(&workspace),
        WorkspaceAction::Rm(WsRmCmd { workspaces }) => rm(&workspaces),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("chimera: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// A workspace named on the command line: an id under the state directory,
/// or a path to a workspace directory. Either way it must exist.
fn resolve(selector: &str) -> io::Result<PathBuf> {
    let root = if selector.contains('/') {
        PathBuf::from(selector)
    } else {
        workspaces_dir().join(selector)
    };
    if !root.join("data").is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no workspace at {}", root.display()),
        ));
    }
    Ok(root)
}

fn list() -> io::Result<()> {
    let base = workspaces_dir();
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
        println!(
            "{id:<10} {age:>5} {:>8}  {}",
            human_size(tree_size(&root.join("data"))),
            field("command ="),
        );
    }
    Ok(())
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

/// One entry of a workspace's change-set, keyed by the guest-visible path.
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

fn rm(selectors: &[String]) -> io::Result<()> {
    for selector in selectors {
        let root = resolve(selector)?;
        fs::remove_dir_all(&root)?;
    }
    Ok(())
}

/// Copy the workspace's changes onto the host. A modified file whose host
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
                    "chimera: conflict: {} changed on the host since the workspace copied it (skipped)",
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

fn apply_change(change: &Change) -> Result<(), ApplyError> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};

    let host = &change.path;
    if change.kind == Kind::Deleted {
        // The guest deleted the name; whether file or directory, it goes.
        match fs::symlink_metadata(host) {
            Ok(md) if md.is_dir() => fs::remove_dir_all(host)?,
            Ok(_) => fs::remove_file(host)?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        return Ok(());
    }

    let md = fs::symlink_metadata(&change.upper)?;
    if md.is_dir() {
        // An opaque directory replaces the lower one wholesale; its children
        // follow in the walk.
        if fs::symlink_metadata(host).is_ok() {
            fs::remove_dir_all(host)?;
        }
        fs::create_dir_all(host)?;
        fs::set_permissions(host, md.permissions())?;
        return Ok(());
    }

    if md.is_file() {
        // The origin check: a copied-up file knows which lower it shadowed.
        // A guest-created file has no origin and simply lands.
        if let Some(o) = origin(&change.upper).map_err(errno_to_io)?
            && let Ok(host_md) = fs::symlink_metadata(host)
        {
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
        if let Some(parent) = host.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&change.upper, host)?;
        return Ok(());
    }

    if md.file_type().is_symlink() {
        let target = fs::read_link(&change.upper)?;
        if fs::symlink_metadata(host).is_ok() {
            fs::remove_file(host)?;
        }
        std::os::unix::fs::symlink(target, host)?;
        return Ok(());
    }

    if md.file_type().is_fifo() {
        let cpath = std::ffi::CString::new(host.as_os_str().as_bytes())
            .map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;
        if fs::symlink_metadata(host).is_ok() {
            fs::remove_file(host)?;
        }
        if unsafe { libc::mkfifo(cpath.as_ptr(), (md.mode() & 0o7777) as libc::mode_t) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        return Ok(());
    }

    // Sockets and device nodes have no meaningful "apply".
    eprintln!("chimera: skipping {} (special file)", change.path.display());
    Ok(())
}
