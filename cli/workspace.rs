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
    })
}

impl Workspace {
    /// End-of-session disposition. `discard` removes the workspace outright
    /// (`--rm`). Otherwise a fresh workspace whose delta is empty vanishes
    /// silently — `chimera run ls` leaves no residue — and a kept one prints
    /// its one-line notice. Attached workspaces are left exactly as they
    /// are.
    pub fn finish(self, discard: bool) {
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
