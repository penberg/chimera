//! The [`Namespace`]: the mount tree that maps a guest's absolute path to the
//! [`Vfs`] serving it, resolving symlinks and crossing mountpoints as it walks,
//! and confining the whole walk to the namespace root.
//!
//! This is the confining resolver the Phase 1 stub grew into. It walks a path
//! one component at a time — the only correct order when symlinks are involved,
//! since `..` must act on the symlink-resolved path, not the lexical one — and
//! never lets the resolved path climb above `/`. A symlink whose target is
//! absolute restarts at the namespace root, so a link to `/etc/passwd` inside a
//! confined root resolves *within* that root, not on the host.

use std::{
    collections::VecDeque,
    ffi::OsString,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use super::vfs::{Errno, FileType, Vfs};

/// How many symlinks a single resolution may expand before giving up with
/// `ELOOP`, matching Linux's `MAXSYMLINKS`.
const MAX_SYMLINKS: u32 = 40;

/// Mount options. Today just read-only; the seam for `noexec`, `nosuid`, … later.
#[derive(Clone, Copy, Default)]
pub struct MountFlags {
    pub rdonly: bool,
}

impl MountFlags {
    pub const NONE: MountFlags = MountFlags { rdonly: false };
    pub const RDONLY: MountFlags = MountFlags { rdonly: true };
}

/// One mounted filesystem at a point in the tree.
struct Mount {
    point: PathBuf,
    fs: Arc<dyn Vfs>,
    flags: MountFlags,
}

/// A mount tree. There is always a root mount at `/`; further mounts shadow it
/// along their subtrees, picked by longest matching mount point.
pub struct Namespace {
    mounts: Vec<Mount>,
}

/// The outcome of resolving a guest path: the serving filesystem, the path
/// relative to that filesystem's mount point, whether writes are allowed, and
/// the canonical absolute path within the namespace.
pub struct Resolved {
    pub fs: Arc<dyn Vfs>,
    pub rel: PathBuf,
    pub writable: bool,
    pub abs: PathBuf,
}

impl Namespace {
    /// A namespace with `root` mounted at `/` under `flags`.
    pub fn with_root(root: Arc<dyn Vfs>, flags: MountFlags) -> Self {
        Self {
            mounts: vec![Mount {
                point: PathBuf::from("/"),
                fs: root,
                flags,
            }],
        }
    }

    /// Mount `fs` at the absolute path `point`. A later mount on the same or a
    /// deeper point shadows an earlier one.
    pub fn mount(&mut self, point: impl Into<PathBuf>, fs: Arc<dyn Vfs>, flags: MountFlags) {
        self.mounts.push(Mount {
            point: point.into(),
            fs,
            flags,
        });
    }

    /// Resolve an absolute guest path. `follow_final` chooses whether a symlink
    /// in the last position is dereferenced (`open`/`stat`) or named as itself
    /// (`lstat`/`unlink`/`readlink`). Intermediate symlinks are always followed.
    ///
    /// The path must be absolute; the Personality joins it against the cwd or a
    /// dirfd first. `..` is confined — popping at the root stays at the root.
    pub fn resolve(&self, path: &Path, follow_final: bool) -> Result<Resolved, Errno> {
        let mut pending: VecDeque<Part> = parts(path).into();
        let mut cur: Vec<OsString> = Vec::new();
        let mut symlinks = 0u32;

        while let Some(part) = pending.pop_front() {
            let name = match part {
                Part::Dot => continue,
                Part::DotDot => {
                    cur.pop();
                    continue;
                }
                Part::Name(name) => name,
            };

            let is_final = pending.is_empty();
            let mut probe = cur.clone();
            probe.push(name.clone());
            let abs = join_abs(&probe);
            let (mount, rel) = self.lookup(&abs);

            match mount.fs.stat(&rel, false) {
                Ok(st) if st.file_type == FileType::Symlink && (!is_final || follow_final) => {
                    symlinks += 1;
                    if symlinks > MAX_SYMLINKS {
                        return Err(Errno::ELOOP);
                    }
                    let target = mount.fs.readlink(&rel)?;
                    if target.is_absolute() {
                        cur.clear(); // absolute target restarts at the namespace root
                    }
                    for p in parts(&target).into_iter().rev() {
                        pending.push_front(p);
                    }
                }
                // Exists (and not a symlink we're following): take the component.
                Ok(_) => cur.push(name),
                // A missing *final* component is fine — the caller may be
                // creating it (`open(O_CREAT)`, `mkdir`, `rename` target).
                Err(Errno::ENOENT) if is_final => cur.push(name),
                Err(e) => return Err(e),
            }
        }

        let abs = join_abs(&cur);
        let (mount, rel) = self.lookup(&abs);
        Ok(Resolved {
            fs: Arc::clone(&mount.fs),
            rel,
            writable: !mount.flags.rdonly,
            abs,
        })
    }

    /// The mount serving `abs` (longest matching mount point) and the path
    /// relative to it.
    fn lookup(&self, abs: &Path) -> (&Mount, PathBuf) {
        let mut best = &self.mounts[0]; // the root mount always matches
        for m in &self.mounts {
            if abs.starts_with(&m.point)
                && m.point.components().count() >= best.point.components().count()
            {
                best = m;
            }
        }
        let rest = abs.strip_prefix(&best.point).unwrap_or(Path::new(""));
        let mut rel = PathBuf::from("/");
        rel.push(rest);
        (best, rel)
    }
}

/// A single path step, normalized away from the `Component` enum so a resolved
/// symlink target can be spliced back into the pending queue.
enum Part {
    Dot,
    DotDot,
    Name(OsString),
}

/// Split a path into its steps, dropping the root/prefix markers.
fn parts(path: &Path) -> Vec<Part> {
    path.components()
        .filter_map(|c| match c {
            Component::CurDir => Some(Part::Dot),
            Component::ParentDir => Some(Part::DotDot),
            Component::Normal(n) => Some(Part::Name(n.to_os_string())),
            Component::RootDir | Component::Prefix(_) => None,
        })
        .collect()
}

/// Build an absolute path from already-resolved components.
fn join_abs(components: &[OsString]) -> PathBuf {
    let mut p = PathBuf::from("/");
    p.extend(components);
    p
}

/// Lexically normalize an absolute path (drop `.`, pop on `..`, clamp at root).
/// Used only to seed the initial cwd from the host's; path *resolution* goes
/// through [`Namespace::resolve`], which must handle `..` after symlinks, not
/// lexically.
pub fn normalize(path: &Path) -> PathBuf {
    let mut out: Vec<OsString> = Vec::new();
    for c in path.components() {
        match c {
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(n) => out.push(n.to_os_string()),
        }
    }
    join_abs(&out)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::super::hostfs::HostFs;
    use super::super::vfs::{Mode, OpenFlags};
    use super::*;

    struct Scratch {
        path: PathBuf,
    }

    impl Scratch {
        fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("chimera-ns-{}-{}", std::process::id(), n));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// A namespace whose root is a `HostFs` over a fresh scratch directory.
    fn ns(scratch: &Scratch, flags: MountFlags) -> Namespace {
        let root = Arc::new(HostFs::new(&scratch.path).unwrap());
        Namespace::with_root(root, flags)
    }

    #[test]
    fn dotdot_is_confined_to_the_root() {
        let s = Scratch::new();
        std::fs::create_dir(s.path.join("sub")).unwrap();
        let ns = ns(&s, MountFlags::NONE);
        // Climbing above the root lands back at the root, never outside it.
        let r = ns.resolve(Path::new("/sub/../../../.."), true).unwrap();
        assert_eq!(r.abs, PathBuf::from("/"));
    }

    #[test]
    fn absolute_symlink_is_rerooted_into_the_namespace() {
        let s = Scratch::new();
        std::fs::create_dir(s.path.join("etc")).unwrap();
        std::fs::write(s.path.join("etc/passwd"), b"jailed").unwrap();
        // A link to the *host* /etc/passwd must resolve to the in-root one.
        std::os::unix::fs::symlink("/etc/passwd", s.path.join("escape")).unwrap();
        let ns = ns(&s, MountFlags::NONE);

        let r = ns.resolve(Path::new("/escape"), true).unwrap();
        assert_eq!(r.abs, PathBuf::from("/etc/passwd"));
        let data = {
            let f =
                r.fs.open(&r.rel, OpenFlags(libc::O_RDONLY), Mode(0))
                    .unwrap();
            let mut buf = [0u8; 6];
            f.pread(&mut buf, 0).unwrap();
            buf
        };
        assert_eq!(&data, b"jailed"); // the confined file, not the host's
    }

    #[test]
    fn dangling_intermediate_is_enoent_but_missing_final_is_kept() {
        let s = Scratch::new();
        let ns = ns(&s, MountFlags::NONE);
        assert_eq!(
            ns.resolve(Path::new("/nope/inner"), true).err(),
            Some(Errno::ENOENT)
        );
        // A missing final component resolves (for create), pointing where it would be.
        let r = ns.resolve(Path::new("/newfile"), true).unwrap();
        assert_eq!(r.abs, PathBuf::from("/newfile"));
    }

    #[test]
    fn symlink_loop_is_eloop() {
        let s = Scratch::new();
        std::os::unix::fs::symlink("a", s.path.join("a")).unwrap();
        let ns = ns(&s, MountFlags::NONE);
        assert_eq!(ns.resolve(Path::new("/a"), true).err(), Some(Errno::ELOOP));
    }

    #[test]
    fn readonly_mount_marks_unwritable() {
        let s = Scratch::new();
        std::fs::write(s.path.join("f"), b"x").unwrap();
        let ns = ns(&s, MountFlags::RDONLY);
        assert!(!ns.resolve(Path::new("/f"), true).unwrap().writable);
    }

    #[test]
    fn nofollow_final_names_the_symlink() {
        let s = Scratch::new();
        std::fs::write(s.path.join("target"), b"x").unwrap();
        std::os::unix::fs::symlink("target", s.path.join("link")).unwrap();
        let ns = ns(&s, MountFlags::NONE);

        // follow → the target; no-follow → the link itself.
        assert_eq!(
            ns.resolve(Path::new("/link"), true).unwrap().abs,
            PathBuf::from("/target")
        );
        assert_eq!(
            ns.resolve(Path::new("/link"), false).unwrap().abs,
            PathBuf::from("/link")
        );
    }
}
