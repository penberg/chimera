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

use super::vfs::{Errno, FileType, Stat, Vfs};

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
/// relative to that filesystem's mount point, whether writes are allowed, the
/// absolute path within the namespace, and — when the per-component walk already
/// stat'd the final target — its [`Stat`], so the caller need not stat again.
///
/// `abs` is canonical (symlink-resolved) on the walked path and lexically
/// normalized on the confining fast path; either way it is only an anchor for
/// later `dirfd`-relative resolution, which re-resolves through the filesystem.
pub struct Resolved {
    pub fs: Arc<dyn Vfs>,
    pub rel: PathBuf,
    pub writable: bool,
    pub abs: PathBuf,
    pub stat: Option<Stat>,
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
        // Fast path: a single mount of a self-confining filesystem (HostFs)
        // resolves and confines internally, so the per-component walk is
        // unnecessary. Hand it the raw path; the kernel does the resolution the
        // walk would otherwise do one stat at a time. The mount is at `/`, so the
        // mount-relative path is the path itself.
        if self.mounts.len() == 1 && self.mounts[0].fs.confines() {
            let m = &self.mounts[0];
            return Ok(Resolved {
                fs: Arc::clone(&m.fs),
                rel: path.to_path_buf(),
                writable: !m.flags.rdonly,
                abs: normalize(path),
                stat: None,
            });
        }

        let mut pending: VecDeque<Part> = parts(path).into();
        let mut acc = PathBuf::from("/"); // resolved-so-far absolute path
        let mut symlinks = 0u32;
        let mut last = None; // stat of the final component, if it exists

        while let Some(part) = pending.pop_front() {
            let name = match part {
                Part::Dot => continue,
                Part::DotDot => {
                    acc.pop(); // popping at `/` stays at `/` — confined
                    continue;
                }
                Part::Name(name) => name,
            };

            let is_final = pending.is_empty();
            acc.push(&name);
            let (mount, rel) = self.lookup(&acc);

            match mount.fs.stat(&rel, false) {
                Ok(st) if st.file_type == FileType::Symlink && (!is_final || follow_final) => {
                    acc.pop(); // drop the symlink; resolve its target instead
                    symlinks += 1;
                    if symlinks > MAX_SYMLINKS {
                        return Err(Errno::ELOOP);
                    }
                    let target = mount.fs.readlink(&rel)?;
                    if target.is_absolute() {
                        acc = PathBuf::from("/"); // absolute target restarts at the root
                    }
                    for p in parts(&target).into_iter().rev() {
                        pending.push_front(p);
                    }
                    last = None;
                }
                // Exists (and not a symlink we're following): keep the component
                // and remember its stat — the final one is the caller's answer.
                Ok(st) => last = Some(st),
                // A missing *final* component is fine — the caller may be
                // creating it (`open(O_CREAT)`, `mkdir`, `rename` target).
                Err(Errno::ENOENT) if is_final => last = None,
                Err(e) => return Err(e),
            }
        }

        let (mount, rel) = self.lookup(&acc);
        Ok(Resolved {
            fs: Arc::clone(&mount.fs),
            rel,
            writable: !mount.flags.rdonly,
            abs: acc,
            stat: last,
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

    /// A namespace that is a single confining `HostFs` mount — exercises the
    /// fast path (no per-component walk).
    fn ns(scratch: &Scratch, flags: MountFlags) -> Namespace {
        let root = Arc::new(HostFs::new(&scratch.path).unwrap());
        Namespace::with_root(root, flags)
    }

    /// A namespace with a second mount, so `resolve` takes the per-component
    /// walker (the single-confining-mount fast path needs exactly one mount).
    /// These tests pin the walker's behavior; the root still serves every path
    /// here, since nothing lives under the extra mount point.
    fn walked_ns(scratch: &Scratch, flags: MountFlags) -> Namespace {
        let root: Arc<dyn Vfs> = Arc::new(HostFs::new(&scratch.path).unwrap());
        let mut ns = Namespace::with_root(Arc::clone(&root), flags);
        ns.mount("/__force_walk", root, MountFlags::NONE);
        ns
    }

    fn read6(r: &Resolved) -> [u8; 6] {
        let f =
            r.fs.open(&r.rel, OpenFlags(libc::O_RDONLY), Mode(0))
                .unwrap();
        let mut buf = [0u8; 6];
        f.pread(&mut buf, 0).unwrap();
        buf
    }

    // --- walker (forced via a second mount) ---

    #[test]
    fn dotdot_is_confined_to_the_root() {
        let s = Scratch::new();
        std::fs::create_dir(s.path.join("sub")).unwrap();
        let ns = walked_ns(&s, MountFlags::NONE);
        // Climbing above the root lands back at the root, never outside it.
        let r = ns.resolve(Path::new("/sub/../../../.."), true).unwrap();
        assert_eq!(r.abs, PathBuf::from("/"));
    }

    #[test]
    fn absolute_symlink_is_rerooted_into_the_namespace() {
        let s = Scratch::new();
        std::fs::create_dir(s.path.join("etc")).unwrap();
        std::fs::write(s.path.join("etc/passwd"), b"jailed").unwrap();
        std::os::unix::fs::symlink("/etc/passwd", s.path.join("escape")).unwrap();
        let ns = walked_ns(&s, MountFlags::NONE);

        let r = ns.resolve(Path::new("/escape"), true).unwrap();
        assert_eq!(r.abs, PathBuf::from("/etc/passwd"));
        assert_eq!(&read6(&r), b"jailed"); // the confined file, not the host's
    }

    #[test]
    fn dangling_intermediate_is_enoent_but_missing_final_is_kept() {
        let s = Scratch::new();
        let ns = walked_ns(&s, MountFlags::NONE);
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
        let ns = walked_ns(&s, MountFlags::NONE);
        assert_eq!(ns.resolve(Path::new("/a"), true).err(), Some(Errno::ELOOP));
    }

    #[test]
    fn nofollow_final_names_the_symlink() {
        let s = Scratch::new();
        std::fs::write(s.path.join("target"), b"x").unwrap();
        std::os::unix::fs::symlink("target", s.path.join("link")).unwrap();
        let ns = walked_ns(&s, MountFlags::NONE);

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

    #[test]
    fn walker_returns_the_final_stat() {
        let s = Scratch::new();
        std::fs::write(s.path.join("f"), b"x").unwrap();
        let ns = walked_ns(&s, MountFlags::NONE);
        // The walk already stat'd the final component, so the caller can reuse it.
        assert_eq!(
            ns.resolve(Path::new("/f"), true)
                .unwrap()
                .stat
                .unwrap()
                .file_type,
            FileType::Regular
        );
    }

    // --- fast path (single confining mount) ---

    #[test]
    fn readonly_mount_marks_unwritable() {
        let s = Scratch::new();
        std::fs::write(s.path.join("f"), b"x").unwrap();
        let ns = ns(&s, MountFlags::RDONLY);
        assert!(!ns.resolve(Path::new("/f"), true).unwrap().writable);
    }

    #[test]
    fn fast_path_confines_via_hostfs() {
        let s = Scratch::new();
        std::fs::create_dir(s.path.join("etc")).unwrap();
        std::fs::write(s.path.join("etc/passwd"), b"jailed").unwrap();
        std::os::unix::fs::symlink("/etc/passwd", s.path.join("escape")).unwrap();
        let ns = ns(&s, MountFlags::NONE); // single confining mount → fast path

        // An absolute symlink and a climbing `..` both stay inside the root —
        // confinement done by HostFs's openat2(RESOLVE_IN_ROOT), not the walker.
        assert_eq!(
            &read6(&ns.resolve(Path::new("/escape"), true).unwrap()),
            b"jailed"
        );
        assert_eq!(
            &read6(&ns.resolve(Path::new("/../../etc/passwd"), true).unwrap()),
            b"jailed"
        );
    }
}
