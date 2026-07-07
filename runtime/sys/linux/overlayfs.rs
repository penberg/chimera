//! [`OverlayFs`]: copy-on-write over the live host. A read-only lower (the
//! host tree) merges with a writable upper (a workspace's delta directory of
//! real host files); lookup consults the upper, then whiteout markers, then
//! the lower.
//!
//! The upper being real files is the load-bearing decision: [`File::host_fd`]
//! keeps working, so file-backed `mmap`, fcntl record locks, `O_APPEND`, and
//! sparse files are all served by the kernel at native speed, and the delta
//! is a plain directory tree a human can read. The upper is itself served
//! through a [`HostFs`] rooted at the delta's `data/` tree, so every handle
//! this filesystem returns is an ordinary [`HostFile`].
//!
//! [`OverlayFs::confines`] is `false`: the namespace walker resolves each
//! component in the merged view, which is the only correct order when an
//! upper symlink can point at a lower file. The walker hands this filesystem
//! paths whose intermediate components are already resolved, so the
//! upper-visibility walk below never crosses a symlink.
//!
//! [`HostFile`]: super::hostfs::HostFile

use std::{
    collections::HashSet,
    ffi::OsString,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use super::{
    delta::{Delta, is_opaque, is_whiteout},
    hostfs::HostFs,
    vfs::{
        DirEntry, Errno, File, FileType, Mode, OpenFlags, RenameFlags, Stat, StatFs, Timespec, Vfs,
        WriteResult,
    },
};

/// How the upper layer answers for a path.
enum Visibility {
    /// An upper entry serves the path. `lower_masked` reports an opaque
    /// ancestor, which cuts the lower out of any directory merge below it.
    Upper { lower_masked: bool },
    /// The upper has nothing on this path; the lower serves it.
    Lower,
    /// A whiteout (or an opaque ancestor over a lower-only tail) deletes the
    /// path from the merged view.
    Hidden,
}

/// A copy-on-write overlay: `upper` (the delta's `data/` tree) over `lower`.
pub struct OverlayFs {
    lower: Arc<dyn Vfs>,
    upper: HostFs,
    delta: Delta,
}

impl OverlayFs {
    /// Overlay a delta directory at `delta_root` (created if missing) over
    /// `lower`. Fails with `ENOTSUP` when the delta's filesystem lacks the
    /// user xattrs the format encodes markers in.
    pub fn new(lower: Arc<dyn Vfs>, delta_root: impl Into<PathBuf>) -> Result<Self, Errno> {
        let delta = Delta::open(delta_root)?;
        let upper = HostFs::new(delta.data_path(Path::new("/")))?;
        Ok(Self {
            lower,
            upper,
            delta,
        })
    }

    /// Walk `rel` down the upper tree and classify who serves it. One lstat
    /// (plus marker probes) per component — correctness first; the optimistic
    /// lower fast path is a measured, later step.
    fn visibility(&self, rel: &Path) -> Result<Visibility, Errno> {
        let mut cur = self.delta.data_path(Path::new("/"));
        let mut lower_masked = false;
        let mut components = rel
            .components()
            .filter_map(|c| match c {
                Component::Normal(n) => Some(n),
                _ => None,
            })
            .peekable();
        while let Some(name) = components.next() {
            cur.push(name);
            let md = match std::fs::symlink_metadata(&cur) {
                Ok(md) => md,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(if lower_masked {
                        Visibility::Hidden
                    } else {
                        Visibility::Lower
                    });
                }
                Err(e) => return Err(Errno::from_io(&e)),
            };
            if is_whiteout(&cur)? {
                // The name is deleted; nothing below it survives either.
                return Ok(Visibility::Hidden);
            }
            let is_final = components.peek().is_none();
            if is_final {
                return Ok(Visibility::Upper { lower_masked });
            }
            if !md.is_dir() {
                // An upper non-directory shadows the lower; descending into
                // it is exactly the kernel's answer. (The walker resolves
                // intermediate symlinks before this filesystem sees them, so
                // a symlink here is unreachable, not followed.)
                return Err(Errno::ENOTDIR);
            }
            if is_opaque(&cur)? {
                lower_masked = true;
            }
        }
        // The root itself: the upper `data/` directory always exists.
        Ok(Visibility::Upper {
            lower_masked: false,
        })
    }

    /// Whether the merged directory at `rel` (already known to be served by
    /// the upper) still folds lower entries in.
    fn merges_lower(&self, rel: &Path, lower_masked: bool) -> Result<bool, Errno> {
        Ok(!lower_masked && !is_opaque(&self.delta.data_path(rel))?)
    }
}

impl Vfs for OverlayFs {
    fn open(&self, path: &Path, flags: OpenFlags, mode: Mode) -> Result<Box<dyn File>, Errno> {
        match self.visibility(path)? {
            Visibility::Upper { lower_masked } => {
                let file = self.upper.open(path, flags, mode)?;
                if file.fstat()?.file_type != FileType::Directory {
                    return Ok(file);
                }
                // A merged directory: fold in the lower's entries unless an
                // opaque marker cuts them off. A lower that has no directory
                // here (or none at all) merges nothing.
                let lower = if self.merges_lower(path, lower_masked)? {
                    self.lower
                        .open(path, OpenFlags(libc::O_RDONLY | libc::O_DIRECTORY), Mode(0))
                        .ok()
                } else {
                    None
                };
                Ok(Box::new(OverlayDir {
                    upper: file,
                    lower,
                    upper_host: self.delta.data_path(path),
                }))
            }
            Visibility::Lower => self.lower.open(path, flags, mode),
            Visibility::Hidden => Err(Errno::ENOENT),
        }
    }

    fn stat(&self, path: &Path, follow: bool) -> Result<Stat, Errno> {
        match self.visibility(path)? {
            Visibility::Upper { .. } => self.upper.stat(path, follow),
            Visibility::Lower => self.lower.stat(path, follow),
            Visibility::Hidden => Err(Errno::ENOENT),
        }
    }

    fn readlink(&self, path: &Path) -> Result<PathBuf, Errno> {
        match self.visibility(path)? {
            Visibility::Upper { .. } => self.upper.readlink(path),
            Visibility::Lower => self.lower.readlink(path),
            Visibility::Hidden => Err(Errno::ENOENT),
        }
    }

    // The write path lands with the copy-up and namespace-mutation tasks;
    // until then the overlay is mounted read-only and these are unreachable.

    fn mkdir(&self, _path: &Path, _mode: Mode) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn unlink(&self, _path: &Path) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn rmdir(&self, _path: &Path) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn symlink(&self, _target: &Path, _link: &Path) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn link(&self, _old: &Path, _new: &Path) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn rename(&self, _from: &Path, _to: &Path, _flags: RenameFlags) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn chmod(&self, _path: &Path, _follow: bool, _mode: Mode) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn chown(&self, _path: &Path, _follow: bool, _uid: u32, _gid: u32) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn utimens(
        &self,
        _path: &Path,
        _follow: bool,
        _times: Option<[Timespec; 2]>,
    ) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn mknod(&self, _path: &Path, _mode: Mode, _dev: u64) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn setxattr(
        &self,
        _path: &Path,
        _follow: bool,
        _name: &std::ffi::OsStr,
        _value: &[u8],
        _flags: i32,
    ) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn removexattr(
        &self,
        _path: &Path,
        _follow: bool,
        _name: &std::ffi::OsStr,
    ) -> Result<(), Errno> {
        Err(Errno::EROFS)
    }

    fn statfs(&self, path: &Path) -> Result<StatFs, Errno> {
        match self.visibility(path)? {
            Visibility::Upper { .. } => self.upper.statfs(path),
            Visibility::Lower => self.lower.statfs(path),
            Visibility::Hidden => Err(Errno::ENOENT),
        }
    }

    fn host_path(&self, path: &Path) -> Option<PathBuf> {
        match self.visibility(path).ok()? {
            // The serving layer answers, so an execve of a modified binary
            // loads the delta's copy, not the host's.
            Visibility::Upper { .. } => Some(self.delta.data_path(path)),
            Visibility::Lower => self.lower.host_path(path),
            Visibility::Hidden => None,
        }
    }

    fn confines(&self) -> bool {
        false
    }
}

/// A merged directory handle: the upper directory (which owns the identity
/// the guest observes) plus the lower directory whose entries fold into
/// `getdents`. Everything but the merge delegates to the upper handle.
struct OverlayDir {
    upper: Box<dyn File>,
    lower: Option<Box<dyn File>>,
    /// The upper directory's host path, where whiteout markers on its entries
    /// live.
    upper_host: PathBuf,
}

impl File for OverlayDir {
    fn pread(&self, buf: &mut [u8], offset: u64) -> Result<usize, Errno> {
        self.upper.pread(buf, offset)
    }

    fn pwrite(&self, buf: &[u8], offset: u64) -> Result<usize, Errno> {
        self.upper.pwrite(buf, offset)
    }

    fn append(&self, buf: &[u8]) -> Result<WriteResult, Errno> {
        self.upper.append(buf)
    }

    fn fstat(&self) -> Result<Stat, Errno> {
        self.upper.fstat()
    }

    fn fstatfs(&self) -> Result<StatFs, Errno> {
        self.upper.fstatfs()
    }

    fn ftruncate(&self, len: u64) -> Result<(), Errno> {
        self.upper.ftruncate(len)
    }

    fn fsync(&self) -> Result<(), Errno> {
        self.upper.fsync()
    }

    fn fchmod(&self, mode: Mode) -> Result<(), Errno> {
        self.upper.fchmod(mode)
    }

    fn fchown(&self, uid: u32, gid: u32) -> Result<(), Errno> {
        self.upper.fchown(uid, gid)
    }

    fn futimens(&self, times: Option<[Timespec; 2]>) -> Result<(), Errno> {
        self.upper.futimens(times)
    }

    fn fsetxattr(&self, name: &std::ffi::OsStr, value: &[u8], flags: i32) -> Result<(), Errno> {
        self.upper.fsetxattr(name, value, flags)
    }

    fn fremovexattr(&self, name: &std::ffi::OsStr) -> Result<(), Errno> {
        self.upper.fremovexattr(name)
    }

    fn fallocate(&self, mode: i32, offset: u64, len: u64) -> Result<(), Errno> {
        self.upper.fallocate(mode, offset, len)
    }

    fn getdents(&self) -> Result<Vec<DirEntry>, Errno> {
        let mut out = Vec::new();
        // An upper name shadows the lower's whether it is live (upper wins)
        // or a whiteout (deleted) — and the whiteout marker itself never
        // appears.
        let mut shadowed: HashSet<OsString> = HashSet::new();
        for e in self.upper.getdents()? {
            let whiteout = is_whiteout(&self.upper_host.join(&e.name))?;
            shadowed.insert(e.name.clone());
            if !whiteout {
                out.push(e);
            }
        }
        if let Some(lower) = &self.lower {
            for e in lower.getdents()? {
                if !shadowed.contains(&e.name) {
                    out.push(e);
                }
            }
        }
        Ok(out)
    }

    fn host_fd(&self) -> Option<std::os::fd::RawFd> {
        self.upper.host_fd()
    }

    fn detach_reservation(&mut self) -> Option<std::os::fd::OwnedFd> {
        self.upper.detach_reservation()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    struct Scratch {
        path: PathBuf,
    }

    impl Scratch {
        fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("chimera-overlay-{}-{}", std::process::id(), n));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn join(&self, rel: &str) -> PathBuf {
            self.path.join(rel)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// An overlay of a fresh delta over a scratch lower tree, plus the delta
    /// handle for planting upper state — or `None` without user xattrs.
    fn overlay(scratch: &Scratch) -> Option<(OverlayFs, Delta)> {
        std::fs::create_dir_all(scratch.join("lower")).unwrap();
        let lower = Arc::new(HostFs::new(scratch.join("lower")).unwrap());
        let fs = match OverlayFs::new(lower, scratch.join("delta")) {
            Ok(fs) => fs,
            Err(Errno(libc::ENOTSUP)) => return None,
            Err(e) => panic!("overlay open failed: {e:?}"),
        };
        let delta = Delta::open(scratch.join("delta")).unwrap();
        Some((fs, delta))
    }

    fn read_all(f: &dyn File) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        let n = f.pread(&mut buf, 0).unwrap();
        buf.truncate(n);
        buf
    }

    fn names(f: &dyn File) -> Vec<String> {
        let mut v: Vec<String> = f
            .getdents()
            .unwrap()
            .into_iter()
            .map(|e| e.name.to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn empty_delta_passes_through() {
        let scratch = Scratch::new();
        let Some((fs, _)) = overlay(&scratch) else {
            return;
        };
        std::fs::write(scratch.join("lower/f"), b"lower").unwrap();
        std::fs::create_dir(scratch.join("lower/d")).unwrap();

        assert_eq!(
            fs.stat(Path::new("/f"), true).unwrap().file_type,
            FileType::Regular
        );
        let f = fs
            .open(Path::new("/f"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(read_all(f.as_ref()), b"lower");
        assert_eq!(
            fs.host_path(Path::new("/f")).unwrap(),
            scratch.join("lower/f").canonicalize().unwrap()
        );

        // The merged root lists the lower's entries.
        let root = fs
            .open(Path::new("/"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(names(root.as_ref()), vec!["d", "f"]);
        assert_eq!(
            fs.stat(Path::new("/absent"), true).unwrap_err(),
            Errno::ENOENT
        );
    }

    #[test]
    fn whiteout_hides_lower() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::write(scratch.join("lower/gone"), b"x").unwrap();
        std::fs::write(scratch.join("lower/kept"), b"y").unwrap();
        delta.whiteout(Path::new("/gone")).unwrap();

        assert_eq!(
            fs.stat(Path::new("/gone"), true).unwrap_err(),
            Errno::ENOENT
        );
        assert_eq!(
            fs.open(Path::new("/gone"), OpenFlags(libc::O_RDONLY), Mode(0))
                .err(),
            Some(Errno::ENOENT)
        );
        assert_eq!(fs.host_path(Path::new("/gone")), None);

        // The name is gone from the listing, the marker file invisible.
        let root = fs
            .open(Path::new("/"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(names(root.as_ref()), vec!["kept"]);
    }

    #[test]
    fn whiteout_hides_whole_lower_subtree() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::create_dir_all(scratch.join("lower/d")).unwrap();
        std::fs::write(scratch.join("lower/d/f"), b"x").unwrap();
        delta.whiteout(Path::new("/d")).unwrap();

        assert_eq!(fs.stat(Path::new("/d"), true).unwrap_err(), Errno::ENOENT);
        assert_eq!(fs.stat(Path::new("/d/f"), true).unwrap_err(), Errno::ENOENT);
    }

    #[test]
    fn upper_wins_and_upper_only_entries_appear() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::write(scratch.join("lower/shared"), b"lower").unwrap();
        delta
            .copy_up(&scratch.join("lower/shared"), Path::new("/shared"))
            .unwrap();
        std::fs::write(delta.data_path(Path::new("/shared")), b"upper").unwrap();
        std::fs::write(delta.data_path(Path::new("/fresh")), b"new").unwrap();

        let f = fs
            .open(Path::new("/shared"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(read_all(f.as_ref()), b"upper");
        assert_eq!(
            fs.host_path(Path::new("/shared")).unwrap(),
            delta.data_path(Path::new("/shared"))
        );

        let root = fs
            .open(Path::new("/"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(names(root.as_ref()), vec!["fresh", "shared"]);
    }

    #[test]
    fn opaque_directory_stops_the_merge() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::create_dir_all(scratch.join("lower/d")).unwrap();
        std::fs::write(scratch.join("lower/d/lower-only"), b"x").unwrap();
        std::fs::create_dir_all(delta.data_path(Path::new("/d"))).unwrap();
        std::fs::write(delta.data_path(Path::new("/d/upper-only")), b"y").unwrap();
        delta.set_opaque(Path::new("/d")).unwrap();

        let d = fs
            .open(Path::new("/d"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(names(d.as_ref()), vec!["upper-only"]);
        // Lookups below the opaque directory mask the lower too.
        assert_eq!(
            fs.stat(Path::new("/d/lower-only"), true).unwrap_err(),
            Errno::ENOENT
        );
        assert_eq!(
            fs.stat(Path::new("/d/upper-only"), true).unwrap().file_type,
            FileType::Regular
        );
    }

    #[test]
    fn merged_directory_unions_both_layers() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::create_dir_all(scratch.join("lower/d")).unwrap();
        std::fs::write(scratch.join("lower/d/a"), b"1").unwrap();
        std::fs::write(scratch.join("lower/d/b"), b"2").unwrap();
        std::fs::create_dir_all(delta.data_path(Path::new("/d"))).unwrap();
        std::fs::write(delta.data_path(Path::new("/d/b")), b"upper").unwrap();
        std::fs::write(delta.data_path(Path::new("/d/c")), b"3").unwrap();
        delta.whiteout(Path::new("/d/a")).unwrap();

        let d = fs
            .open(Path::new("/d"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        // a is whiteouted, b deduplicates (upper wins), c is upper-only.
        assert_eq!(names(d.as_ref()), vec!["b", "c"]);
        let b = fs
            .open(Path::new("/d/b"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(read_all(b.as_ref()), b"upper");
    }

    #[test]
    fn readlink_serves_from_the_owning_layer() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::os::unix::fs::symlink("lower-target", scratch.join("lower/l")).unwrap();
        std::os::unix::fs::symlink("upper-target", delta.data_path(Path::new("/u"))).unwrap();

        assert_eq!(
            fs.readlink(Path::new("/l")).unwrap(),
            PathBuf::from("lower-target")
        );
        assert_eq!(
            fs.readlink(Path::new("/u")).unwrap(),
            PathBuf::from("upper-target")
        );
        assert_eq!(
            fs.stat(Path::new("/u"), false).unwrap().file_type,
            FileType::Symlink
        );
    }
}
