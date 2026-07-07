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
/// The state lives behind an `Arc` so open handles can reach back for the
/// copy-up an fd-level mutator on a lower-opened file needs.
pub struct OverlayFs {
    inner: Arc<OverlayInner>,
}

struct OverlayInner {
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
            inner: Arc::new(OverlayInner {
                lower,
                upper,
                delta,
            }),
        })
    }
}

impl OverlayInner {
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

    /// Materialize `rel` in the upper so a mutation can land on it: a regular
    /// file copies up, a directory mirrors (its attribute change must not
    /// touch the lower), a symlink is recreated with the same target. Special
    /// files never copy up — the node lives on the lower and stays there —
    /// so a mutation aimed at one is refused. Idempotent: an existing upper
    /// entry is left alone.
    fn ensure_upper(&self, rel: &Path) -> Result<(), Errno> {
        if std::fs::symlink_metadata(self.delta.data_path(rel)).is_ok() {
            return Ok(());
        }
        let st = self.lower.stat(rel, false)?;
        match st.file_type {
            FileType::Regular => {
                let src = self.lower.host_path(rel).ok_or(Errno::EROFS)?;
                self.delta.copy_up(&src, rel)
            }
            FileType::Directory => {
                std::fs::create_dir_all(self.delta.data_path(rel)).map_err(|e| Errno::from_io(&e))
            }
            FileType::Symlink => {
                let target = self.lower.readlink(rel)?;
                self.delta.materialize_parents(rel)?;
                std::os::unix::fs::symlink(&target, self.delta.data_path(rel))
                    .map_err(|e| Errno::from_io(&e))
            }
            _ => Err(Errno::EROFS),
        }
    }

    /// `ensure_upper` followed by the mutation, the shape every attribute
    /// mutator shares.
    fn mutate_upper(
        &self,
        rel: &Path,
        apply: impl FnOnce(&HostFs) -> Result<(), Errno>,
    ) -> Result<(), Errno> {
        self.ensure_upper(rel)?;
        apply(&self.upper)
    }
}

impl Vfs for OverlayFs {
    fn open(&self, path: &Path, flags: OpenFlags, mode: Mode) -> Result<Box<dyn File>, Errno> {
        let inner = &self.inner;
        match inner.visibility(path)? {
            Visibility::Upper { lower_masked } => {
                let file = inner.upper.open(path, flags, mode)?;
                if file.fstat()?.file_type != FileType::Directory {
                    return Ok(file);
                }
                // A merged directory: fold in the lower's entries unless an
                // opaque marker cuts them off. A lower that has no directory
                // here (or none at all) merges nothing.
                let lower = if inner.merges_lower(path, lower_masked)? {
                    inner
                        .lower
                        .open(path, OpenFlags(libc::O_RDONLY | libc::O_DIRECTORY), Mode(0))
                        .ok()
                } else {
                    None
                };
                Ok(Box::new(OverlayDir {
                    upper: file,
                    lower,
                    upper_host: inner.delta.data_path(path),
                }))
            }
            Visibility::Lower => {
                // Copy-up is eager, at write intent: a write-capable handle
                // always points at an upper file, which is what makes a
                // writable MAP_SHARED mapping trivially correct.
                let writes = flags.writable() || flags.truncate();
                let existing = match inner.lower.stat(path, false) {
                    Ok(st) => Some(st),
                    Err(Errno::ENOENT) => None,
                    Err(e) => return Err(e),
                };
                match existing {
                    Some(_) if flags.create() && flags.excl() => Err(Errno::EEXIST),
                    Some(st) if writes => match st.file_type {
                        FileType::Regular => {
                            let src = inner.lower.host_path(path).ok_or(Errno::EROFS)?;
                            inner.delta.copy_up(&src, path)?;
                            inner.upper.open(path, flags, mode)
                        }
                        // O_TMPFILE writes an anonymous inode into a
                        // directory: it must land in the upper, or the host
                        // filesystem takes the bytes.
                        FileType::Directory if flags.raw() & libc::O_TMPFILE == libc::O_TMPFILE => {
                            inner.ensure_upper(path)?;
                            inner.upper.open(path, flags, mode)
                        }
                        // Writing a special file sends bytes to the object
                        // behind the node, not to the filesystem holding it;
                        // a directory write-open is the kernel's EISDIR to
                        // give. Both pass through.
                        _ => inner.lower.open(path, flags, mode),
                    },
                    Some(st) => {
                        let file = inner.lower.open(path, flags, mode)?;
                        if matches!(st.file_type, FileType::Regular | FileType::Directory) {
                            // Remember the overlay path so an fd-level
                            // mutator on this read-only handle can copy up.
                            Ok(Box::new(OverlayFile {
                                inner: file,
                                fs: Arc::clone(inner),
                                rel: path.to_path_buf(),
                            }))
                        } else {
                            Ok(file)
                        }
                    }
                    None if flags.create() => {
                        // A brand-new file: the merged parents exist (the
                        // walker proved that), so mirror them and create in
                        // the upper.
                        inner.delta.materialize_parents(path)?;
                        inner.upper.open(path, flags, mode)
                    }
                    None => Err(Errno::ENOENT),
                }
            }
            Visibility::Hidden => Err(Errno::ENOENT),
        }
    }

    fn stat(&self, path: &Path, follow: bool) -> Result<Stat, Errno> {
        match self.inner.visibility(path)? {
            Visibility::Upper { .. } => self.inner.upper.stat(path, follow),
            Visibility::Lower => self.inner.lower.stat(path, follow),
            Visibility::Hidden => Err(Errno::ENOENT),
        }
    }

    fn readlink(&self, path: &Path) -> Result<PathBuf, Errno> {
        match self.inner.visibility(path)? {
            Visibility::Upper { .. } => self.inner.upper.readlink(path),
            Visibility::Lower => self.inner.lower.readlink(path),
            Visibility::Hidden => Err(Errno::ENOENT),
        }
    }

    // Namespace mutation (unlink, rename, mkdir, …) lands with the next
    // task; until then those answer EROFS.

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

    fn chmod(&self, path: &Path, follow: bool, mode: Mode) -> Result<(), Errno> {
        self.inner
            .mutate_upper(path, |upper| upper.chmod(path, follow, mode))
    }

    fn chown(&self, path: &Path, follow: bool, uid: u32, gid: u32) -> Result<(), Errno> {
        self.inner
            .mutate_upper(path, |upper| upper.chown(path, follow, uid, gid))
    }

    fn utimens(
        &self,
        path: &Path,
        follow: bool,
        times: Option<[Timespec; 2]>,
    ) -> Result<(), Errno> {
        self.inner
            .mutate_upper(path, |upper| upper.utimens(path, follow, times))
    }

    fn mknod(&self, path: &Path, mode: Mode, dev: u64) -> Result<(), Errno> {
        // The node is a new name: a lower-visible one must answer EEXIST
        // here, because the upper (where the node lands) cannot see it.
        match self.inner.visibility(path)? {
            Visibility::Lower if self.inner.lower.stat(path, false).is_ok() => {
                return Err(Errno::EEXIST);
            }
            Visibility::Hidden => return Err(Errno::ENOENT),
            _ => {}
        }
        self.inner.delta.materialize_parents(path)?;
        self.inner.upper.mknod(path, mode, dev)
    }

    fn setxattr(
        &self,
        path: &Path,
        follow: bool,
        name: &std::ffi::OsStr,
        value: &[u8],
        flags: i32,
    ) -> Result<(), Errno> {
        self.inner.mutate_upper(path, |upper| {
            upper.setxattr(path, follow, name, value, flags)
        })
    }

    fn removexattr(&self, path: &Path, follow: bool, name: &std::ffi::OsStr) -> Result<(), Errno> {
        self.inner
            .mutate_upper(path, |upper| upper.removexattr(path, follow, name))
    }

    fn statfs(&self, path: &Path) -> Result<StatFs, Errno> {
        match self.inner.visibility(path)? {
            Visibility::Upper { .. } => self.inner.upper.statfs(path),
            Visibility::Lower => self.inner.lower.statfs(path),
            Visibility::Hidden => Err(Errno::ENOENT),
        }
    }

    fn host_path(&self, path: &Path) -> Option<PathBuf> {
        match self.inner.visibility(path).ok()? {
            // The serving layer answers, so an execve of a modified binary
            // loads the delta's copy, not the host's.
            Visibility::Upper { .. } => Some(self.inner.delta.data_path(path)),
            Visibility::Lower => self.inner.lower.host_path(path),
            Visibility::Hidden => None,
        }
    }

    fn confines(&self) -> bool {
        false
    }
}

/// A lower file opened without write intent. I/O flows through the lower
/// handle untouched; what the wrapper adds is the overlay path, so an
/// fd-level attribute mutator can copy the file up and apply in the upper.
/// The handle itself keeps serving the lower content afterward — an fd
/// opened before a copy-up sees the frozen lower, the documented v1
/// deviation.
struct OverlayFile {
    inner: Box<dyn File>,
    fs: Arc<OverlayInner>,
    rel: PathBuf,
}

impl File for OverlayFile {
    fn pread(&self, buf: &mut [u8], offset: u64) -> Result<usize, Errno> {
        self.inner.pread(buf, offset)
    }

    fn pwrite(&self, buf: &[u8], offset: u64) -> Result<usize, Errno> {
        self.inner.pwrite(buf, offset)
    }

    fn append(&self, buf: &[u8]) -> Result<WriteResult, Errno> {
        self.inner.append(buf)
    }

    fn fstat(&self) -> Result<Stat, Errno> {
        self.inner.fstat()
    }

    fn fstatfs(&self) -> Result<StatFs, Errno> {
        self.inner.fstatfs()
    }

    fn ftruncate(&self, len: u64) -> Result<(), Errno> {
        // A read-only handle cannot truncate (the kernel answers EINVAL);
        // write-capable handles are always upper, never wrapped.
        self.inner.ftruncate(len)
    }

    fn fsync(&self) -> Result<(), Errno> {
        self.inner.fsync()
    }

    fn fchmod(&self, mode: Mode) -> Result<(), Errno> {
        self.fs
            .mutate_upper(&self.rel, |upper| upper.chmod(&self.rel, false, mode))
    }

    fn fchown(&self, uid: u32, gid: u32) -> Result<(), Errno> {
        self.fs
            .mutate_upper(&self.rel, |upper| upper.chown(&self.rel, false, uid, gid))
    }

    fn futimens(&self, times: Option<[Timespec; 2]>) -> Result<(), Errno> {
        self.fs
            .mutate_upper(&self.rel, |upper| upper.utimens(&self.rel, false, times))
    }

    fn fsetxattr(&self, name: &std::ffi::OsStr, value: &[u8], flags: i32) -> Result<(), Errno> {
        self.fs.mutate_upper(&self.rel, |upper| {
            upper.setxattr(&self.rel, false, name, value, flags)
        })
    }

    fn fremovexattr(&self, name: &std::ffi::OsStr) -> Result<(), Errno> {
        self.fs
            .mutate_upper(&self.rel, |upper| upper.removexattr(&self.rel, false, name))
    }

    fn fallocate(&self, mode: i32, offset: u64, len: u64) -> Result<(), Errno> {
        // Like ftruncate: needs a write-capable fd, which is never wrapped.
        self.inner.fallocate(mode, offset, len)
    }

    fn getdents(&self) -> Result<Vec<DirEntry>, Errno> {
        self.inner.getdents()
    }

    fn host_fd(&self) -> Option<std::os::fd::RawFd> {
        self.inner.host_fd()
    }

    fn detach_reservation(&mut self) -> Option<std::os::fd::OwnedFd> {
        self.inner.detach_reservation()
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
    fn write_open_copies_up_and_leaves_lower_intact() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::write(scratch.join("lower/f"), b"original").unwrap();

        let f = fs
            .open(Path::new("/f"), OpenFlags(libc::O_RDWR), Mode(0))
            .unwrap();
        assert_eq!(f.pwrite(b"MODIFIED", 0).unwrap(), 8);

        // The guest observes the new bytes, the host keeps the old ones, and
        // the upper copy records its origin.
        let g = fs
            .open(Path::new("/f"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(read_all(g.as_ref()), b"MODIFIED");
        assert_eq!(std::fs::read(scratch.join("lower/f")).unwrap(), b"original");
        assert!(
            super::super::delta::origin(&delta.data_path(Path::new("/f")))
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn truncating_open_lands_in_upper() {
        let scratch = Scratch::new();
        let Some((fs, _)) = overlay(&scratch) else {
            return;
        };
        std::fs::write(scratch.join("lower/f"), b"original").unwrap();

        let f = fs
            .open(
                Path::new("/f"),
                OpenFlags(libc::O_WRONLY | libc::O_TRUNC),
                Mode(0),
            )
            .unwrap();
        assert_eq!(f.fstat().unwrap().size, 0);
        assert_eq!(std::fs::read(scratch.join("lower/f")).unwrap(), b"original");
    }

    #[test]
    fn create_lands_in_upper_and_excl_sees_lower() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::create_dir_all(scratch.join("lower/d")).unwrap();
        std::fs::write(scratch.join("lower/existing"), b"x").unwrap();

        // A new file under a lower-only directory: parents mirror, the file
        // lands in the delta, the host directory stays empty.
        let f = fs
            .open(
                Path::new("/d/new"),
                OpenFlags(libc::O_CREAT | libc::O_WRONLY),
                Mode(0o644),
            )
            .unwrap();
        assert_eq!(f.pwrite(b"fresh", 0).unwrap(), 5);
        assert!(delta.data_path(Path::new("/d/new")).is_file());
        assert_eq!(
            std::fs::read_dir(scratch.join("lower/d")).unwrap().count(),
            0
        );

        // O_EXCL must see through to the lower.
        assert_eq!(
            fs.open(
                Path::new("/existing"),
                OpenFlags(libc::O_CREAT | libc::O_EXCL | libc::O_WRONLY),
                Mode(0o644),
            )
            .err(),
            Some(Errno::EEXIST)
        );
    }

    #[test]
    fn attribute_mutators_copy_up() {
        use std::os::unix::fs::PermissionsExt;

        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::write(scratch.join("lower/f"), b"x").unwrap();
        std::fs::set_permissions(
            scratch.join("lower/f"),
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();

        // Path form.
        fs.chmod(Path::new("/f"), true, Mode(0o600)).unwrap();
        assert_eq!(
            fs.stat(Path::new("/f"), true).unwrap().mode.0 & 0o7777,
            0o600
        );
        let host = std::fs::metadata(scratch.join("lower/f")).unwrap();
        assert_eq!(host.permissions().mode() & 0o7777, 0o644);

        // Fd form on a read-only lower handle: copies up, applies in the
        // upper, and the host still keeps its bits.
        std::fs::write(scratch.join("lower/g"), b"y").unwrap();
        std::fs::set_permissions(
            scratch.join("lower/g"),
            std::fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        let g = fs
            .open(Path::new("/g"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        g.fchmod(Mode(0o640)).unwrap();
        assert!(delta.data_path(Path::new("/g")).is_file());
        assert_eq!(
            fs.stat(Path::new("/g"), true).unwrap().mode.0 & 0o7777,
            0o640
        );
        let host = std::fs::metadata(scratch.join("lower/g")).unwrap();
        assert_eq!(host.permissions().mode() & 0o7777, 0o644);
    }

    #[test]
    fn mknod_creates_fifo_in_upper() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::write(scratch.join("lower/taken"), b"x").unwrap();

        fs.mknod(Path::new("/pipe"), Mode(libc::S_IFIFO | 0o644), 0)
            .unwrap();
        assert_eq!(
            fs.stat(Path::new("/pipe"), true).unwrap().file_type,
            FileType::Fifo
        );
        assert!(delta.data_path(Path::new("/pipe")).exists());
        assert!(!scratch.join("lower/pipe").exists());

        // A lower-visible name is EEXIST even though the upper cannot see it.
        assert_eq!(
            fs.mknod(Path::new("/taken"), Mode(libc::S_IFIFO | 0o644), 0),
            Err(Errno::EEXIST)
        );
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
