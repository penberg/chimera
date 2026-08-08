//! [`OverlayFs`]: copy-on-write over the live host. A read-only lower (the
//! host tree) merges with a writable upper (a filesystem's delta directory of
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
    delta::{Delta, is_opaque, is_whiteout, is_whiteout_fd, mark_opaque, may_be_whiteout, origin},
    hostfs::HostFs,
    vfs::{
        DirEntry, Errno, File, FileType, Mode, OpenFlags, RenameFlags, Stat, StatFs, Timespec, Vfs,
        WriteResult,
    },
};

/// `true` for the kernel's virtual trees, whose files are interfaces rather
/// than content and never participate in copy-up.
fn kernel_virtual(rel: &Path) -> bool {
    rel.starts_with("/proc") || rel.starts_with("/sys")
}

/// How the upper layer answers for a path.
enum Visibility {
    /// An upper entry serves the path. `lower_masked` reports an opaque
    /// ancestor, which cuts the lower out of any directory merge below it.
    Upper { lower_masked: bool },
    /// The upper has nothing on this path; the lower serves it.
    Lower,
    /// The final entry is a whiteout: the name reads as deleted, and a create
    /// replaces the marker.
    Whiteout,
    /// The name is absent and an opaque ancestor (or a whiteout above it)
    /// keeps any lower content invisible: reads say ENOENT, creates land in
    /// the upper.
    Masked,
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
            let is_final = components.peek().is_none();
            let md = match std::fs::symlink_metadata(&cur) {
                Ok(md) => md,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(if lower_masked {
                        Visibility::Masked
                    } else {
                        Visibility::Lower
                    });
                }
                Err(e) => return Err(Errno::from_io(&e)),
            };
            if may_be_whiteout(&md) && is_whiteout(&cur)? {
                // The name is deleted; nothing below it survives either.
                return Ok(if is_final {
                    Visibility::Whiteout
                } else {
                    Visibility::Masked
                });
            }
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
    /// file copies up, a directory is deliberately copied up (mirroring the
    /// lower metadata its scaffold-or-absent upper has been presenting, then
    /// claiming the identity with an origin), a symlink is recreated with the
    /// same target. Special files never copy up — the node lives on the lower
    /// and stays there — so a mutation aimed at one is refused. Idempotent:
    /// an upper entry that already owns its identity is left alone.
    fn ensure_upper(&self, rel: &Path) -> Result<(), Errno> {
        let upper = self.delta.data_path(rel);
        if let Ok(md) = std::fs::symlink_metadata(&upper) {
            // A directory that has so far been pure scaffolding is about to
            // take a mutation aimed at itself: claim it, or the change would
            // land invisibly behind the lower metadata stat keeps serving.
            if md.is_dir()
                && matches!(
                    self.visibility(rel)?,
                    Visibility::Upper {
                        lower_masked: false
                    }
                )
                && !is_opaque(&upper)?
                && origin(&upper)?.is_none()
                && let Some(src) = self.lower.host_path(rel)
                && matches!(self.lower.stat(rel, false), Ok(st) if st.file_type == FileType::Directory)
            {
                self.delta.copy_up_dir(&src, rel)?;
            }
            return Ok(());
        }
        let st = self.lower.stat(rel, false)?;
        match st.file_type {
            FileType::Regular => {
                let src = self.lower.host_path(rel).ok_or(Errno::EROFS)?;
                self.delta.copy_up(&src, rel)
            }
            FileType::Directory => {
                let src = self.lower.host_path(rel).ok_or(Errno::EROFS)?;
                self.delta.copy_up_dir(&src, rel)
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

    /// The lower stat a scaffold directory keeps presenting, `None` when the
    /// upper entry genuinely owns the name: a non-directory, a copied-up or
    /// opaque directory, or a directory with no merged lower counterpart.
    fn scaffold_stat(&self, rel: &Path, upper: &Stat, lower_masked: bool) -> Option<Stat> {
        if upper.file_type != FileType::Directory || lower_masked {
            return None;
        }
        let data = self.delta.data_path(rel);
        if is_opaque(&data).ok()? || origin(&data).ok()?.is_some() {
            return None;
        }
        match self.lower.stat(rel, false) {
            Ok(st) if st.file_type == FileType::Directory => Some(st),
            _ => None,
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

    /// Whether the lower still has an entry at `rel` — the test that decides
    /// if a deletion needs a whiteout and a rename must cover its source.
    fn lower_visible(&self, rel: &Path) -> bool {
        self.lower.stat(rel, false).is_ok()
    }

    /// The merged view's stat of `rel`, `None` when the name is absent.
    fn merged_stat(&self, rel: &Path) -> Result<Option<Stat>, Errno> {
        match self.visibility(rel)? {
            Visibility::Upper { .. } => self.upper.stat(rel, false).map(Some),
            Visibility::Lower => match self.lower.stat(rel, false) {
                Ok(st) => Ok(Some(st)),
                Err(Errno::ENOENT) => Ok(None),
                Err(e) => Err(e),
            },
            Visibility::Whiteout | Visibility::Masked => Ok(None),
        }
    }

    /// Drop the whiteout marker at `rel` so a create can take the name over.
    fn remove_marker(&self, rel: &Path) -> Result<(), Errno> {
        std::fs::remove_file(self.delta.data_path(rel)).map_err(|e| Errno::from_io(&e))
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
                    fs: Arc::clone(inner),
                    rel: path.to_path_buf(),
                }))
            }
            Visibility::Lower => {
                // /proc and /sys are kernel interfaces, not filesystem
                // content: a write there is a control operation aimed at the
                // kernel (an oom score, a sysctl), and copying the file up
                // would turn it into a silent no-op against a stale snapshot.
                // Their files pass through whole.
                if kernel_virtual(path) {
                    return inner.lower.open(path, flags, mode);
                }
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
                            let file = inner.upper.open(path, flags, mode)?;
                            // A lost publication continues through whatever
                            // won the name — which a concurrent unlink may
                            // have made a whiteout. The marker on the opened
                            // fd proves the name was deleted by then, and
                            // this open serializes after that deletion.
                            if let Some(fd) = file.host_fd()
                                && is_whiteout_fd(fd)?
                            {
                                return Err(Errno::ENOENT);
                            }
                            Ok(file)
                        }
                        // O_TMPFILE writes an anonymous inode into a
                        // directory: it must land in the upper, or the host
                        // filesystem takes the bytes. The directory itself is
                        // only scaffolding for it — not a claim on its
                        // identity.
                        FileType::Directory if flags.raw() & libc::O_TMPFILE == libc::O_TMPFILE => {
                            inner.delta.scaffold(path)?;
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
            // Creating over a deleted name replaces the whiteout — the lower
            // file it hid must not bleed through, which the fresh upper file
            // now guarantees. (O_EXCL succeeds: the merged view had no name.)
            // The syscall's result is the open descriptor, so this creation
            // cannot be staged; a failure restores the marker instead,
            // compare-and-swap so a concurrent creator keeps the name.
            Visibility::Whiteout if flags.create() => {
                inner.remove_marker(path)?;
                match inner.upper.open(path, flags, mode) {
                    Ok(file) => Ok(file),
                    Err(e) => {
                        let _ = inner.delta.restore_whiteout(path);
                        Err(e)
                    }
                }
            }
            Visibility::Masked if flags.create() => {
                inner.delta.materialize_parents(path)?;
                inner.upper.open(path, flags, mode)
            }
            Visibility::Whiteout | Visibility::Masked => Err(Errno::ENOENT),
        }
    }

    fn stat(&self, path: &Path, follow: bool) -> Result<Stat, Errno> {
        match self.inner.visibility(path)? {
            Visibility::Upper { lower_masked } => {
                let st = self.inner.upper.stat(path, follow)?;
                Ok(self
                    .inner
                    .scaffold_stat(path, &st, lower_masked)
                    .unwrap_or(st))
            }
            Visibility::Lower => self.inner.lower.stat(path, follow),
            Visibility::Whiteout | Visibility::Masked => Err(Errno::ENOENT),
        }
    }

    fn readlink(&self, path: &Path) -> Result<PathBuf, Errno> {
        match self.inner.visibility(path)? {
            Visibility::Upper { .. } => self.inner.upper.readlink(path),
            Visibility::Lower => self.inner.lower.readlink(path),
            Visibility::Whiteout | Visibility::Masked => Err(Errno::ENOENT),
        }
    }

    fn mkdir(&self, path: &Path, mode: Mode) -> Result<(), Errno> {
        let inner = &self.inner;
        match inner.visibility(path)? {
            Visibility::Upper { .. } => inner.upper.mkdir(path, mode),
            Visibility::Lower if inner.lower_visible(path) => Err(Errno::EEXIST),
            Visibility::Lower | Visibility::Masked => {
                inner.delta.materialize_parents(path)?;
                inner.upper.mkdir(path, mode)
            }
            // Over a whiteout the new directory must be opaque — the deleted
            // lower directory’s contents must not merge back in — and the
            // opacity is part of the publication: the directory is built and
            // marked under a staging name, then takes the marker's place in
            // one rename.
            Visibility::Whiteout => inner.delta.replace_whiteout(path, |staging| {
                let mut builder = std::fs::DirBuilder::new();
                std::os::unix::fs::DirBuilderExt::mode(&mut builder, mode.0);
                builder.create(staging).map_err(|e| Errno::from_io(&e))?;
                mark_opaque(staging)
            }),
        }
    }

    fn unlink(&self, path: &Path) -> Result<(), Errno> {
        let inner = &self.inner;
        let st = inner.merged_stat(path)?.ok_or(Errno::ENOENT)?;
        if st.file_type == FileType::Directory {
            return Err(Errno::EISDIR);
        }
        if inner.lower_visible(path) {
            // The whiteout’s staged rename atomically replaces any live
            // upper entry, so the name never flickers back to the lower.
            inner.delta.whiteout(path)
        } else {
            inner.upper.unlink(path)
        }
    }

    fn rmdir(&self, path: &Path) -> Result<(), Errno> {
        let inner = &self.inner;
        let vis = inner.visibility(path)?;
        let st = match &vis {
            Visibility::Upper { .. } => inner.upper.stat(path, false)?,
            Visibility::Lower => inner.lower.stat(path, false)?,
            Visibility::Whiteout | Visibility::Masked => return Err(Errno::ENOENT),
        };
        if st.file_type != FileType::Directory {
            return Err(Errno::ENOTDIR);
        }
        // Emptiness is a merged-view question: whiteouts hide their lower
        // names, so a directory of nothing but markers is empty.
        let dir = self.open(path, OpenFlags(libc::O_RDONLY | libc::O_DIRECTORY), Mode(0))?;
        if !dir.getdents()?.is_empty() {
            return Err(Errno::ENOTEMPTY);
        }
        if let Visibility::Upper { .. } = vis {
            // Only whiteout markers can remain inside; clear them with the
            // directory.
            std::fs::remove_dir_all(inner.delta.data_path(path)).map_err(|e| Errno::from_io(&e))?;
        }
        if inner.lower_visible(path) {
            inner.delta.whiteout(path)?;
        }
        Ok(())
    }

    fn symlink(&self, target: &Path, link: &Path) -> Result<(), Errno> {
        let inner = &self.inner;
        match inner.visibility(link)? {
            Visibility::Upper { .. } => inner.upper.symlink(target, link),
            Visibility::Lower if inner.lower_visible(link) => Err(Errno::EEXIST),
            Visibility::Lower | Visibility::Masked => {
                inner.delta.materialize_parents(link)?;
                inner.upper.symlink(target, link)
            }
            Visibility::Whiteout => inner.delta.replace_whiteout(link, |staging| {
                std::os::unix::fs::symlink(target, staging).map_err(|e| Errno::from_io(&e))
            }),
        }
    }

    fn link(&self, old: &Path, new: &Path) -> Result<(), Errno> {
        let inner = &self.inner;
        // The new name must link to a real inode, so the source materializes
        // in the upper first. (A lower hardlink pair breaks here — the
        // documented v1 deviation.)
        inner.ensure_upper(old)?;
        match inner.visibility(new)? {
            Visibility::Upper { .. } => inner.upper.link(old, new),
            Visibility::Lower if inner.lower_visible(new) => Err(Errno::EEXIST),
            Visibility::Lower | Visibility::Masked => {
                inner.delta.materialize_parents(new)?;
                inner.upper.link(old, new)
            }
            Visibility::Whiteout => inner.delta.replace_whiteout(new, |staging| {
                std::fs::hard_link(inner.delta.data_path(old), staging)
                    .map_err(|e| Errno::from_io(&e))
            }),
        }
    }

    fn rename(&self, from: &Path, to: &Path, flags: RenameFlags) -> Result<(), Errno> {
        let inner = &self.inner;
        let src = inner.merged_stat(from)?.ok_or(Errno::ENOENT)?;
        let dst = inner.merged_stat(to)?;
        if flags.noreplace() && dst.is_some() {
            return Err(Errno::EEXIST);
        }
        let src_dir = src.file_type == FileType::Directory;
        if src_dir {
            // Only a directory that lives purely in the upper can move as a
            // rename; anything lower-visible would need a redirect the format
            // does not carry, so the guest gets EXDEV and its libc falls back
            // to copy+delete, as across any mount.
            let upper_only = matches!(inner.visibility(from)?, Visibility::Upper { .. })
                && !inner.lower_visible(from);
            if !upper_only {
                return Err(Errno::EXDEV);
            }
        }
        if flags.exchange() {
            // Both names stay live, so both sides materialize and swap in
            // the upper; nothing needs a whiteout. A lower-visible directory
            // destination cannot swap for the same reason a lower source
            // cannot move: its children stay behind.
            let Some(dst) = dst else {
                return Err(Errno::ENOENT);
            };
            if dst.file_type == FileType::Directory
                && (inner.lower_visible(to)
                    || !matches!(inner.visibility(to)?, Visibility::Upper { .. }))
            {
                return Err(Errno::EXDEV);
            }
            inner.ensure_upper(from)?;
            inner.ensure_upper(to)?;
            return inner.upper.rename(from, to, flags);
        }
        // The remaining preconditions are Linux's, answered in the merged
        // view — the physical upper may be absent or an empty scaffold where
        // the merged destination is a populated lower directory.
        match &dst {
            Some(d) if src_dir && d.file_type != FileType::Directory => {
                return Err(Errno::ENOTDIR);
            }
            Some(d) if !src_dir && d.file_type == FileType::Directory => {
                return Err(Errno::EISDIR);
            }
            Some(_) if src_dir => {
                // Directory over directory: the destination must be
                // merged-empty (whiteouts hide their lower names).
                let dir = self.open(to, OpenFlags(libc::O_RDONLY | libc::O_DIRECTORY), Mode(0))?;
                if !dir.getdents()?.is_empty() {
                    return Err(Errno::ENOTEMPTY);
                }
            }
            _ => {}
        }
        inner.ensure_upper(from)?;
        inner.delta.materialize_parents(to)?;
        // Prepare the physical destination where the merged and physical
        // views disagree. A whiteout marker is a regular file the kernel
        // would refuse to rename a directory over (and NOREPLACE would trip
        // on); the merged name is free, so the marker goes. A merged-empty
        // upper destination directory may still hold whiteout markers the
        // kernel would count as entries; they vanish with the directory.
        let to_upper = inner.delta.data_path(to);
        let mut removed_marker = false;
        match std::fs::symlink_metadata(&to_upper) {
            Ok(md)
                if may_be_whiteout(&md)
                    && is_whiteout(&to_upper)?
                    && (src_dir || flags.noreplace()) =>
            {
                inner.remove_marker(to)?;
                removed_marker = true;
            }
            Ok(md) if src_dir && md.is_dir() && dst.is_some() => {
                std::fs::remove_dir_all(&to_upper).map_err(|e| Errno::from_io(&e))?;
            }
            _ => {}
        }
        // The upper rename atomically claims `to` — replacing a live upper
        // entry or a whiteout marker alike — and the moved entry then
        // shadows any lower `to`. A failure after the marker went restores
        // it (compare-and-swap), so the deleted name never resurrects.
        if let Err(e) = inner.upper.rename(from, to, flags) {
            if removed_marker {
                let _ = inner.delta.restore_whiteout(to);
            }
            return Err(e);
        }
        // A directory that replaced or shadows a lower name must keep that
        // lower directory's children (present and future) hidden.
        if src_dir && inner.lower_visible(to) {
            inner.delta.set_opaque(to)?;
        }
        if inner.lower_visible(from) {
            inner.delta.whiteout(from)?;
        }
        Ok(())
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
            Visibility::Lower if self.inner.lower_visible(path) => {
                return Err(Errno::EEXIST);
            }
            Visibility::Whiteout => {
                return self.inner.delta.replace_whiteout(path, |staging| {
                    use std::os::unix::ffi::OsStrExt;
                    let cstaging = std::ffi::CString::new(staging.as_os_str().as_bytes())
                        .map_err(|_| Errno::EINVAL)?;
                    if unsafe {
                        libc::mknod(
                            cstaging.as_ptr(),
                            mode.0 as libc::mode_t,
                            dev as libc::dev_t,
                        )
                    } != 0
                    {
                        return Err(Errno::from_io(&std::io::Error::last_os_error()));
                    }
                    Ok(())
                });
            }
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
            Visibility::Upper { lower_masked } => {
                let st = self.inner.upper.stat(path, false)?;
                if self.inner.scaffold_stat(path, &st, lower_masked).is_some() {
                    return self.inner.lower.statfs(path);
                }
                self.inner.upper.statfs(path)
            }
            Visibility::Lower => self.inner.lower.statfs(path),
            Visibility::Whiteout | Visibility::Masked => Err(Errno::ENOENT),
        }
    }

    fn resolve_fast(&self, path: &Path) -> Option<Stat> {
        // Only a pure-lower path can skip the merge. The upper walk stops at
        // the first missing component — one lstat on the common no-shadow
        // prefix — and the lower then proves no symlink participates, which
        // makes the lexical path identical to the merged resolution.
        match self.inner.visibility(path) {
            Ok(Visibility::Lower) => self.inner.lower.resolve_fast(path),
            _ => None,
        }
    }

    fn host_path(&self, path: &Path) -> Option<PathBuf> {
        match self.inner.visibility(path).ok()? {
            // The serving layer answers, so an execve of a modified binary
            // loads the delta's copy, not the host's.
            Visibility::Upper { .. } => Some(self.inner.delta.data_path(path)),
            Visibility::Lower => self.inner.lower.host_path(path),
            Visibility::Whiteout | Visibility::Masked => None,
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

/// A merged directory handle: the upper directory plus the lower directory
/// whose entries fold into `getdents`. While the upper is pure scaffolding
/// the lower owns the identity the guest observes, so `fstat` answers from
/// it; an attribute mutation claims the directory first, after which the
/// upper answers.
struct OverlayDir {
    upper: Box<dyn File>,
    lower: Option<Box<dyn File>>,
    /// The upper directory's host path, where whiteout markers on its entries
    /// live.
    upper_host: PathBuf,
    fs: Arc<OverlayInner>,
    rel: PathBuf,
}

impl OverlayDir {
    /// The lower handle, while it still owns the merged identity (see
    /// [`OverlayInner::scaffold_stat`]; a `lower` here already implies the
    /// merge is live).
    fn scaffold_lower(&self) -> Option<&dyn File> {
        match &self.lower {
            Some(lower) if origin(&self.upper_host).ok()?.is_none() => Some(lower.as_ref()),
            _ => None,
        }
    }
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
        if let Some(lower) = self.scaffold_lower() {
            return lower.fstat();
        }
        self.upper.fstat()
    }

    fn fstatfs(&self) -> Result<StatFs, Errno> {
        if let Some(lower) = self.scaffold_lower() {
            return lower.fstatfs();
        }
        self.upper.fstatfs()
    }

    fn ftruncate(&self, len: u64) -> Result<(), Errno> {
        self.upper.ftruncate(len)
    }

    fn fsync(&self) -> Result<(), Errno> {
        self.upper.fsync()
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
        self.upper.fallocate(mode, offset, len)
    }

    fn getdents(&self) -> Result<Vec<DirEntry>, Errno> {
        let mut out = Vec::new();
        // An upper name shadows the lower's whether it is live (upper wins)
        // or a whiteout (deleted) — and the whiteout marker itself never
        // appears.
        let mut shadowed: HashSet<OsString> = HashSet::new();
        for e in self.upper.getdents()? {
            let whiteout =
                e.file_type == FileType::Regular && is_whiteout(&self.upper_host.join(&e.name))?;
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

    /// The filesystem root's `data/` always exists and scaffolds appear as
    /// upper children land, yet the merged identity — inode, device, mode —
    /// must keep coming from the lower directory until a mutation aimed at
    /// the directory itself claims it.
    #[test]
    fn scaffold_keeps_lower_directory_identity() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let scratch = Scratch::new();
        let Some((fs, _)) = overlay(&scratch) else {
            return;
        };
        std::fs::create_dir(scratch.join("lower/d")).unwrap();
        std::fs::set_permissions(
            scratch.join("lower/d"),
            std::fs::Permissions::from_mode(0o1777),
        )
        .unwrap();
        let lower_root = std::fs::metadata(scratch.join("lower")).unwrap();
        let lower_d = std::fs::metadata(scratch.join("lower/d")).unwrap();

        // "/" is physically upper from the moment the delta exists.
        let st = fs.stat(Path::new("/"), true).unwrap();
        assert_eq!((st.dev, st.ino), (lower_root.dev(), lower_root.ino()));

        // An upper child materializes /d as a scaffold; identity stays put.
        fs.open(
            Path::new("/d/new"),
            OpenFlags(libc::O_CREAT | libc::O_WRONLY),
            Mode(0o644),
        )
        .unwrap();
        let st = fs.stat(Path::new("/d"), true).unwrap();
        assert_eq!(st.mode.0 & 0o7777, 0o1777);
        assert_eq!((st.dev, st.ino), (lower_d.dev(), lower_d.ino()));
        let d = fs
            .open(Path::new("/d"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        let st = d.fstat().unwrap();
        assert_eq!((st.dev, st.ino), (lower_d.dev(), lower_d.ino()));

        // A chmod aimed at /d claims it: the change is visible and the upper
        // owns the identity from here on.
        fs.chmod(Path::new("/d"), true, Mode(0o700)).unwrap();
        let st = fs.stat(Path::new("/d"), true).unwrap();
        assert_eq!(st.mode.0 & 0o7777, 0o700);
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
    fn unlink_lower_writes_whiteout() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::write(scratch.join("lower/f"), b"x").unwrap();

        fs.unlink(Path::new("/f")).unwrap();
        assert_eq!(fs.stat(Path::new("/f"), true).unwrap_err(), Errno::ENOENT);
        let root = fs
            .open(Path::new("/"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert!(names(root.as_ref()).is_empty());
        assert!(super::super::delta::is_whiteout(&delta.data_path(Path::new("/f"))).unwrap());
        assert!(scratch.join("lower/f").is_file());

        // Deleting it again is ENOENT, and unlinking a directory is EISDIR.
        assert_eq!(fs.unlink(Path::new("/f")), Err(Errno::ENOENT));
        std::fs::create_dir(scratch.join("lower/d")).unwrap();
        assert_eq!(fs.unlink(Path::new("/d")), Err(Errno::EISDIR));
    }

    #[test]
    fn unlink_upper_only_leaves_no_marker() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        fs.open(
            Path::new("/fresh"),
            OpenFlags(libc::O_CREAT | libc::O_WRONLY),
            Mode(0o644),
        )
        .unwrap();
        fs.unlink(Path::new("/fresh")).unwrap();
        assert_eq!(
            fs.stat(Path::new("/fresh"), true).unwrap_err(),
            Errno::ENOENT
        );
        assert!(!delta.data_path(Path::new("/fresh")).exists());
    }

    #[test]
    fn delete_then_recreate_does_not_bleed_lower() {
        let scratch = Scratch::new();
        let Some((fs, _)) = overlay(&scratch) else {
            return;
        };
        std::fs::write(scratch.join("lower/f"), b"old lower bytes").unwrap();

        fs.unlink(Path::new("/f")).unwrap();
        // O_EXCL succeeds: the merged view has no such name anymore.
        let f = fs
            .open(
                Path::new("/f"),
                OpenFlags(libc::O_CREAT | libc::O_EXCL | libc::O_RDWR),
                Mode(0o644),
            )
            .unwrap();
        assert_eq!(f.fstat().unwrap().size, 0);
        assert_eq!(f.pwrite(b"new", 0).unwrap(), 3);
        let g = fs
            .open(Path::new("/f"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(read_all(g.as_ref()), b"new");
        assert_eq!(
            std::fs::read(scratch.join("lower/f")).unwrap(),
            b"old lower bytes"
        );
    }

    #[test]
    fn rmdir_whiteouts_lower_directory() {
        let scratch = Scratch::new();
        let Some((fs, _)) = overlay(&scratch) else {
            return;
        };
        std::fs::create_dir(scratch.join("lower/d")).unwrap();
        std::fs::write(scratch.join("lower/d/f"), b"x").unwrap();

        // Not empty in the merged view yet.
        assert_eq!(fs.rmdir(Path::new("/d")), Err(Errno::ENOTEMPTY));
        fs.unlink(Path::new("/d/f")).unwrap();
        // Now merged-empty: the upper holds only the whiteout marker.
        fs.rmdir(Path::new("/d")).unwrap();
        assert_eq!(fs.stat(Path::new("/d"), true).unwrap_err(), Errno::ENOENT);
        assert!(scratch.join("lower/d/f").is_file());

        // mkdir over the deleted name: opaque, so the lower file must not
        // bleed back through.
        fs.mkdir(Path::new("/d"), Mode(0o755)).unwrap();
        assert_eq!(fs.stat(Path::new("/d/f"), true).unwrap_err(), Errno::ENOENT);
        let d = fs
            .open(Path::new("/d"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert!(names(d.as_ref()).is_empty());
    }

    #[test]
    fn symlink_and_link_materialize_in_upper() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::write(scratch.join("lower/orig"), b"data").unwrap();

        fs.symlink(Path::new("orig"), Path::new("/sl")).unwrap();
        assert_eq!(
            fs.readlink(Path::new("/sl")).unwrap(),
            PathBuf::from("orig")
        );
        assert!(!scratch.join("lower/sl").exists());

        // A hard link to a lower file copies the source up first; the two
        // upper names share an inode, and the host never grows a link.
        fs.link(Path::new("/orig"), Path::new("/alias")).unwrap();
        let a = fs.stat(Path::new("/orig"), false).unwrap();
        let b = fs.stat(Path::new("/alias"), false).unwrap();
        assert_eq!(a.ino, b.ino);
        assert_eq!(b.nlink, 2);
        assert!(delta.data_path(Path::new("/alias")).is_file());
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            std::fs::metadata(scratch.join("lower/orig"))
                .unwrap()
                .nlink(),
            1
        );

        assert_eq!(
            fs.symlink(Path::new("x"), Path::new("/orig")),
            Err(Errno::EEXIST)
        );
        assert_eq!(
            fs.link(Path::new("/orig"), Path::new("/alias")),
            Err(Errno::EEXIST)
        );
    }

    #[test]
    fn rename_copies_up_and_whiteouts_source() {
        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::write(scratch.join("lower/from"), b"payload").unwrap();
        std::fs::write(scratch.join("lower/victim"), b"gone").unwrap();

        fs.rename(Path::new("/from"), Path::new("/to"), RenameFlags::EMPTY)
            .unwrap();
        assert_eq!(
            fs.stat(Path::new("/from"), true).unwrap_err(),
            Errno::ENOENT
        );
        let to = fs
            .open(Path::new("/to"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(read_all(to.as_ref()), b"payload");
        assert!(scratch.join("lower/from").is_file());
        assert!(super::super::delta::is_whiteout(&delta.data_path(Path::new("/from"))).unwrap());

        // Renaming over a lower-visible name shadows it; no whiteout needed,
        // and the host victim survives.
        fs.rename(Path::new("/to"), Path::new("/victim"), RenameFlags::EMPTY)
            .unwrap();
        let v = fs
            .open(Path::new("/victim"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(read_all(v.as_ref()), b"payload");
        assert_eq!(
            std::fs::read(scratch.join("lower/victim")).unwrap(),
            b"gone"
        );

        // RENAME_NOREPLACE sees the merged target.
        std::fs::write(scratch.join("lower/other"), b"z").unwrap();
        assert_eq!(
            fs.rename(
                Path::new("/victim"),
                Path::new("/other"),
                RenameFlags(libc::RENAME_NOREPLACE),
            ),
            Err(Errno::EEXIST)
        );
    }

    /// A directory renamed over a lower directory is committed opaque:
    /// children the lower gains afterward must never merge into the moved
    /// directory the guest sees at that name.
    #[test]
    fn rename_over_lower_directory_commits_opacity() {
        let scratch = Scratch::new();
        let Some((fs, _)) = overlay(&scratch) else {
            return;
        };
        std::fs::create_dir(scratch.join("lower/dest")).unwrap();
        fs.mkdir(Path::new("/moved"), Mode(0o755)).unwrap();
        fs.rename(Path::new("/moved"), Path::new("/dest"), RenameFlags::EMPTY)
            .unwrap();

        // The host writes into the replaced lower directory afterward.
        std::fs::write(scratch.join("lower/dest/late"), b"x").unwrap();
        let dir = fs
            .open(Path::new("/dest"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert!(names(dir.as_ref()).is_empty());
    }

    #[test]
    fn rename_of_lower_directory_is_exdev() {
        let scratch = Scratch::new();
        let Some((fs, _)) = overlay(&scratch) else {
            return;
        };
        std::fs::create_dir(scratch.join("lower/d")).unwrap();
        std::fs::write(scratch.join("lower/d/f"), b"x").unwrap();

        assert_eq!(
            fs.rename(Path::new("/d"), Path::new("/e"), RenameFlags::EMPTY),
            Err(Errno::EXDEV)
        );

        // A directory born in the upper renames freely.
        fs.mkdir(Path::new("/u"), Mode(0o755)).unwrap();
        fs.rename(Path::new("/u"), Path::new("/v"), RenameFlags::EMPTY)
            .unwrap();
        assert_eq!(
            fs.stat(Path::new("/v"), true).unwrap().file_type,
            FileType::Directory
        );
    }

    #[test]
    fn resolve_fast_proves_or_declines() {
        use super::super::namespace::{MountFlags, Namespace};

        let scratch = Scratch::new();
        let Some((fs, delta)) = overlay(&scratch) else {
            return;
        };
        std::fs::create_dir_all(scratch.join("lower/d")).unwrap();
        std::fs::write(scratch.join("lower/d/f"), b"x").unwrap();
        std::fs::create_dir_all(scratch.join("lower/real")).unwrap();
        std::fs::write(scratch.join("lower/real/g"), b"lower-g").unwrap();
        std::os::unix::fs::symlink("real", scratch.join("lower/link")).unwrap();

        // A symlink-free, unshadowed path resolves in one step.
        let st = fs.resolve_fast(Path::new("/d/f")).expect("pure lower");
        assert_eq!(st.file_type, FileType::Regular);

        // A symlink in the chain, a shadowed path, and a missing file all
        // decline to the walk.
        assert!(fs.resolve_fast(Path::new("/link/g")).is_none());
        delta
            .copy_up(&scratch.join("lower/d/f"), Path::new("/d/f"))
            .unwrap();
        assert!(fs.resolve_fast(Path::new("/d/f")).is_none());
        assert!(fs.resolve_fast(Path::new("/absent")).is_none());

        // End to end: the walk the fast path declines to must see an upper
        // override through a lower symlink — the case the no-symlink proof
        // exists for.
        std::fs::create_dir_all(delta.data_path(Path::new("/real"))).unwrap();
        std::fs::write(delta.data_path(Path::new("/real/g")), b"upper-g").unwrap();
        let ns = Namespace::with_root(
            Arc::new(OverlayFs {
                inner: Arc::clone(&fs.inner),
            }),
            MountFlags::NONE,
        );
        let via_link = ns.stat_path(Path::new("/link/g"), true).unwrap();
        let direct = ns.stat_path(Path::new("/real/g"), true).unwrap();
        assert_eq!(via_link.ino, direct.ino);
        assert_eq!(via_link.size, 7); // "upper-g", not "lower-g"'s identity
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
