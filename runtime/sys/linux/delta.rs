//! The delta directory: the on-disk format of a filesystem's writable upper
//! layer, and the copy-up engine that populates it.
//!
//! `<delta>/data/` is the upper tree of real host files; `<delta>/tmp/` stages
//! copy-ups so the final `rename` into `data/` is atomic on the same
//! filesystem — a crash can never leave a truncated upper file shadowing a
//! good lower one. All bookkeeping is encoded *in* the tree, never in process
//! memory: Chimera emulates guest `fork` with a host fork, so one sandbox is
//! many host processes, and an in-memory index would tear at the first fork.
//! With the metadata in the filesystem, the kernel is the shared, crash-safe,
//! offline-readable database.
//!
//! The markers: a whiteout is an empty file carrying the
//! `user.chimera.whiteout` xattr; a directory that must not merge lower
//! entries carries `user.chimera.opaque`; every copied-up file records the
//! identity of the lower file it shadows (`dev:ino:size:mtime`) in
//! `user.chimera.origin`. The delta must therefore sit on a filesystem with
//! user xattrs (ext4, XFS, btrfs, tmpfs); [`Delta::open`] probes for them and
//! refuses at startup rather than failing at the first unlink.

use std::{
    ffi::{CStr, CString},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::ffi::OsStrExt,
    },
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use super::vfs::Errno;

/// The xattr namespace holding overlay bookkeeping. Task 6 hides it from the
/// guest: names under this prefix are invisible to `getxattr`/`listxattr` and
/// rejected by `setxattr`, so a guest can neither read nor forge markers.
pub const XATTR_NAMESPACE: &[u8] = b"user.chimera.";

const WHITEOUT_XATTR: &CStr = c"user.chimera.whiteout";
const OPAQUE_XATTR: &CStr = c"user.chimera.opaque";
const ORIGIN_XATTR: &CStr = c"user.chimera.origin";
const APPLIED_XATTR: &CStr = c"user.chimera.applied";
const PROBE_XATTR: &CStr = c"user.chimera.probe";

/// A filesystem's delta directory: the writable upper tree under `data/` and
/// the `tmp/` staging area copy-ups rename in from.
pub struct Delta {
    data: PathBuf,
    tmp: PathBuf,
    /// Distinguishes concurrent stagings within one process; the pid in the
    /// staging name keeps forked processes apart.
    staging_seq: AtomicU64,
}

impl Delta {
    /// Create or open the delta directory at `root`. Idempotent: an existing
    /// delta is reopened as-is, which is how a filesystem survives both `fork`
    /// (each process holds its own `Delta` over the same tree) and a later
    /// attach. Fails with `ENOTSUP` when the backing filesystem lacks user
    /// xattrs.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, Errno> {
        let root = root.into();
        let delta = Self {
            data: root.join("data"),
            tmp: root.join("tmp"),
            staging_seq: AtomicU64::new(0),
        };
        std::fs::create_dir_all(&delta.data).map_err(|e| Errno::from_io(&e))?;
        std::fs::create_dir_all(&delta.tmp).map_err(|e| Errno::from_io(&e))?;
        delta.probe_xattrs()?;
        Ok(delta)
    }

    /// The upper file backing a mount-relative guest path.
    pub fn data_path(&self, rel: &Path) -> PathBuf {
        let rel = rel.strip_prefix("/").unwrap_or(rel);
        self.data.join(rel)
    }

    /// Whether the format's xattrs work here, checked once at open so a delta
    /// on the wrong filesystem is refused up front, not discovered at the
    /// first unlink.
    fn probe_xattrs(&self) -> Result<(), Errno> {
        let probe = self.tmp.join(format!("probe.{}", std::process::id()));
        let cprobe = cpath(&probe)?;
        let fd = unsafe {
            libc::open(
                cprobe.as_ptr(),
                libc::O_CREAT | libc::O_WRONLY | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err(last_errno());
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let r = unsafe {
            libc::fsetxattr(
                fd.as_raw_fd(),
                PROBE_XATTR.as_ptr(),
                c"1".as_ptr() as *const libc::c_void,
                1,
                0,
            )
        };
        let result = if r < 0 { Err(last_errno()) } else { Ok(()) };
        unsafe { libc::unlink(cprobe.as_ptr()) };
        result
    }

    /// A fresh, collision-free staging path in `tmp/`.
    fn staging_path(&self) -> PathBuf {
        let seq = self.staging_seq.fetch_add(1, Ordering::Relaxed);
        self.tmp
            .join(format!("staging.{}.{}", std::process::id(), seq))
    }

    /// Create the upper directories a copy-up or marker at `rel` needs.
    /// Parents materialize lazily — only when something lands beneath them —
    /// so an untouched subtree leaves no trace in the delta.
    pub fn materialize_parents(&self, rel: &Path) -> Result<(), Errno> {
        if let Some(parent) = self.data_path(rel).parent() {
            std::fs::create_dir_all(parent).map_err(|e| Errno::from_io(&e))?;
        }
        Ok(())
    }

    /// Create the upper directory at `rel` itself as scaffolding: structure
    /// an upper entry needs beneath it, not a claim on the directory's
    /// identity. A scaffold carries no origin, which is what keeps stat
    /// serving the lower directory it merges with.
    pub fn scaffold(&self, rel: &Path) -> Result<(), Errno> {
        std::fs::create_dir_all(self.data_path(rel)).map_err(|e| Errno::from_io(&e))
    }

    /// Deliberately copy the lower directory at the host path `src` up to
    /// `rel`, for a mutation aimed at the directory itself: mirror its
    /// permission bits and timestamps, then record the origin. The origin
    /// lands last, so an interruption leaves ordinary scaffolding — which
    /// keeps serving the lower — never a half-claimed directory.
    pub fn copy_up_dir(&self, src: &Path, rel: &Path) -> Result<(), Errno> {
        let csrc = cpath(src)?;
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        check(unsafe { libc::stat(csrc.as_ptr(), &mut st) })?;

        std::fs::create_dir_all(self.data_path(rel)).map_err(|e| Errno::from_io(&e))?;
        let cdst = cpath(&self.data_path(rel))?;
        check(unsafe { libc::chmod(cdst.as_ptr(), (st.st_mode & 0o7777) as libc::mode_t) })?;
        let times = [
            libc::timespec {
                tv_sec: st.st_atime,
                tv_nsec: st.st_atime_nsec,
            },
            libc::timespec {
                tv_sec: st.st_mtime,
                tv_nsec: st.st_mtime_nsec,
            },
        ];
        check(unsafe { libc::utimensat(libc::AT_FDCWD, cdst.as_ptr(), times.as_ptr(), 0) })?;
        let origin = Origin::of(&st).encode();
        check(unsafe {
            libc::lsetxattr(
                cdst.as_ptr(),
                ORIGIN_XATTR.as_ptr(),
                origin.as_ptr() as *const libc::c_void,
                origin.len(),
                0,
            )
        })
    }

    /// Build a fresh marker file under a staging name.
    fn stage_whiteout(&self) -> Result<PathBuf, Errno> {
        let staging = self.staging_path();
        let cstaging = cpath(&staging)?;
        // Owner-readable, or the marker xattr itself becomes unreadable: the
        // kernel gates a user.* xattr read on read permission to the file.
        let fd = unsafe {
            libc::open(
                cstaging.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_WRONLY | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err(last_errno());
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        match check(unsafe {
            libc::fsetxattr(
                fd.as_raw_fd(),
                WHITEOUT_XATTR.as_ptr(),
                c"1".as_ptr() as *const libc::c_void,
                1,
                0,
            )
        }) {
            Ok(()) => Ok(staging),
            Err(e) => {
                let _ = std::fs::remove_file(&staging);
                Err(e)
            }
        }
    }

    /// Write a whiteout for `rel`: the upper entry that says "this name is
    /// deleted", hiding any lower file. Staged and renamed like a copy-up —
    /// a marker file observed without its xattr would read as an empty upper
    /// file shadowing live lower content.
    pub fn whiteout(&self, rel: &Path) -> Result<(), Errno> {
        let staging = self.stage_whiteout()?;
        let publish = (|| {
            self.materialize_parents(rel)?;
            std::fs::rename(&staging, self.data_path(rel)).map_err(|e| Errno::from_io(&e))
        })();
        if publish.is_err() {
            let _ = std::fs::remove_file(&staging);
        }
        publish
    }

    /// Put a whiteout back at `rel` after a failed creation removed the
    /// marker — compare-and-swap, so a concurrent creator that took the name
    /// in the meantime keeps it.
    pub fn restore_whiteout(&self, rel: &Path) -> Result<(), Errno> {
        let staging = self.stage_whiteout()?;
        match rename_noreplace(&staging, &self.data_path(rel)) {
            Ok(()) => Ok(()),
            Err(Errno(libc::EEXIST)) => {
                let _ = std::fs::remove_file(&staging);
                Ok(())
            }
            Err(e) => {
                let _ = std::fs::remove_file(&staging);
                Err(e)
            }
        }
    }

    /// Replace the whiteout at `rel` with an object `build` constructs under
    /// a staging name: the marker leaves the tree only in the same exchange
    /// that publishes its fully-made replacement (plain rename cannot put a
    /// directory over the marker file), so a failed construction leaves the
    /// recorded deletion exactly in place and the merged view never flickers
    /// back to the lower entry. A marker already replaced by a concurrent
    /// creator answers `EEXIST`; if the exchange nevertheless swaps out a
    /// live object rather than a marker, the object is left in `tmp/` as
    /// garbage, never destroyed.
    pub fn replace_whiteout(
        &self,
        rel: &Path,
        build: impl FnOnce(&Path) -> Result<(), Errno>,
    ) -> Result<(), Errno> {
        let staging = self.staging_path();
        build(&staging)?;
        let discard = |staging: &Path| {
            let _ = match std::fs::symlink_metadata(staging) {
                Ok(md) if md.is_dir() => std::fs::remove_dir_all(staging),
                _ => std::fs::remove_file(staging),
            };
        };
        match rename_exchange(&staging, &self.data_path(rel)) {
            Ok(()) => {
                if is_whiteout(&staging).unwrap_or(false) {
                    let _ = std::fs::remove_file(&staging);
                }
                Ok(())
            }
            Err(Errno(libc::ENOENT)) => {
                discard(&staging);
                Err(Errno::EEXIST)
            }
            Err(e) => {
                discard(&staging);
                Err(e)
            }
        }
    }

    /// Mark an existing upper directory opaque: lower entries beneath it no
    /// longer merge into the guest's view. A single xattr write is already
    /// atomic, so no staging is needed.
    pub fn set_opaque(&self, rel: &Path) -> Result<(), Errno> {
        mark_opaque(&self.data_path(rel))
    }

    /// Copy the lower regular file at the host path `src` up to `rel`:
    /// contents (clone → `copy_file_range` → read/write loop), ownership,
    /// permission bits, timestamps, and the carried xattrs, with the lower's
    /// identity recorded in the origin xattr — all staged in `tmp/` and
    /// renamed into `data/` in one atomic step. Publication is first-wins:
    /// the staging file claims the name only
    /// if no upper entry exists (`RENAME_NOREPLACE`). Two staged copies stop
    /// being equivalent the moment the first publisher opens its result, so a
    /// loser must never detach the winning inode — it discards its staging
    /// copy and returns success, and the caller continues through the
    /// published entry, which a concurrent operation may also have made a
    /// whiteout or another object entirely. The caller owns validating that.
    pub fn copy_up(&self, src: &Path, rel: &Path) -> Result<(), Errno> {
        let csrc = cpath(src)?;
        let src_fd = unsafe { libc::open(csrc.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        if src_fd < 0 {
            return Err(last_errno());
        }
        let src_fd = unsafe { OwnedFd::from_raw_fd(src_fd) };
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(src_fd.as_raw_fd(), &mut st) } < 0 {
            return Err(last_errno());
        }

        let staging = self.staging_path();
        let cstaging = cpath(&staging)?;
        let dst_fd = unsafe {
            libc::open(
                cstaging.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_WRONLY | libc::O_CLOEXEC,
                0o600,
            )
        };
        if dst_fd < 0 {
            return Err(last_errno());
        }
        let dst_fd = unsafe { OwnedFd::from_raw_fd(dst_fd) };

        let publish = (|| {
            transfer(src_fd.as_raw_fd(), dst_fd.as_raw_fd())?;
            // Everything lands on the staging file before the rename
            // publishes it: an upper file is never observable half-made.
            // Ownership and ACLs come before the mode bits — chown can
            // clear set-id bits and an ACL write can rewrite the group
            // class — and timestamps go last (below), after every step that
            // could disturb them.
            copy_fd_xattrs(src_fd.as_raw_fd(), dst_fd.as_raw_fd())?;
            check(unsafe { libc::fchown(dst_fd.as_raw_fd(), st.st_uid, st.st_gid) })?;
            // The open's 0o600 was filtered by the umask; set the real bits.
            check(unsafe {
                libc::fchmod(dst_fd.as_raw_fd(), (st.st_mode & 0o7777) as libc::mode_t)
            })?;
            let origin = Origin::of(&st).encode();
            check(unsafe {
                libc::fsetxattr(
                    dst_fd.as_raw_fd(),
                    ORIGIN_XATTR.as_ptr(),
                    origin.as_ptr() as *const libc::c_void,
                    origin.len(),
                    0,
                )
            })?;
            let times = [
                libc::timespec {
                    tv_sec: st.st_atime,
                    tv_nsec: st.st_atime_nsec,
                },
                libc::timespec {
                    tv_sec: st.st_mtime,
                    tv_nsec: st.st_mtime_nsec,
                },
            ];
            check(unsafe { libc::futimens(dst_fd.as_raw_fd(), times.as_ptr()) })?;
            self.materialize_parents(rel)?;
            match rename_noreplace(&staging, &self.data_path(rel)) {
                Err(Errno(libc::EEXIST)) => {
                    let _ = std::fs::remove_file(&staging);
                    Ok(())
                }
                r => r,
            }
        })();
        if publish.is_err() {
            let _ = std::fs::remove_file(&staging);
        }
        publish
    }
}

/// `rename(2)` that refuses to replace an existing destination — the
/// compare-and-swap a first-wins publication needs.
fn rename_noreplace(from: &Path, to: &Path) -> Result<(), Errno> {
    let cfrom = cpath(from)?;
    let cto = cpath(to)?;
    check(unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            cfrom.as_ptr(),
            libc::AT_FDCWD,
            cto.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    })
}

/// `rename(2)` that atomically swaps two existing entries — how a staged
/// object of any type takes a whiteout marker's place in one step.
fn rename_exchange(a: &Path, b: &Path) -> Result<(), Errno> {
    let ca = cpath(a)?;
    let cb = cpath(b)?;
    check(unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            ca.as_ptr(),
            libc::AT_FDCWD,
            cb.as_ptr(),
            libc::RENAME_EXCHANGE,
        )
    })
}

/// Whether an entry with this metadata could be a whiteout at all: a
/// whiteout is always an *empty* regular file — built by
/// [`Delta::stage_whiteout`] and never written — so a file with bytes in it,
/// and every other type, provably carries no marker and needs no xattr read.
pub fn may_be_whiteout(md: &std::fs::Metadata) -> bool {
    md.is_file() && md.len() == 0
}

/// `true` when the upper entry at `path` is a whiteout. A missing entry is
/// not one, and a filesystem answer of "no such attribute" (or no xattrs at
/// all) is an ordinary upper file.
pub fn is_whiteout(path: &Path) -> Result<bool, Errno> {
    has_marker(path, WHITEOUT_XATTR)
}

/// `true` when the upper directory at `path` is opaque.
pub fn is_opaque(path: &Path) -> Result<bool, Errno> {
    has_marker(path, OPAQUE_XATTR)
}

/// Mark the directory at `path` (a host path — an upper entry or a staging
/// directory about to be published as one) opaque.
pub fn mark_opaque(path: &Path) -> Result<(), Errno> {
    let cdir = cpath(path)?;
    check(unsafe {
        libc::lsetxattr(
            cdir.as_ptr(),
            OPAQUE_XATTR.as_ptr(),
            c"1".as_ptr() as *const libc::c_void,
            1,
            0,
        )
    })
}

/// Record `origin` on the upper entry at `path` — how apply advances a
/// change's origin to the exact host identity it just produced, so a rerun
/// recognizes its own work instead of conflicting with it.
pub fn record_origin(path: &Path, origin: &Origin) -> Result<(), Errno> {
    let cpath = cpath(path)?;
    let encoded = origin.encode();
    check(unsafe {
        libc::lsetxattr(
            cpath.as_ptr(),
            ORIGIN_XATTR.as_ptr(),
            encoded.as_ptr() as *const libc::c_void,
            encoded.len(),
            0,
        )
    })
}

/// Mark the whiteout at `path` as applied: its host removal has happened (or
/// was found already done), and any host entry at the name afterward is the
/// host's own, never to be deleted again by a rerun.
pub fn mark_applied(path: &Path) -> Result<(), Errno> {
    let cpath = cpath(path)?;
    check(unsafe {
        libc::lsetxattr(
            cpath.as_ptr(),
            APPLIED_XATTR.as_ptr(),
            c"1".as_ptr() as *const libc::c_void,
            1,
            0,
        )
    })
}

/// `true` when the whiteout at `path` has already been applied to the host.
pub fn is_applied(path: &Path) -> Result<bool, Errno> {
    has_marker(path, APPLIED_XATTR)
}

/// `true` when the open file is a whiteout marker. Race-free where the
/// path-based probe is not: a whiteout only ever arrives whole by rename,
/// never by marking a live file in place, so an fd carrying the xattr proves
/// the name was already deleted when the open resolved it.
pub fn is_whiteout_fd(fd: std::os::fd::RawFd) -> Result<bool, Errno> {
    let n = unsafe { libc::fgetxattr(fd, WHITEOUT_XATTR.as_ptr(), std::ptr::null_mut(), 0) };
    if n >= 0 {
        return Ok(true);
    }
    match last_errno() {
        Errno(libc::ENODATA) | Errno(libc::ENOTSUP) => Ok(false),
        e => Err(e),
    }
}

fn has_marker(path: &Path, name: &CStr) -> Result<bool, Errno> {
    let cpath = cpath(path)?;
    let n = unsafe { libc::lgetxattr(cpath.as_ptr(), name.as_ptr(), std::ptr::null_mut(), 0) };
    if n >= 0 {
        return Ok(true);
    }
    match last_errno() {
        Errno(libc::ENODATA) | Errno(libc::ENOTSUP) | Errno(libc::ENOENT) => Ok(false),
        e => Err(e),
    }
}

/// The identity of the lower file a copy-up shadows, as recorded in the
/// origin xattr. `dev:ino` name the file; `size` and `mtime` date the copy,
/// so a later run can tell that the lower changed underneath the upper.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Origin {
    pub dev: u64,
    pub ino: u64,
    pub size: u64,
    pub mtime_sec: i64,
    pub mtime_nsec: i64,
}

impl Origin {
    pub fn of(st: &libc::stat) -> Origin {
        Origin {
            dev: st.st_dev,
            ino: st.st_ino,
            size: st.st_size as u64,
            mtime_sec: st.st_mtime,
            mtime_nsec: st.st_mtime_nsec,
        }
    }

    fn encode(&self) -> String {
        format!(
            "{}:{}:{}:{}.{}",
            self.dev, self.ino, self.size, self.mtime_sec, self.mtime_nsec
        )
    }

    fn decode(bytes: &[u8]) -> Option<Origin> {
        let s = std::str::from_utf8(bytes).ok()?;
        let mut fields = s.split(':');
        let dev = fields.next()?.parse().ok()?;
        let ino = fields.next()?.parse().ok()?;
        let size = fields.next()?.parse().ok()?;
        let mut mtime = fields.next()?.split('.');
        let mtime_sec = mtime.next()?.parse().ok()?;
        let mtime_nsec = mtime.next()?.parse().ok()?;
        if fields.next().is_some() || mtime.next().is_some() {
            return None;
        }
        Some(Origin {
            dev,
            ino,
            size,
            mtime_sec,
            mtime_nsec,
        })
    }
}

/// The origin recorded on the upper file at `path`, or `None` for an entry
/// the guest created in the upper (nothing lower to compare against) — a
/// malformed record reads as `None` too, failing toward "guest-created".
pub fn origin(path: &Path) -> Result<Option<Origin>, Errno> {
    let cpath = cpath(path)?;
    let mut buf = [0u8; 128];
    let n = unsafe {
        libc::lgetxattr(
            cpath.as_ptr(),
            ORIGIN_XATTR.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            buf.len(),
        )
    };
    if n < 0 {
        return match last_errno() {
            Errno(libc::ENODATA) | Errno(libc::ENOTSUP) => Ok(None),
            e => Err(e),
        };
    }
    Ok(Origin::decode(&buf[..n as usize]))
}

/// The lower attributes a copy-up carries: the guest-visible user namespace
/// and the POSIX ACLs. The reserved `user.chimera.*` names never come along
/// — a lower file carrying one, forged or stray, must not become delta
/// bookkeeping — and kernel-managed namespaces (`security.*`) belong to the
/// upper file's own filesystem.
fn carried_xattr(name: &[u8]) -> bool {
    (name.starts_with(b"user.") && !name.starts_with(XATTR_NAMESPACE))
        || name == b"system.posix_acl_access"
        || name == b"system.posix_acl_default"
}

/// Copy the carried xattrs from one open file to another.
fn copy_fd_xattrs(src: libc::c_int, dst: libc::c_int) -> Result<(), Errno> {
    let mut names = vec![0u8; 1024];
    let len = loop {
        let n =
            unsafe { libc::flistxattr(src, names.as_mut_ptr() as *mut libc::c_char, names.len()) };
        if n >= 0 {
            break n as usize;
        }
        match last_errno() {
            Errno(libc::ERANGE) => names.resize(names.len() * 2, 0),
            // A lower filesystem without xattrs holds none to carry.
            Errno(libc::ENOTSUP) => return Ok(()),
            e => return Err(e),
        }
    };
    for name in names[..len].split(|&b| b == 0).filter(|n| !n.is_empty()) {
        if !carried_xattr(name) {
            continue;
        }
        let cname = CString::new(name).map_err(|_| Errno::EINVAL)?;
        let mut value = vec![0u8; 256];
        let vlen = loop {
            let n = unsafe {
                libc::fgetxattr(
                    src,
                    cname.as_ptr(),
                    value.as_mut_ptr() as *mut libc::c_void,
                    value.len(),
                )
            };
            if n >= 0 {
                break n as usize;
            }
            match last_errno() {
                Errno(libc::ERANGE) => value.resize(value.len() * 2, 0),
                e => return Err(e),
            }
        };
        check(unsafe {
            libc::fsetxattr(
                dst,
                cname.as_ptr(),
                value.as_ptr() as *const libc::c_void,
                vlen,
                0,
            )
        })?;
    }
    Ok(())
}

/// Copy `src`'s contents to `dst` (both at offset 0, `dst` freshly created):
/// `FICLONE` shares extents in one ioctl where the filesystem supports
/// reflinks; `copy_file_range` copies in-kernel otherwise; a read/write loop
/// is the portable floor. The fallbacks trigger only when no bytes have
/// moved, so each strategy starts from a clean destination.
fn transfer(src: libc::c_int, dst: libc::c_int) -> Result<(), Errno> {
    if unsafe { libc::ioctl(dst, libc::FICLONE, src) } == 0 {
        return Ok(());
    }

    let mut copied = false;
    loop {
        let n = unsafe {
            libc::copy_file_range(
                src,
                std::ptr::null_mut(),
                dst,
                std::ptr::null_mut(),
                1 << 30,
                0,
            )
        };
        if n == 0 {
            return Ok(());
        }
        if n > 0 {
            copied = true;
            continue;
        }
        let e = last_errno();
        if copied {
            return Err(e);
        }
        match e {
            // The "not supported here" class: fall through to read/write.
            Errno(libc::EXDEV)
            | Errno(libc::EINVAL)
            | Errno(libc::ENOSYS)
            | Errno(libc::EOPNOTSUPP)
            | Errno(libc::EBADF) => break,
            e => return Err(e),
        }
    }

    let mut buf = vec![0u8; 128 * 1024];
    loop {
        let n = unsafe { libc::read(src, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        if n == 0 {
            return Ok(());
        }
        if n < 0 {
            return Err(last_errno());
        }
        let mut off = 0usize;
        while off < n as usize {
            let w = unsafe {
                libc::write(
                    dst,
                    buf[off..].as_ptr() as *const libc::c_void,
                    n as usize - off,
                )
            };
            if w <= 0 {
                return Err(last_errno());
            }
            off += w as usize;
        }
    }
}

/// `CString` from a path's bytes; an embedded NUL is `EINVAL`.
fn cpath(path: &Path) -> Result<CString, Errno> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| Errno::EINVAL)
}

/// Map a libc return of `-1` to the current errno, anything else to `Ok`.
fn check(ret: libc::c_int) -> Result<(), Errno> {
    if ret < 0 { Err(last_errno()) } else { Ok(()) }
}

/// The current errno, wrapped.
fn last_errno() -> Errno {
    Errno::from_io(&std::io::Error::last_os_error())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU32;

    use super::*;

    /// A unique scratch directory, removed on drop.
    struct Scratch {
        path: PathBuf,
    }

    impl Scratch {
        fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("chimera-delta-{}-{}", std::process::id(), n));
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

    /// The delta under test, or `None` when the scratch filesystem has no
    /// user xattrs and the format cannot be exercised at all.
    fn delta(scratch: &Scratch) -> Option<Delta> {
        match Delta::open(scratch.join("delta")) {
            Ok(d) => Some(d),
            Err(Errno(libc::ENOTSUP)) => None,
            Err(e) => panic!("delta open failed: {e:?}"),
        }
    }

    #[test]
    fn open_creates_layout_and_reopens() {
        let scratch = Scratch::new();
        let Some(_) = delta(&scratch) else { return };
        assert!(scratch.join("delta/data").is_dir());
        assert!(scratch.join("delta/tmp").is_dir());
        // Reopening an existing delta is how attach and fork see it.
        assert!(Delta::open(scratch.join("delta")).is_ok());
    }

    #[test]
    fn whiteout_marker_roundtrip() {
        let scratch = Scratch::new();
        let Some(d) = delta(&scratch) else { return };

        d.whiteout(Path::new("/a/b/gone")).unwrap();
        let upper = d.data_path(Path::new("/a/b/gone"));
        assert!(is_whiteout(&upper).unwrap());
        assert!(!is_opaque(&upper).unwrap());
        // The marker is an empty regular file.
        assert_eq!(std::fs::metadata(&upper).unwrap().len(), 0);

        // An ordinary upper file is not a whiteout, and neither is a missing
        // entry.
        std::fs::write(d.data_path(Path::new("/plain")), b"x").unwrap();
        assert!(!is_whiteout(&d.data_path(Path::new("/plain"))).unwrap());
        assert!(!is_whiteout(&d.data_path(Path::new("/absent"))).unwrap());
    }

    #[test]
    fn opaque_marks_directory() {
        let scratch = Scratch::new();
        let Some(d) = delta(&scratch) else { return };

        std::fs::create_dir_all(d.data_path(Path::new("/dir"))).unwrap();
        d.set_opaque(Path::new("/dir")).unwrap();
        assert!(is_opaque(&d.data_path(Path::new("/dir"))).unwrap());
        assert!(!is_whiteout(&d.data_path(Path::new("/dir"))).unwrap());
    }

    #[test]
    fn origin_encoding_roundtrips() {
        let o = Origin {
            dev: 1,
            ino: 42,
            size: 4096,
            mtime_sec: 1700000000,
            mtime_nsec: 123456789,
        };
        assert_eq!(Origin::decode(o.encode().as_bytes()), Some(o));
        assert_eq!(Origin::decode(b"garbage"), None);
        assert_eq!(Origin::decode(b"1:2:3:4.5:6"), None);
    }

    #[test]
    fn copy_up_preserves_content_attributes_and_origin() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let scratch = Scratch::new();
        let Some(d) = delta(&scratch) else { return };

        let src = scratch.join("lower");
        std::fs::write(&src, b"lower bytes").unwrap();
        std::fs::set_permissions(&src, std::fs::Permissions::from_mode(0o640)).unwrap();
        let times = [
            libc::timespec {
                tv_sec: 111,
                tv_nsec: 0,
            },
            libc::timespec {
                tv_sec: 222,
                tv_nsec: 0,
            },
        ];
        let csrc = CString::new(src.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { libc::utimensat(libc::AT_FDCWD, csrc.as_ptr(), times.as_ptr(), 0) },
            0
        );

        d.copy_up(&src, Path::new("/deep/nested/file")).unwrap();

        let upper = d.data_path(Path::new("/deep/nested/file"));
        assert_eq!(std::fs::read(&upper).unwrap(), b"lower bytes");
        let src_meta = std::fs::metadata(&src).unwrap();
        let upper_meta = std::fs::metadata(&upper).unwrap();
        assert_eq!(upper_meta.mode() & 0o7777, 0o640);
        assert_eq!(upper_meta.mtime(), 222);

        let o = origin(&upper).unwrap().expect("copy-up records an origin");
        assert_eq!(o.dev, src_meta.dev());
        assert_eq!(o.ino, src_meta.ino());
        assert_eq!(o.size, src_meta.len());
        assert_eq!(o.mtime_sec, 222);

        // Staging left nothing behind, and a guest-created upper file has no
        // origin.
        assert_eq!(
            std::fs::read_dir(scratch.join("delta/tmp"))
                .unwrap()
                .count(),
            0
        );
        std::fs::write(d.data_path(Path::new("/fresh")), b"y").unwrap();
        assert_eq!(origin(&d.data_path(Path::new("/fresh"))).unwrap(), None);
    }

    /// Copy-up carries the lower's user xattrs — and only those: a forged
    /// reserved-namespace marker on the lower must not become delta
    /// bookkeeping that deletes the file it rides on.
    #[test]
    fn copy_up_carries_xattrs_but_never_markers() {
        use std::os::unix::ffi::OsStrExt;

        let scratch = Scratch::new();
        let Some(d) = delta(&scratch) else { return };

        let src = scratch.join("lower");
        std::fs::write(&src, b"bytes").unwrap();
        let csrc = CString::new(src.as_os_str().as_bytes()).unwrap();
        for (name, value) in [
            (c"user.app", b"v1".as_slice()),
            (c"user.chimera.whiteout", b"1"),
        ] {
            assert_eq!(
                unsafe {
                    libc::setxattr(
                        csrc.as_ptr(),
                        name.as_ptr(),
                        value.as_ptr() as *const libc::c_void,
                        value.len(),
                        0,
                    )
                },
                0
            );
        }

        d.copy_up(&src, Path::new("/f")).unwrap();
        let upper = d.data_path(Path::new("/f"));
        let cupper = CString::new(upper.as_os_str().as_bytes()).unwrap();
        let mut buf = [0u8; 16];
        let n = unsafe {
            libc::getxattr(
                cupper.as_ptr(),
                c"user.app".as_ptr(),
                buf.as_mut_ptr() as *mut libc::c_void,
                buf.len(),
            )
        };
        assert_eq!(n, 2);
        assert_eq!(&buf[..2], b"v1");
        assert!(!is_whiteout(&upper).unwrap());
    }

    /// Publication is first-wins: a later copy-up must not detach an inode
    /// the first publisher has already opened, or the writes acknowledged
    /// through that descriptor would vanish with the unlinked inode.
    #[test]
    fn copy_up_lost_publication_keeps_winning_inode() {
        use std::io::Write;

        let scratch = Scratch::new();
        let Some(d) = delta(&scratch) else { return };

        let src = scratch.join("lower");
        std::fs::write(&src, b"one").unwrap();
        d.copy_up(&src, Path::new("/f")).unwrap();
        let mut winner = std::fs::OpenOptions::new()
            .write(true)
            .open(d.data_path(Path::new("/f")))
            .unwrap();

        // A copy-up that staged before observing the publication loses the
        // rename and reports success, leaving the published entry alone.
        std::fs::write(&src, b"two").unwrap();
        d.copy_up(&src, Path::new("/f")).unwrap();

        // The winner's descriptor still names the inode pathname lookup
        // serves, and the loser left no staging debris behind.
        winner.write_all(b"WIN").unwrap();
        drop(winner);
        assert_eq!(std::fs::read(d.data_path(Path::new("/f"))).unwrap(), b"WIN");
        assert_eq!(
            std::fs::read_dir(scratch.join("delta/tmp"))
                .unwrap()
                .count(),
            0
        );
    }

    /// A multi-block file survives whichever transfer strategy the scratch
    /// filesystem takes (clone, copy_file_range, or the read/write loop).
    #[test]
    fn copy_up_large_file_intact() {
        let scratch = Scratch::new();
        let Some(d) = delta(&scratch) else { return };

        let src = scratch.join("big");
        let payload: Vec<u8> = (0..1_000_000u32).map(|i| i as u8).collect();
        std::fs::write(&src, &payload).unwrap();
        d.copy_up(&src, Path::new("/big")).unwrap();
        assert_eq!(
            std::fs::read(d.data_path(Path::new("/big"))).unwrap(),
            payload
        );
    }
}
