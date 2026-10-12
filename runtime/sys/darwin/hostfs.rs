//! [`HostFs`]: a [`Vfs`] that passes through to a host directory. The reference
//! filesystem — mount one at `/` and the guest sees the real tree under it.
//!
//! Darwin has no `openat2(RESOLVE_IN_ROOT)`, so confinement is split. A
//! `HostFs` rooted at `/` has nothing above it for a symlink or `..` to
//! escape to, and resolves with the kernel's own semantics in one syscall; it
//! is the one case that [`Vfs::confines`]. A `HostFs` rooted anywhere else —
//! the overlay's upper layer, a delta's `data/` tree — is only ever reached
//! through the namespace walker, which has already resolved every symlink in
//! the merged view, so it resolves every path with `O_NOFOLLOW_ANY` and
//! `AT_SYMLINK_NOFOLLOW_ANY`: the kernel refuses any symlink in the chain,
//! and a path that would have escaped the root fails with `ELOOP` instead of
//! leaving it. Identity the guest observes rides through as the host's real
//! `st_ino`/`st_dev` inside [`Stat`].

use std::{
    ffi::{CStr, CString, OsStr, OsString},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::ffi::{OsStrExt, OsStringExt},
    },
    path::{Path, PathBuf},
};

use crate::sys::fs::vfs::{
    DirEntry, Errno, File, FileType, Mode, OpenFlags, RenameFlags, Stat, StatFs, Timespec, Vfs,
    WriteResult,
};

/// Long-lived backing descriptors live at or above this number, keeping the
/// low fd space free for the guest-visible numbers the Personality reserves.
pub const BACKING_FLOOR: libc::c_int = 512;

/// Move a long-lived backing descriptor above [`BACKING_FLOOR`], freeing its
/// low number. The kernel hands `open` the lowest free number, but that low,
/// dense space belongs to the guest: the Personality reserves the guest's
/// numbers there, and idioms like `close(0); open(...)` must observe the
/// freed number again — a hidden backing fd squatting on it would displace
/// them. Best-effort: on failure (fd exhaustion) the low descriptor keeps
/// working, at worst costing numbering fidelity.
///
/// The duplicate is deliberately not `FD_CLOEXEC`: Chimera's emulated execve
/// sweeps host close-on-exec fds by hand, and the backing must survive it —
/// the guest-visible reservation is what exec semantics apply to. (`F_DUPFD`
/// also strips any `O_CLOEXEC` the guest's open flags put on the original.)
fn relocate_high(fd: OwnedFd) -> OwnedFd {
    let high = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_DUPFD, BACKING_FLOOR) };
    if high < 0 {
        return fd;
    }
    unsafe { OwnedFd::from_raw_fd(high) }
}

/// A filesystem backed by a host directory.
pub struct HostFs {
    root: PathBuf,
    /// The root directory, the dirfd every `*at` call is scoped to.
    root_fd: OwnedFd,
    /// Whether the root is somewhere a path could escape from — anywhere but
    /// `/` — in which case no symlink may take part in a resolution.
    confined: bool,
}

impl HostFs {
    /// Root `HostFs` at `root`, which must be an existing directory. The path is
    /// canonicalized so later joins are stable against the caller's cwd.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, Errno> {
        let root = std::fs::canonicalize(root.into()).map_err(|e| Errno::from_io(&e))?;
        let meta = std::fs::symlink_metadata(&root).map_err(|e| Errno::from_io(&e))?;
        if !meta.is_dir() {
            return Err(Errno::ENOTDIR);
        }
        let croot = cpath(&root)?;
        let fd = unsafe {
            libc::open(
                croot.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(last_errno());
        }
        Ok(Self {
            confined: root.as_os_str() != "/",
            root,
            // The root handle lives for the whole run; keep it out of the
            // guest's low fd space like any other backing descriptor.
            root_fd: relocate_high(unsafe { OwnedFd::from_raw_fd(fd) }),
        })
    }

    /// Join a mount-relative guest path onto the host root, for the calls
    /// that have no `*at` form.
    fn host_path(&self, rel: &Path) -> PathBuf {
        let rel = rel.strip_prefix("/").unwrap_or(rel);
        self.root.join(rel)
    }

    /// The `AT_*` resolution flags for a path operation: the final symlink
    /// per `follow`, and no symlink anywhere when confined.
    fn at_flags(&self, follow: bool) -> libc::c_int {
        let mut flags = if follow { 0 } else { libc::AT_SYMLINK_NOFOLLOW };
        if self.confined {
            flags |= AT_SYMLINK_NOFOLLOW_ANY;
        }
        flags
    }

    /// The caller's open flags with confinement applied. A confined open
    /// refuses a symlink anywhere in the path with `O_NOFOLLOW_ANY`, which
    /// subsumes the caller's `O_NOFOLLOW` — and must replace it, since the
    /// kernel rejects the two together with `EINVAL`.
    fn open_flags(&self, flags: libc::c_int) -> libc::c_int {
        if self.confined {
            (flags & !libc::O_NOFOLLOW) | libc::O_NOFOLLOW_ANY
        } else {
            flags
        }
    }

    /// Open `path` (mount-relative) relative to the root.
    fn open_at(&self, path: &Path, flags: libc::c_int, mode: u32) -> Result<OwnedFd, Errno> {
        let crel = crel(path)?;
        let fd = unsafe {
            libc::openat(
                self.root_fd.as_raw_fd(),
                crel.as_ptr(),
                self.open_flags(flags),
                mode as libc::c_uint,
            )
        };
        if fd < 0 {
            return Err(last_errno());
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    fn fstatat(&self, path: &Path, flags: libc::c_int) -> Result<Stat, Errno> {
        let crel = crel(path)?;
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let r = unsafe { libc::fstatat(self.root_fd.as_raw_fd(), crel.as_ptr(), &mut st, flags) };
        if r < 0 {
            return Err(last_errno());
        }
        Ok(stat_to_stat(&st))
    }
}

impl Vfs for HostFs {
    fn open(&self, path: &Path, flags: OpenFlags, mode: Mode) -> Result<Box<dyn File>, Errno> {
        // The guest runs the same OS and ISA, so its O_* flag bits are the
        // host's; forward them verbatim. The handle keeps the low fd the
        // kernel handed out: the Personality detaches it as the guest-visible
        // reservation (`detach_reservation`), relocating the backing high
        // only then.
        let fd = self.open_at(path, flags.raw(), mode.0)?;
        Ok(Box::new(HostFile { fd }))
    }

    fn stat(&self, path: &Path, follow: bool) -> Result<Stat, Errno> {
        let st = self.fstatat(path, self.at_flags(follow))?;
        // `AT_SYMLINK_NOFOLLOW_ANY` leaves a final symlink unfollowed rather
        // than refusing it; a confined mount asked to follow one answers the
        // way it answers every other symlink in its paths.
        if self.confined && follow && st.file_type == FileType::Symlink {
            return Err(Errno::ELOOP);
        }
        Ok(st)
    }

    fn readlink(&self, path: &Path) -> Result<PathBuf, Errno> {
        let crel = crel(path)?;
        let mut buf = vec![0u8; libc::PATH_MAX as usize];
        let n = unsafe {
            libc::readlinkat(
                self.root_fd.as_raw_fd(),
                crel.as_ptr(),
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len(),
            )
        };
        if n < 0 {
            return Err(last_errno());
        }
        buf.truncate(n as usize);
        Ok(PathBuf::from(OsString::from_vec(buf)))
    }

    fn mkdir(&self, path: &Path, mode: Mode) -> Result<(), Errno> {
        let crel = crel(path)?;
        check(unsafe {
            libc::mkdirat(
                self.root_fd.as_raw_fd(),
                crel.as_ptr(),
                mode.0 as libc::mode_t,
            )
        })
    }

    fn unlink(&self, path: &Path) -> Result<(), Errno> {
        let crel = crel(path)?;
        check(unsafe { libc::unlinkat(self.root_fd.as_raw_fd(), crel.as_ptr(), 0) })
    }

    fn rmdir(&self, path: &Path) -> Result<(), Errno> {
        let crel = crel(path)?;
        check(unsafe {
            libc::unlinkat(self.root_fd.as_raw_fd(), crel.as_ptr(), libc::AT_REMOVEDIR)
        })
    }

    fn symlink(&self, target: &Path, link: &Path) -> Result<(), Errno> {
        // `target` is the link's contents, stored verbatim — not resolved
        // against the root. Only the link's own location is a host path.
        let ctarget = cpath(target)?;
        let clink = crel(link)?;
        check(unsafe {
            libc::symlinkat(ctarget.as_ptr(), self.root_fd.as_raw_fd(), clink.as_ptr())
        })
    }

    fn link(&self, old: &Path, new: &Path) -> Result<(), Errno> {
        let cold = crel(old)?;
        let cnew = crel(new)?;
        // No `AT_SYMLINK_FOLLOW`: the new name links to the entry itself.
        check(unsafe {
            libc::linkat(
                self.root_fd.as_raw_fd(),
                cold.as_ptr(),
                self.root_fd.as_raw_fd(),
                cnew.as_ptr(),
                0,
            )
        })
    }

    fn rename(&self, from: &Path, to: &Path, flags: RenameFlags) -> Result<(), Errno> {
        let cfrom = crel(from)?;
        let cto = crel(to)?;
        check(unsafe {
            libc::renameatx_np(
                self.root_fd.as_raw_fd(),
                cfrom.as_ptr(),
                self.root_fd.as_raw_fd(),
                cto.as_ptr(),
                flags.0,
            )
        })
    }

    fn chmod(&self, path: &Path, follow: bool, mode: Mode) -> Result<(), Errno> {
        let crel = crel(path)?;
        check(unsafe {
            libc::fchmodat(
                self.root_fd.as_raw_fd(),
                crel.as_ptr(),
                mode.0 as libc::mode_t,
                self.at_flags(follow),
            )
        })
    }

    fn chown(&self, path: &Path, follow: bool, uid: u32, gid: u32) -> Result<(), Errno> {
        let crel = crel(path)?;
        check(unsafe {
            libc::fchownat(
                self.root_fd.as_raw_fd(),
                crel.as_ptr(),
                uid,
                gid,
                self.at_flags(follow),
            )
        })
    }

    fn utimens(
        &self,
        path: &Path,
        follow: bool,
        times: Option<[Timespec; 2]>,
    ) -> Result<(), Errno> {
        let crel = crel(path)?;
        let ts = times.map(|t| t.map(to_timespec));
        let tp = ts.as_ref().map_or(std::ptr::null(), |t| t.as_ptr());
        check(unsafe {
            libc::utimensat(
                self.root_fd.as_raw_fd(),
                crel.as_ptr(),
                tp,
                self.at_flags(follow),
            )
        })
    }

    fn mknod(&self, path: &Path, mode: Mode, dev: u64) -> Result<(), Errno> {
        let crel = crel(path)?;
        // A FIFO is the one node an unprivileged process may create, and
        // `mknod` refuses it; `mkfifo` is the call that makes it.
        if mode.0 & libc::S_IFMT as u32 == libc::S_IFIFO as u32 {
            return check(unsafe {
                libc::mkfifoat(
                    self.root_fd.as_raw_fd(),
                    crel.as_ptr(),
                    (mode.0 & 0o7777) as libc::mode_t,
                )
            });
        }
        let cpath = cpath(&self.host_path(path))?;
        check(unsafe { libc::mknod(cpath.as_ptr(), mode.0 as libc::mode_t, dev as libc::dev_t) })
    }

    fn setxattr(
        &self,
        path: &Path,
        follow: bool,
        name: &OsStr,
        value: &[u8],
        flags: i32,
    ) -> Result<(), Errno> {
        let cpath = cpath(&self.host_path(path))?;
        let cname = CString::new(name.as_bytes()).map_err(|_| Errno::EINVAL)?;
        check(unsafe {
            libc::setxattr(
                cpath.as_ptr(),
                cname.as_ptr(),
                value.as_ptr() as *const libc::c_void,
                value.len(),
                0,
                flags | self.xattr_options(follow),
            )
        })
    }

    fn removexattr(&self, path: &Path, follow: bool, name: &OsStr) -> Result<(), Errno> {
        let cpath = cpath(&self.host_path(path))?;
        let cname = CString::new(name.as_bytes()).map_err(|_| Errno::EINVAL)?;
        check(unsafe {
            libc::removexattr(cpath.as_ptr(), cname.as_ptr(), self.xattr_options(follow))
        })
    }

    fn statfs(&self, path: &Path) -> Result<StatFs, Errno> {
        let fd = self.open_at(path, libc::O_RDONLY | libc::O_NONBLOCK, 0)?;
        Ok(statfs_to_statfs(&raw_fstatfs(fd.as_raw_fd())?))
    }

    fn resolve_fast(&self, path: &Path) -> Option<Stat> {
        // One fstatat answers both questions: `AT_SYMLINK_NOFOLLOW_ANY`
        // proves no symlink leads to the object (ELOOP otherwise) and the
        // stat is the object itself — unless that object is a symlink,
        // which the flag leaves unfollowed rather than refusing. Every
        // failure falls back to the walk.
        self.fstatat(path, AT_SYMLINK_NOFOLLOW_ANY)
            .ok()
            .filter(|st| st.file_type != FileType::Symlink)
    }

    fn stat_fast(&self, path: &Path, follow: bool) -> Option<Result<Stat, Errno>> {
        if !self.confined {
            return Some(self.stat(path, follow));
        }
        // A confined mount can still answer wholesale when no symlink takes
        // part — which `AT_SYMLINK_NOFOLLOW_ANY` proves after the fact, and
        // which also means the path could not have left the root — provided
        // the path carries no `..`: that would have to be applied after a
        // symlink expansion the flag then hides. A miss is authoritative,
        // since the kernel reports a symlink before any later component.
        if path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return None;
        }
        match self.fstatat(path, AT_SYMLINK_NOFOLLOW_ANY) {
            Ok(st) if follow && st.file_type == FileType::Symlink => None,
            Ok(st) => Some(Ok(st)),
            Err(e @ (Errno::ENOENT | Errno::ENOTDIR)) => Some(Err(e)),
            Err(_) => None,
        }
    }

    fn materialize(&self, path: &Path) -> Result<PathBuf, Errno> {
        Ok(HostFs::host_path(self, path))
    }

    fn host_path(&self, path: &Path) -> Option<PathBuf> {
        Some(HostFs::host_path(self, path))
    }

    fn confines(&self) -> bool {
        !self.confined
    }
}

impl HostFs {
    /// The xattr options for a path operation: the final symlink per
    /// `follow`, and no symlink anywhere when confined.
    fn xattr_options(&self, follow: bool) -> libc::c_int {
        let mut options = if follow { 0 } else { libc::XATTR_NOFOLLOW };
        if self.confined {
            options |= XATTR_NOFOLLOW_ANY;
        }
        options
    }
}

/// `NAME_MAX` (`<sys/syslimits.h>`): the longest entry name.
const NAME_MAX: u64 = 255;

/// `XATTR_NOFOLLOW_ANY` (`<sys/xattr.h>`) and `AT_SYMLINK_NOFOLLOW_ANY`
/// (`<sys/fcntl.h>`): refuse any symlink in the path. The libc crate has no
/// binding for either.
const XATTR_NOFOLLOW_ANY: libc::c_int = 0x0040;
pub const AT_SYMLINK_NOFOLLOW_ANY: libc::c_int = 0x0800;

/// An open host file or directory: a real fd, with the read/write offset held
/// by the caller (the fd table), so `pread`/`pwrite` carry it explicitly.
pub struct HostFile {
    fd: OwnedFd,
}

impl File for HostFile {
    fn pread(&self, buf: &mut [u8], offset: u64) -> Result<usize, Errno> {
        let mut n = unsafe {
            libc::pread(
                self.fd.as_raw_fd(),
                buf.as_mut_ptr() as *mut libc::c_void,
                buf.len(),
                offset as libc::off_t,
            )
        };
        // A tty, FIFO, or socket has no file position, so pread on one fails
        // — ESPIPE for a pipe or socket, ENXIO for a terminal here — but
        // read(2) just transfers bytes. The Personality reads every open
        // file at its table-tracked offset; a non-seekable file ignores that
        // offset exactly the way the kernel does.
        if n < 0 && unseekable(last_errno()) {
            n = unsafe {
                libc::read(
                    self.fd.as_raw_fd(),
                    buf.as_mut_ptr() as *mut libc::c_void,
                    buf.len(),
                )
            };
        }
        if n < 0 {
            return Err(last_errno());
        }
        Ok(n as usize)
    }

    fn pwrite(&self, buf: &[u8], offset: u64) -> Result<usize, Errno> {
        let mut n = unsafe {
            libc::pwrite(
                self.fd.as_raw_fd(),
                buf.as_ptr() as *const libc::c_void,
                buf.len(),
                offset as libc::off_t,
            )
        };
        // See pread: write(2) semantics for the files pwrite cannot serve.
        if n < 0 && unseekable(last_errno()) {
            n = unsafe {
                libc::write(
                    self.fd.as_raw_fd(),
                    buf.as_ptr() as *const libc::c_void,
                    buf.len(),
                )
            };
        }
        if n < 0 {
            return Err(last_errno());
        }
        Ok(n as usize)
    }

    fn append(&self, buf: &[u8]) -> Result<WriteResult, Errno> {
        let n = unsafe {
            libc::write(
                self.fd.as_raw_fd(),
                buf.as_ptr() as *const libc::c_void,
                buf.len(),
            )
        };
        if n < 0 {
            return Err(last_errno());
        }
        let offset = match unsafe { libc::lseek(self.fd.as_raw_fd(), 0, libc::SEEK_CUR) } {
            off if off >= 0 => off as u64,
            _ if unseekable(last_errno()) => n as u64,
            _ => return Err(last_errno()),
        };
        Ok(WriteResult {
            count: n as usize,
            offset,
        })
    }

    fn fstat(&self) -> Result<Stat, Errno> {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(self.fd.as_raw_fd(), &mut st) } < 0 {
            return Err(last_errno());
        }
        Ok(stat_to_stat(&st))
    }

    fn fstatfs(&self) -> Result<StatFs, Errno> {
        Ok(statfs_to_statfs(&raw_fstatfs(self.fd.as_raw_fd())?))
    }

    fn ftruncate(&self, len: u64) -> Result<(), Errno> {
        check(unsafe { libc::ftruncate(self.fd.as_raw_fd(), len as libc::off_t) })
    }

    fn fsync(&self) -> Result<(), Errno> {
        check(unsafe { libc::fsync(self.fd.as_raw_fd()) })
    }

    fn fchmod(&self, mode: Mode) -> Result<(), Errno> {
        check(unsafe { libc::fchmod(self.fd.as_raw_fd(), mode.0 as libc::mode_t) })
    }

    fn fchown(&self, uid: u32, gid: u32) -> Result<(), Errno> {
        check(unsafe { libc::fchown(self.fd.as_raw_fd(), uid, gid) })
    }

    fn futimens(&self, times: Option<[Timespec; 2]>) -> Result<(), Errno> {
        let ts = times.map(|t| t.map(to_timespec));
        let tp = ts.as_ref().map_or(std::ptr::null(), |t| t.as_ptr());
        check(unsafe { libc::futimens(self.fd.as_raw_fd(), tp) })
    }

    fn fsetxattr(&self, name: &OsStr, value: &[u8], flags: i32) -> Result<(), Errno> {
        let cname = CString::new(name.as_bytes()).map_err(|_| Errno::EINVAL)?;
        check(unsafe {
            libc::fsetxattr(
                self.fd.as_raw_fd(),
                cname.as_ptr(),
                value.as_ptr() as *const libc::c_void,
                value.len(),
                0,
                flags,
            )
        })
    }

    fn fremovexattr(&self, name: &OsStr) -> Result<(), Errno> {
        let cname = CString::new(name.as_bytes()).map_err(|_| Errno::EINVAL)?;
        check(unsafe { libc::fremovexattr(self.fd.as_raw_fd(), cname.as_ptr(), 0) })
    }

    fn fallocate(&self, _mode: i32, _offset: u64, _len: u64) -> Result<(), Errno> {
        // No such call on Darwin; the guest's `F_PREALLOCATE` reaches the
        // kernel through the descriptor itself.
        Err(Errno(libc::ENOTSUP))
    }

    fn host_fd(&self) -> Option<libc::c_int> {
        Some(self.fd.as_raw_fd())
    }

    fn detach_reservation(&mut self) -> Option<OwnedFd> {
        // The relocation dup is deliberately not FD_CLOEXEC; see
        // `relocate_high`. On dup failure keep the low backing — the caller
        // reserves off it as before, costing numbering fidelity, not
        // correctness.
        let high = unsafe { libc::fcntl(self.fd.as_raw_fd(), libc::F_DUPFD, BACKING_FLOOR) };
        if high < 0 {
            return None;
        }
        Some(std::mem::replace(&mut self.fd, unsafe {
            OwnedFd::from_raw_fd(high)
        }))
    }

    fn getdents(&self) -> Result<Vec<DirEntry>, Errno> {
        // `fdopendir` adopts the fd it is given (and `closedir` closes it), so
        // hand it a dup and keep ours. Rewind first: each call returns a fresh
        // full snapshot, since the guest's read position lives in the fd table.
        let dupfd = unsafe { libc::dup(self.fd.as_raw_fd()) };
        if dupfd < 0 {
            return Err(last_errno());
        }
        let dir = unsafe { libc::fdopendir(dupfd) };
        if dir.is_null() {
            let e = last_errno();
            unsafe { libc::close(dupfd) };
            return Err(e);
        }
        unsafe { libc::rewinddir(dir) };

        let mut entries = Vec::new();
        let err = loop {
            // `readdir` returns null both at end-of-stream and on error; clear
            // errno first so the two cases can be told apart afterwards.
            unsafe { *libc::__error() = 0 };
            let ent = unsafe { libc::readdir(dir) };
            if ent.is_null() {
                break unsafe { *libc::__error() };
            }
            let name = unsafe { CStr::from_ptr((*ent).d_name.as_ptr()) }.to_bytes();
            // `.`/`..` are the Personality's to synthesize (it knows the dir's
            // and parent's inode numbers); a Vfs reports only real entries.
            if name == b"." || name == b".." {
                continue;
            }
            let d_type = unsafe { (*ent).d_type };
            let file_type = file_type_from_dtype(d_type)
                .map(Ok)
                .unwrap_or_else(|| self.entry_type_fallback(name))?;
            entries.push(DirEntry {
                ino: unsafe { (*ent).d_ino },
                file_type,
                name: OsString::from_vec(name.to_vec()),
            });
        };
        unsafe { libc::closedir(dir) };
        if err != 0 {
            return Err(Errno(err));
        }
        Ok(entries)
    }
}

impl HostFile {
    /// Resolve a directory entry's type when `readdir` reports `DT_UNKNOWN`:
    /// `fstatat` the name against this open dir fd without following a final
    /// symlink.
    fn entry_type_fallback(&self, name: &[u8]) -> Result<FileType, Errno> {
        let cname = CString::new(name).map_err(|_| Errno::EINVAL)?;
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let r = unsafe {
            libc::fstatat(
                self.fd.as_raw_fd(),
                cname.as_ptr(),
                &mut st,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if r < 0 {
            return Err(last_errno());
        }
        Ok(file_type_from_mode(st.st_mode))
    }
}

/// `CString` from a path's bytes; an embedded NUL is `EINVAL`.
fn cpath(path: &Path) -> Result<CString, Errno> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| Errno::EINVAL)
}

/// A mount-relative path as a `root_fd`-relative C string: strip the leading
/// `/`, and name the root itself as ".".
fn crel(path: &Path) -> Result<CString, Errno> {
    let rel = path.strip_prefix("/").unwrap_or(path);
    let bytes = rel.as_os_str().as_bytes();
    CString::new(if bytes.is_empty() {
        b"." as &[u8]
    } else {
        bytes
    })
    .map_err(|_| Errno::EINVAL)
}

/// A symbolic [`Timespec`] as the C struct, `UTIME_*` specials intact.
fn to_timespec(t: Timespec) -> libc::timespec {
    libc::timespec {
        tv_sec: t.sec,
        tv_nsec: t.nsec,
    }
}

/// Whether a positional call failed because the file has no position.
fn unseekable(e: Errno) -> bool {
    matches!(e, Errno(libc::ESPIPE) | Errno(libc::ENXIO))
}

/// Map a libc return of `-1` to the current errno, anything else to `Ok`.
fn check(ret: libc::c_int) -> Result<(), Errno> {
    if ret < 0 { Err(last_errno()) } else { Ok(()) }
}

/// The current errno, wrapped.
fn last_errno() -> Errno {
    Errno::from_io(&std::io::Error::last_os_error())
}

pub fn file_type_from_mode(mode: libc::mode_t) -> FileType {
    match mode & libc::S_IFMT {
        libc::S_IFDIR => FileType::Directory,
        libc::S_IFLNK => FileType::Symlink,
        libc::S_IFCHR => FileType::CharDevice,
        libc::S_IFBLK => FileType::BlockDevice,
        libc::S_IFIFO => FileType::Fifo,
        libc::S_IFSOCK => FileType::Socket,
        _ => FileType::Regular,
    }
}

fn file_type_from_dtype(d_type: u8) -> Option<FileType> {
    Some(match d_type {
        libc::DT_REG => FileType::Regular,
        libc::DT_DIR => FileType::Directory,
        libc::DT_LNK => FileType::Symlink,
        libc::DT_CHR => FileType::CharDevice,
        libc::DT_BLK => FileType::BlockDevice,
        libc::DT_FIFO => FileType::Fifo,
        libc::DT_SOCK => FileType::Socket,
        _ => return None,
    })
}

/// The symbolic view of a Darwin `struct stat`. `dev_t` and the node counts
/// are narrower here than the symbolic fields; they widen losslessly and the
/// Personality narrows them back for the guest.
pub fn stat_to_stat(st: &libc::stat) -> Stat {
    Stat {
        dev: st.st_dev as u64,
        ino: st.st_ino,
        file_type: file_type_from_mode(st.st_mode),
        mode: Mode(st.st_mode as u32),
        nlink: st.st_nlink as u64,
        uid: st.st_uid,
        gid: st.st_gid,
        rdev: st.st_rdev as u64,
        size: st.st_size as u64,
        blksize: st.st_blksize as u64,
        blocks: st.st_blocks as u64,
        atime: Timespec {
            sec: st.st_atime,
            nsec: st.st_atime_nsec,
        },
        mtime: Timespec {
            sec: st.st_mtime,
            nsec: st.st_mtime_nsec,
        },
        ctime: Timespec {
            sec: st.st_ctime,
            nsec: st.st_ctime_nsec,
        },
        btime: Timespec {
            sec: st.st_birthtime,
            nsec: st.st_birthtime_nsec,
        },
        flags: st.st_flags,
    }
}

fn statfs_to_statfs(st: &libc::statfs) -> StatFs {
    StatFs {
        fs_type: st.f_type as i64,
        block_size: st.f_bsize as u64,
        fragment_size: st.f_bsize as u64,
        blocks: st.f_blocks,
        blocks_free: st.f_bfree,
        blocks_available: st.f_bavail,
        files: st.f_files,
        files_free: st.f_ffree,
        // The libc crate keeps `fsid_t`'s pair private; it is two `i32`s.
        fsid: unsafe { std::mem::transmute::<libc::fsid_t, [i32; 2]>(st.f_fsid) },
        name_max: NAME_MAX,
        flags: st.f_flags as u64,
    }
}

fn raw_fstatfs(fd: libc::c_int) -> Result<libc::statfs, Errno> {
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstatfs(fd, &mut st) } < 0 {
        return Err(last_errno());
    }
    Ok(st)
}

#[cfg(test)]
mod tests {
    use std::{
        os::unix::fs::MetadataExt,
        sync::atomic::{AtomicU32, Ordering},
    };

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
                std::env::temp_dir().join(format!("chimera-hostfs-{}-{}", std::process::id(), n));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn fs(scratch: &Scratch) -> HostFs {
        HostFs::new(&scratch.path).unwrap()
    }

    #[test]
    fn open_write_read_roundtrip() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);

        let flags = OpenFlags(libc::O_CREAT | libc::O_RDWR | libc::O_APPEND);
        let file = fs.open(Path::new("hello.txt"), flags, Mode(0o644)).unwrap();
        assert_eq!(file.pwrite(b"hello world", 0).unwrap(), 11);

        let mut buf = [0u8; 11];
        assert_eq!(file.pread(&mut buf, 0).unwrap(), 11);
        assert_eq!(&buf, b"hello world");

        let write = file.append(b"!").unwrap();
        assert_eq!(write.offset, 12);
        assert_eq!(file.fstat().unwrap().size, 12);
    }

    /// A `/`-rooted mount resolves with kernel semantics and agrees with the
    /// host's own view on identity; a confined mount refuses any symlink.
    #[test]
    fn root_follows_and_confined_refuses_symlinks() {
        let scratch = Scratch::new();
        let base = scratch.path.canonicalize().unwrap();
        std::fs::write(base.join("f"), b"x").unwrap();
        std::os::unix::fs::symlink("f", base.join("link")).unwrap();
        std::fs::create_dir(base.join("d")).unwrap();
        std::os::unix::fs::symlink("..", base.join("d/up")).unwrap();

        let root = HostFs::new("/").unwrap();
        assert!(root.confines());
        let st = root.stat(&base.join("f"), true).unwrap();
        let meta = std::fs::metadata(base.join("f")).unwrap();
        assert_eq!(st.ino, meta.ino());
        assert_eq!(st.dev, meta.dev() as u64);
        assert_eq!(
            root.stat(&base.join("link"), true).unwrap().file_type,
            FileType::Regular
        );
        assert_eq!(
            root.stat(&base.join("link"), false).unwrap().file_type,
            FileType::Symlink
        );
        assert_eq!(
            root.stat(&base.join("d/up/f"), true).unwrap().ino,
            meta.ino()
        );

        let confined = HostFs::new(&base).unwrap();
        assert!(!confined.confines());
        assert_eq!(
            confined.stat(Path::new("/link"), false).unwrap().file_type,
            FileType::Symlink
        );
        assert_eq!(
            confined.stat(Path::new("/link"), true).err(),
            Some(Errno::ELOOP)
        );
        assert_eq!(
            confined.stat(Path::new("/d/up/f"), true).err(),
            Some(Errno::ELOOP)
        );
        assert_eq!(
            confined
                .open(Path::new("/d/up/f"), OpenFlags(libc::O_RDONLY), Mode(0))
                .err(),
            Some(Errno::ELOOP)
        );
        assert!(confined.resolve_fast(Path::new("/f")).is_some());
        assert!(confined.resolve_fast(Path::new("/link")).is_none());
    }

    #[test]
    fn mkdir_then_getdents_lists_entries() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);

        fs.mkdir(Path::new("d"), Mode(0o755)).unwrap();
        for name in ["d/a", "d/b"] {
            fs.open(
                Path::new(name),
                OpenFlags(libc::O_CREAT | libc::O_WRONLY),
                Mode(0o644),
            )
            .unwrap();
        }
        let dir = fs
            .open(
                Path::new("d"),
                OpenFlags(libc::O_RDONLY | libc::O_DIRECTORY),
                Mode(0),
            )
            .unwrap();
        let mut names: Vec<String> = dir
            .getdents()
            .unwrap()
            .into_iter()
            .map(|e| e.name.to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(dir.getdents().unwrap().len(), 2);
    }

    #[test]
    fn rename_flags_and_links() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);
        std::fs::write(scratch.path.join("a"), b"a").unwrap();
        std::fs::write(scratch.path.join("b"), b"b").unwrap();

        assert_eq!(
            fs.rename(Path::new("a"), Path::new("b"), RenameFlags::NOREPLACE),
            Err(Errno::EEXIST)
        );
        fs.rename(Path::new("a"), Path::new("b"), RenameFlags::EXCHANGE)
            .unwrap();
        assert_eq!(std::fs::read(scratch.path.join("a")).unwrap(), b"b");
        fs.rename(Path::new("a"), Path::new("c"), RenameFlags::EMPTY)
            .unwrap();
        assert_eq!(fs.stat(Path::new("a"), false).err(), Some(Errno::ENOENT));

        fs.link(Path::new("c"), Path::new("alias")).unwrap();
        assert_eq!(fs.stat(Path::new("alias"), false).unwrap().nlink, 2);
        fs.symlink(Path::new("c"), Path::new("sl")).unwrap();
        assert_eq!(fs.readlink(Path::new("sl")).unwrap(), PathBuf::from("c"));
    }

    #[test]
    fn metadata_mutators_and_fifo() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);
        let file = fs
            .open(
                Path::new("f"),
                OpenFlags(libc::O_CREAT | libc::O_RDWR),
                Mode(0o644),
            )
            .unwrap();
        fs.chmod(Path::new("f"), true, Mode(0o600)).unwrap();
        assert_eq!(
            fs.stat(Path::new("f"), true).unwrap().mode.0 & 0o7777,
            0o600
        );
        file.fchmod(Mode(0o640)).unwrap();
        assert_eq!(
            std::fs::metadata(scratch.path.join("f")).unwrap().mode() & 0o7777,
            0o640
        );
        fs.chown(Path::new("f"), true, u32::MAX, u32::MAX).unwrap();
        let times = [
            Timespec { sec: 111, nsec: 0 },
            Timespec { sec: 222, nsec: 0 },
        ];
        fs.utimens(Path::new("f"), true, Some(times)).unwrap();
        let st = fs.stat(Path::new("f"), true).unwrap();
        assert_eq!((st.atime.sec, st.mtime.sec), (111, 222));
        file.futimens(Some([
            Timespec { sec: 333, nsec: 0 },
            Timespec { sec: 444, nsec: 0 },
        ]))
        .unwrap();
        assert_eq!(file.fstat().unwrap().mtime.sec, 444);

        fs.mknod(Path::new("pipe"), Mode(libc::S_IFIFO as u32 | 0o644), 0)
            .unwrap();
        assert_eq!(
            fs.stat(Path::new("pipe"), true).unwrap().file_type,
            FileType::Fifo
        );

        let name = OsStr::new("user.chimera-test");
        fs.setxattr(Path::new("f"), true, name, b"v", 0).unwrap();
        file.fsetxattr(name, b"w", 0).unwrap();
        fs.removexattr(Path::new("f"), true, name).unwrap();
        assert_eq!(
            fs.removexattr(Path::new("f"), true, name).unwrap_err(),
            Errno(libc::ENOATTR)
        );
    }
}
