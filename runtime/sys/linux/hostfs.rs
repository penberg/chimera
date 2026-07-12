//! [`HostFs`]: a [`Vfs`] that passes through to a host directory. The reference
//! filesystem — mount one at `/` and the guest sees the real tree under it.
//!
//! `HostFs` confines itself: it holds an `O_PATH` descriptor for its root and
//! resolves the read path with `openat2(RESOLVE_IN_ROOT)`, so the kernel follows
//! symlinks and `..` scoped to the root in a single syscall. Because it confines
//! ([`Vfs::confines`] returns `true`), the namespace skips its per-component
//! walk for a single `HostFs` mount and hands the raw path straight here — the
//! kernel does the resolution the walk would otherwise do one `stat` at a time.
//! Identity the guest observes still rides through as the host's real
//! `st_ino`/`st_dev` inside [`Stat`].

// Scaffolding: no Personality drives this yet, so the public surface is unused
// outside the tests below. Drop this once it is wired in.
#![allow(dead_code)]

use std::{
    ffi::{CStr, CString, OsString},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::ffi::{OsStrExt, OsStringExt},
    },
    path::{Path, PathBuf},
};

use super::vfs::{
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

/// Linux x86-64 `struct statfs`, including the fields Rust's `libc` binding
/// still hides behind a stale definition.
#[repr(C)]
struct RawStatFs {
    f_type: libc::c_long,
    f_bsize: libc::c_long,
    f_blocks: u64,
    f_bfree: u64,
    f_bavail: u64,
    f_files: u64,
    f_ffree: u64,
    f_fsid: [i32; 2],
    f_namelen: libc::c_long,
    f_frsize: libc::c_long,
    f_flags: libc::c_long,
    f_spare: [libc::c_long; 4],
}

/// `resolve` flags for `openat2` (`<linux/openat2.h>`; libc has no binding).
/// `RESOLVE_IN_ROOT` scopes the whole walk — `..` and absolute symlinks
/// included — to the dirfd, which is exactly the namespace's confinement model.
const RESOLVE_IN_ROOT: u64 = 0x10;

/// `struct open_how` for `openat2`.
#[repr(C)]
struct OpenHow {
    flags: u64,
    mode: u64,
    resolve: u64,
}

/// A filesystem backed by a host directory. Read operations resolve through
/// `root_fd` with `openat2(RESOLVE_IN_ROOT)`, confined to the root in-kernel.
pub struct HostFs {
    root: PathBuf,
    /// `O_PATH` descriptor for the root, the dirfd every `openat2` is scoped to.
    root_fd: OwnedFd,
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
                libc::O_PATH | libc::O_DIRECTORY | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(last_errno());
        }
        Ok(Self {
            root,
            // The root handle lives for the whole run; keep it out of the
            // guest's low fd space like any other backing descriptor.
            root_fd: relocate_high(unsafe { OwnedFd::from_raw_fd(fd) }),
        })
    }

    /// Join a mount-relative guest path onto the host root. Used by the path
    /// mutators, which are not yet `openat2`-confined (they are reachable only
    /// under a writable mount).
    fn host_path(&self, rel: &Path) -> PathBuf {
        let rel = rel.strip_prefix("/").unwrap_or(rel);
        self.root.join(rel)
    }

    /// Open `path` (mount-relative) confined to the root via `openat2`. `flags`
    /// is the raw open-flag set; the kernel scopes resolution with
    /// `RESOLVE_IN_ROOT`, so symlinks and `..` cannot escape.
    fn open_in_root(&self, path: &Path, flags: libc::c_int, mode: u32) -> Result<OwnedFd, Errno> {
        let crel = crel(path)?;
        let how = OpenHow {
            flags: flags as u64,
            mode: mode as u64,
            resolve: RESOLVE_IN_ROOT,
        };
        let fd = unsafe {
            libc::syscall(
                libc::SYS_openat2,
                self.root_fd.as_raw_fd(),
                crel.as_ptr(),
                &how as *const OpenHow,
                std::mem::size_of::<OpenHow>(),
            )
        };
        if fd < 0 {
            return Err(last_errno());
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fd as libc::c_int) })
    }
}

impl Vfs for HostFs {
    fn open(&self, path: &Path, flags: OpenFlags, mode: Mode) -> Result<Box<dyn File>, Errno> {
        // The guest runs the same OS and ISA, so its O_* flag bits are the
        // host's; forward them verbatim. The kernel confines resolution to the
        // root, and honors O_NOFOLLOW on the final component. The handle keeps
        // the low fd the kernel handed out: the Personality detaches it as the
        // guest-visible reservation (`detach_reservation`), relocating the
        // backing high only then.
        let fd = self.open_in_root(path, flags.raw(), mode.0)?;
        Ok(Box::new(HostFile { fd }))
    }

    fn stat(&self, path: &Path, follow: bool) -> Result<Stat, Errno> {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        // Confining to the host's own root is vacuous — there is nothing above
        // `/` for a symlink or `..` to escape to — so a plain fstatat against
        // root_fd resolves identically, in one syscall instead of the
        // openat2 + fstat + close triple.
        if self.root.as_os_str() == "/" {
            let crel = crel(path)?;
            let flags = if follow { 0 } else { libc::AT_SYMLINK_NOFOLLOW };
            let r =
                unsafe { libc::fstatat(self.root_fd.as_raw_fd(), crel.as_ptr(), &mut st, flags) };
            if r < 0 {
                return Err(last_errno());
            }
            return Ok(stat_to_stat(&st));
        }
        // Resolve to an O_PATH handle confined to the root, then stat the
        // handle. O_NOFOLLOW makes the handle name a final symlink itself
        // (lstat); without it the final symlink is followed (stat).
        let oflags = libc::O_PATH | if follow { 0 } else { libc::O_NOFOLLOW };
        let fd = self.open_in_root(path, oflags, 0)?;
        let r =
            unsafe { libc::fstatat(fd.as_raw_fd(), c"".as_ptr(), &mut st, libc::AT_EMPTY_PATH) };
        if r < 0 {
            return Err(last_errno());
        }
        Ok(stat_to_stat(&st))
    }

    fn readlink(&self, path: &Path) -> Result<PathBuf, Errno> {
        // O_PATH | O_NOFOLLOW names the symlink itself; readlinkat with an empty
        // path then reads its target.
        let fd = self.open_in_root(path, libc::O_PATH | libc::O_NOFOLLOW, 0)?;
        let mut buf = vec![0u8; libc::PATH_MAX as usize];
        let n = unsafe {
            libc::readlinkat(
                fd.as_raw_fd(),
                c"".as_ptr(),
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len(),
            )
        };
        if n < 0 {
            let e = last_errno();
            // The empty-path readlinkat reports ENOENT when the handle is not
            // a symlink, but readlink(2) by path says EINVAL — and glibc's
            // realpath depends on EINVAL to mean "a real component, keep
            // walking". The open above already proved the path exists, so
            // ENOENT here can only mean "not a symlink".
            return Err(if e == Errno::ENOENT { Errno::EINVAL } else { e });
        }
        buf.truncate(n as usize);
        Ok(PathBuf::from(OsString::from_vec(buf)))
    }

    fn mkdir(&self, path: &Path, mode: Mode) -> Result<(), Errno> {
        let cpath = cpath(&self.host_path(path))?;
        check(unsafe { libc::mkdir(cpath.as_ptr(), mode.0 as libc::mode_t) })
    }

    fn unlink(&self, path: &Path) -> Result<(), Errno> {
        let cpath = cpath(&self.host_path(path))?;
        check(unsafe { libc::unlink(cpath.as_ptr()) })
    }

    fn rmdir(&self, path: &Path) -> Result<(), Errno> {
        let cpath = cpath(&self.host_path(path))?;
        check(unsafe { libc::rmdir(cpath.as_ptr()) })
    }

    fn symlink(&self, target: &Path, link: &Path) -> Result<(), Errno> {
        // `target` is the link's contents, stored verbatim — not resolved
        // against the root. Only the link's own location is a host path.
        let ctarget = cpath(target)?;
        let clink = cpath(&self.host_path(link))?;
        check(unsafe { libc::symlink(ctarget.as_ptr(), clink.as_ptr()) })
    }

    fn link(&self, old: &Path, new: &Path) -> Result<(), Errno> {
        let cold = cpath(&self.host_path(old))?;
        let cnew = cpath(&self.host_path(new))?;
        check(unsafe { libc::link(cold.as_ptr(), cnew.as_ptr()) })
    }

    fn rename(&self, from: &Path, to: &Path, flags: RenameFlags) -> Result<(), Errno> {
        let cfrom = cpath(&self.host_path(from))?;
        let cto = cpath(&self.host_path(to))?;
        let r = if flags.0 == 0 {
            unsafe { libc::rename(cfrom.as_ptr(), cto.as_ptr()) }
        } else {
            // No direct libc binding for renameat2; issue the raw syscall.
            unsafe {
                libc::syscall(
                    libc::SYS_renameat2,
                    libc::AT_FDCWD,
                    cfrom.as_ptr(),
                    libc::AT_FDCWD,
                    cto.as_ptr(),
                    flags.0 as libc::c_uint,
                ) as libc::c_int
            }
        };
        check(r)
    }

    fn chmod(&self, path: &Path, follow: bool, mode: Mode) -> Result<(), Errno> {
        let cpath = cpath(&self.host_path(path))?;
        let r = if follow {
            unsafe { libc::chmod(cpath.as_ptr(), mode.0 as libc::mode_t) }
        } else {
            unsafe {
                libc::syscall(
                    libc::SYS_fchmodat2,
                    libc::AT_FDCWD,
                    cpath.as_ptr(),
                    mode.0 as libc::mode_t,
                    libc::AT_SYMLINK_NOFOLLOW,
                ) as libc::c_int
            }
        };
        check(r)
    }

    fn statfs(&self, path: &Path) -> Result<StatFs, Errno> {
        let fd = self.open_in_root(path, libc::O_PATH, 0)?;
        Ok(statfs_to_statfs(&raw_fstatfs(fd.as_raw_fd())?))
    }

    fn host_path(&self, path: &Path) -> Option<PathBuf> {
        Some(HostFs::host_path(self, path))
    }

    fn confines(&self) -> bool {
        true
    }
}

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
        // A tty, FIFO, or socket has no file position, so pread on one is
        // ESPIPE — but read(2) just transfers bytes. The Personality reads
        // every open file at its table-tracked offset; a non-seekable file
        // ignores that offset exactly the way the kernel does. (Bun reopens
        // its controlling terminal by path and reads input through it.)
        if n < 0 && unsafe { *libc::__errno_location() } == libc::ESPIPE {
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
        if n < 0 && unsafe { *libc::__errno_location() } == libc::ESPIPE {
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
            _ if unsafe { *libc::__errno_location() } == libc::ESPIPE => n as u64,
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

    fn fchmodat_empty(&self, mode: Mode) -> Result<(), Errno> {
        check(unsafe {
            libc::syscall(
                libc::SYS_fchmodat2,
                self.fd.as_raw_fd(),
                c"".as_ptr(),
                mode.0 as libc::mode_t,
                libc::AT_EMPTY_PATH,
            ) as libc::c_int
        })
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
            unsafe { *libc::__errno_location() = 0 };
            let ent = unsafe { libc::readdir(dir) };
            if ent.is_null() {
                break unsafe { *libc::__errno_location() };
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
    /// Resolve a directory entry's type when `readdir` reports `DT_UNKNOWN`
    /// (some filesystems do): `fstatat` the name against this open dir fd
    /// without following a final symlink.
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

/// Map a libc return of `-1` to the current errno, anything else to `Ok`.
fn check(ret: libc::c_int) -> Result<(), Errno> {
    if ret < 0 { Err(last_errno()) } else { Ok(()) }
}

/// The current errno, wrapped.
fn last_errno() -> Errno {
    Errno::from_io(&std::io::Error::last_os_error())
}

fn file_type_from_mode(mode: libc::mode_t) -> FileType {
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

fn stat_to_stat(st: &libc::stat) -> Stat {
    Stat {
        dev: st.st_dev,
        ino: st.st_ino,
        file_type: file_type_from_mode(st.st_mode),
        mode: Mode(st.st_mode),
        nlink: st.st_nlink,
        uid: st.st_uid,
        gid: st.st_gid,
        rdev: st.st_rdev,
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
    }
}

fn statfs_to_statfs(st: &RawStatFs) -> StatFs {
    StatFs {
        fs_type: st.f_type,
        block_size: st.f_bsize as u64,
        fragment_size: st.f_frsize as u64,
        blocks: st.f_blocks,
        blocks_free: st.f_bfree,
        blocks_available: st.f_bavail,
        files: st.f_files,
        files_free: st.f_ffree,
        fsid: st.f_fsid,
        name_max: st.f_namelen as u64,
        flags: st.f_flags as u64,
    }
}

fn raw_fstatfs(fd: libc::c_int) -> Result<RawStatFs, Errno> {
    let mut st: RawStatFs = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::syscall(libc::SYS_fstatfs, fd, &mut st) };
    if r < 0 {
        return Err(last_errno());
    }
    Ok(st)
}

fn raw_statfs(path: &CStr) -> Result<RawStatFs, Errno> {
    let mut st: RawStatFs = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::syscall(libc::SYS_statfs, path.as_ptr(), &mut st) };
    if r < 0 {
        return Err(last_errno());
    }
    Ok(st)
}

#[cfg(test)]
mod tests {
    use std::{
        os::unix::fs::PermissionsExt,
        sync::atomic::{AtomicU32, Ordering},
    };

    use super::*;

    /// A unique scratch directory, removed on drop. Avoids a `tempfile`
    /// dev-dependency for the handful of tests here.
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

        let flags = OpenFlags(libc::O_CREAT | libc::O_RDWR);
        let file = fs.open(Path::new("hello.txt"), flags, Mode(0o644)).unwrap();
        assert_eq!(file.pwrite(b"hello world", 0).unwrap(), 11);

        let mut buf = [0u8; 11];
        assert_eq!(file.pread(&mut buf, 0).unwrap(), 11);
        assert_eq!(&buf, b"hello world");

        // Positional: a read at an offset does not depend on prior reads.
        let mut tail = [0u8; 5];
        assert_eq!(file.pread(&mut tail, 6).unwrap(), 5);
        assert_eq!(&tail, b"world");
    }

    #[test]
    fn append_updates_shared_offset_and_contents() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);

        let file = fs
            .open(
                Path::new("hello.txt"),
                OpenFlags(libc::O_CREAT | libc::O_RDWR | libc::O_APPEND),
                Mode(0o644),
            )
            .unwrap();
        assert_eq!(file.pwrite(b"hello", 0).unwrap(), 5);

        let write = file.append(b" world").unwrap();
        assert_eq!(write.count, 6);
        assert_eq!(write.offset, 11);

        let mut buf = [0u8; 11];
        assert_eq!(file.pread(&mut buf, 0).unwrap(), 11);
        assert_eq!(&buf, b"hello world");
    }

    #[test]
    fn stat_reports_real_identity_and_type() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);

        fs.open(
            Path::new("f"),
            OpenFlags(libc::O_CREAT | libc::O_WRONLY),
            Mode(0o600),
        )
        .unwrap();
        let st = fs.stat(Path::new("f"), true).unwrap();
        assert_eq!(st.file_type, FileType::Regular);
        assert_ne!(st.ino, 0);

        // The fd-side fstat agrees with the path-side stat on identity.
        let file = fs
            .open(Path::new("f"), OpenFlags(libc::O_RDONLY), Mode(0))
            .unwrap();
        assert_eq!(file.fstat().unwrap().ino, st.ino);
    }

    /// A `/`-rooted mount takes the single-fstatat fast path; it must agree
    /// with the host's own view on identity and honor `follow` on a final
    /// symlink.
    #[test]
    fn host_root_stat_fast_path() {
        use std::os::unix::fs::MetadataExt;

        let scratch = Scratch::new();
        let base = scratch.path.canonicalize().unwrap();
        std::fs::write(base.join("f"), b"x").unwrap();
        std::os::unix::fs::symlink("f", base.join("link")).unwrap();

        let fs = HostFs::new("/").unwrap();
        let st = fs.stat(&base.join("f"), true).unwrap();
        let meta = std::fs::metadata(base.join("f")).unwrap();
        assert_eq!(st.ino, meta.ino());
        assert_eq!(st.dev, meta.dev());

        let link = base.join("link");
        assert_eq!(fs.stat(&link, false).unwrap().file_type, FileType::Symlink);
        assert_eq!(fs.stat(&link, true).unwrap().file_type, FileType::Regular);
        assert_eq!(
            fs.stat(Path::new("/"), true).unwrap().file_type,
            FileType::Directory
        );
        assert_eq!(
            fs.stat(&base.join("nope"), true).unwrap_err(),
            Errno::ENOENT
        );
    }

    #[test]
    fn missing_path_is_enoent() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);
        assert_eq!(fs.stat(Path::new("nope"), true).unwrap_err(), Errno::ENOENT);
    }

    #[test]
    fn mkdir_then_getdents_lists_entries() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);

        fs.mkdir(Path::new("d"), Mode(0o755)).unwrap();
        assert_eq!(
            fs.stat(Path::new("d"), true).unwrap().file_type,
            FileType::Directory
        );

        fs.open(
            Path::new("d/a"),
            OpenFlags(libc::O_CREAT | libc::O_WRONLY),
            Mode(0o644),
        )
        .unwrap();
        fs.open(
            Path::new("d/b"),
            OpenFlags(libc::O_CREAT | libc::O_WRONLY),
            Mode(0o644),
        )
        .unwrap();

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

        // A second snapshot is just as complete (the handle was rewound).
        assert_eq!(dir.getdents().unwrap().len(), 2);
    }

    #[test]
    fn symlink_roundtrips_and_lstat_sees_link() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);

        fs.open(
            Path::new("target"),
            OpenFlags(libc::O_CREAT | libc::O_WRONLY),
            Mode(0o644),
        )
        .unwrap();
        fs.symlink(Path::new("target"), Path::new("link")).unwrap();

        assert_eq!(
            fs.readlink(Path::new("link")).unwrap(),
            PathBuf::from("target")
        );
        assert_eq!(
            fs.stat(Path::new("link"), false).unwrap().file_type,
            FileType::Symlink
        );
        assert_eq!(
            fs.stat(Path::new("link"), true).unwrap().file_type,
            FileType::Regular
        );
    }

    #[test]
    fn hardlink_shares_inode() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);

        fs.open(
            Path::new("orig"),
            OpenFlags(libc::O_CREAT | libc::O_WRONLY),
            Mode(0o644),
        )
        .unwrap();
        fs.link(Path::new("orig"), Path::new("alias")).unwrap();

        let a = fs.stat(Path::new("orig"), true).unwrap();
        let b = fs.stat(Path::new("alias"), true).unwrap();
        assert_eq!(a.ino, b.ino);
        assert_eq!(b.nlink, 2);
    }

    #[test]
    fn chmod_applies_permission_bits() {
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
        assert_eq!(file.fstat().unwrap().mode.0 & 0o7777, 0o640);
    }

    #[test]
    fn chmod_nofollow_refuses_symlink_without_changing_target() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);
        std::fs::write(scratch.path.join("target"), b"").unwrap();
        std::fs::set_permissions(
            scratch.path.join("target"),
            std::fs::Permissions::from_mode(0o640),
        )
        .unwrap();
        std::os::unix::fs::symlink("target", scratch.path.join("link")).unwrap();

        assert_eq!(
            fs.chmod(Path::new("link"), false, Mode(0o600)),
            Err(Errno(libc::EOPNOTSUPP))
        );
        assert_eq!(
            std::fs::metadata(scratch.path.join("target"))
                .unwrap()
                .permissions()
                .mode()
                & 0o7777,
            0o640
        );
    }

    #[test]
    fn rename_moves_entry() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);

        fs.open(
            Path::new("from"),
            OpenFlags(libc::O_CREAT | libc::O_WRONLY),
            Mode(0o644),
        )
        .unwrap();
        fs.rename(Path::new("from"), Path::new("to"), RenameFlags::EMPTY)
            .unwrap();

        assert_eq!(fs.stat(Path::new("from"), true).unwrap_err(), Errno::ENOENT);
        assert_eq!(
            fs.stat(Path::new("to"), true).unwrap().file_type,
            FileType::Regular
        );
    }

    #[test]
    fn fstatfs_survives_rename_and_unlink() {
        let scratch = Scratch::new();
        let fs = fs(&scratch);

        let file = fs
            .open(
                Path::new("mounted.txt"),
                OpenFlags(libc::O_CREAT | libc::O_RDWR),
                Mode(0o644),
            )
            .unwrap();
        let before = file.fstatfs().unwrap();

        fs.rename(
            Path::new("mounted.txt"),
            Path::new("renamed.txt"),
            RenameFlags::EMPTY,
        )
        .unwrap();
        fs.unlink(Path::new("renamed.txt")).unwrap();

        let after = file.fstatfs().unwrap();
        assert_eq!(after.fs_type, before.fs_type);
        assert_eq!(after.fsid, before.fsid);
        assert_eq!(after.flags, before.flags);
    }
}
