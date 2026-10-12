//! The Darwin decoder: arm64 BSD syscall numbers in, Darwin struct layouts
//! out. Every method body here is one of the parent's operations behind the
//! kernel's own spelling of it.
//!
//! Three things make the Darwin surface wider than the Linux one. libc
//! answers `getcwd` with `fcntl(F_GETPATH)` on an open `.`, so that command
//! must name the guest's path rather than the backing file's. The `fts`
//! walkers behind `ls`, `find`, `rm -r` and `cp -R` read directories with
//! `getattrlistbulk`, and `realpath` resolves with `getattrlist`, so both
//! are synthesized here from the symbolic [`Stat`] rather than forwarded —
//! a forwarded call would answer for the backing file in the delta, not the
//! merged entry the guest sees. And the attribute mutators that have no
//! Linux counterpart (`setattrlist`, which is how `utimensat` and `touch`
//! set times here, and `chflags`) are forwarded against the materialized
//! host path, so a layered filesystem copies the entry up first.

use std::{
    ffi::CString,
    os::unix::ffi::OsStrExt,
    path::{Component, Path, PathBuf},
    ptr,
};

use crate::{
    ForkHold, SyscallResult, SystemCall, SystemCalls, host_syscall,
    sys::{fs::vfs::FileType, fsops},
};

use super::{
    AtTarget, DirEntry, Errno, Personality, Stat, check_host, finish, guest_mut, marker_name,
    read_cstr, strip_marker_names,
};

// Darwin/arm64 BSD syscall numbers (`<sys/syscall.h>`).
const SYS_READ: u64 = 3;
const SYS_WRITE: u64 = 4;
const SYS_OPEN: u64 = 5;
const SYS_CLOSE: u64 = 6;
const SYS_LINK: u64 = 9;
const SYS_UNLINK: u64 = 10;
const SYS_CHDIR: u64 = 12;
const SYS_FCHDIR: u64 = 13;
const SYS_MKNOD: u64 = 14;
const SYS_CHMOD: u64 = 15;
const SYS_CHOWN: u64 = 16;
const SYS_ACCESS: u64 = 33;
const SYS_CHFLAGS: u64 = 34;
const SYS_FCHFLAGS: u64 = 35;
const SYS_DUP: u64 = 41;
const SYS_IOCTL: u64 = 54;
const SYS_SYMLINK: u64 = 57;
const SYS_READLINK: u64 = 58;
const SYS_DUP2: u64 = 90;
const SYS_FCNTL: u64 = 92;
const SYS_FSYNC: u64 = 95;
const SYS_READV: u64 = 120;
const SYS_WRITEV: u64 = 121;
const SYS_FCHOWN: u64 = 123;
const SYS_FCHMOD: u64 = 124;
const SYS_RENAME: u64 = 128;
const SYS_MKFIFO: u64 = 132;
const SYS_MKDIR: u64 = 136;
const SYS_RMDIR: u64 = 137;
const SYS_UTIMES: u64 = 138;
const SYS_FUTIMES: u64 = 139;
const SYS_PREAD: u64 = 153;
const SYS_PWRITE: u64 = 154;
const SYS_STATFS: u64 = 157;
const SYS_FDATASYNC: u64 = 187;
const SYS_STAT: u64 = 188;
const SYS_LSTAT: u64 = 190;
const SYS_PATHCONF: u64 = 191;
const SYS_LSEEK: u64 = 199;
const SYS_TRUNCATE: u64 = 200;
const SYS_FTRUNCATE: u64 = 201;
const SYS_OPEN_DPROTECTED_NP: u64 = 216;
const SYS_GETATTRLIST: u64 = 220;
const SYS_SETATTRLIST: u64 = 221;
const SYS_EXCHANGEDATA: u64 = 223;
const SYS_COPYFILE: u64 = 227;
const SYS_FGETATTRLIST: u64 = 228;
const SYS_FSETATTRLIST: u64 = 229;
const SYS_GETXATTR: u64 = 234;
const SYS_FGETXATTR: u64 = 235;
const SYS_SETXATTR: u64 = 236;
const SYS_FSETXATTR: u64 = 237;
const SYS_REMOVEXATTR: u64 = 238;
const SYS_FREMOVEXATTR: u64 = 239;
const SYS_LISTXATTR: u64 = 240;
const SYS_FLISTXATTR: u64 = 241;
const SYS_STAT64: u64 = 338;
const SYS_FSTAT64: u64 = 339;
const SYS_LSTAT64: u64 = 340;
const SYS_STAT64_EXTENDED: u64 = 341;
const SYS_LSTAT64_EXTENDED: u64 = 342;
const SYS_GETDIRENTRIES64: u64 = 344;
const SYS_STATFS64: u64 = 345;
const SYS_LCHOWN: u64 = 364;
const SYS_READ_NOCANCEL: u64 = 396;
const SYS_WRITE_NOCANCEL: u64 = 397;
const SYS_OPEN_NOCANCEL: u64 = 398;
const SYS_CLOSE_NOCANCEL: u64 = 399;
const SYS_FCNTL_NOCANCEL: u64 = 406;
const SYS_FSYNC_NOCANCEL: u64 = 408;
const SYS_READV_NOCANCEL: u64 = 411;
const SYS_WRITEV_NOCANCEL: u64 = 412;
const SYS_PREAD_NOCANCEL: u64 = 414;
const SYS_PWRITE_NOCANCEL: u64 = 415;
const SYS_GETATTRLISTBULK: u64 = 461;
const SYS_CLONEFILEAT: u64 = 462;
const SYS_OPENAT: u64 = 463;
const SYS_OPENAT_NOCANCEL: u64 = 464;
const SYS_RENAMEAT: u64 = 465;
const SYS_FACCESSAT: u64 = 466;
const SYS_FCHMODAT: u64 = 467;
const SYS_FCHOWNAT: u64 = 468;
const SYS_FSTATAT64: u64 = 470;
const SYS_LINKAT: u64 = 471;
const SYS_UNLINKAT: u64 = 472;
const SYS_READLINKAT: u64 = 473;
const SYS_SYMLINKAT: u64 = 474;
const SYS_MKDIRAT: u64 = 475;
const SYS_GETATTRLISTAT: u64 = 476;
const SYS_RENAMEATX_NP: u64 = 488;
const SYS_FCLONEFILEAT: u64 = 517;
const SYS_SETATTRLISTAT: u64 = 524;
const SYS_MKFIFOAT: u64 = 553;
const SYS_MKNODAT: u64 = 554;

/// `AT_SYMLINK_NOFOLLOW_ANY` (`<sys/fcntl.h>`): no symlink anywhere in the
/// path. The namespace walk has already resolved every component by the
/// time a flagged call reaches a filesystem, so the flag is accepted and
/// carries nothing further.
const AT_SYMLINK_NOFOLLOW_ANY: i32 = 0x0800;

/// `ioctl` requests that edit the descriptor's close-on-exec flag
/// (`<sys/filio.h>`), which lives in the table for a virtual descriptor.
const FIOCLEX: u64 = 0x2000_6601;
const FIONCLEX: u64 = 0x2000_6602;

/// The fd links the kernel resolves to the open file itself.
pub const FD_LINK_PREFIXES: &[&[u8]] = &[b"/dev/fd/"];

/// Darwin has no path that names the executed image; `proc_pidpath` and the
/// `apple[]` vector serve that, outside the filesystem.
pub fn is_magic_exe(_p: &Path) -> bool {
    false
}

/// `execve` always names its file by path here.
pub fn exec_names_fd(_path: &[u8], _flags: i32) -> bool {
    false
}

/// Apply `open(2)`'s rule to the raw mode before any filesystem sees it:
/// the third argument is variadic, so a non-creating open carries whatever
/// the caller left in the register, and a creating one masks everything
/// outside the permission bits.
pub fn sanitize_open(flags: i32, mode: u32) -> (i32, u32) {
    let mode = if flags & libc::O_CREAT != 0 {
        mode & 0o7777
    } else {
        0
    };
    (flags, mode)
}

/// Reopen the object behind a host descriptor through its fd link.
pub fn reopen_host_fd(fd: i32, flags: i32, mode: u32) -> Result<i64, Errno> {
    let path = format!("/dev/fd/{fd}\0");
    check_host(unsafe { libc::open(path.as_ptr() as *const libc::c_char, flags, mode) })
}

/// `timeval` microseconds as nanoseconds; `suseconds_t` is 32 bits here.
pub fn usec_to_nsec(usec: libc::suseconds_t) -> i64 {
    i64::from(usec) * 1000
}

/// The `AT_*` flags an attribute mutator accepts, decoded to whether the
/// final symlink is named itself. No Darwin call lets an empty path name the
/// dirfd.
pub fn decode_at_flags(flags: i32) -> Result<(bool, bool), Errno> {
    const ALLOWED: i32 = libc::AT_SYMLINK_NOFOLLOW | AT_SYMLINK_NOFOLLOW_ANY;
    if flags & !ALLOWED != 0 {
        return Err(Errno::EINVAL);
    }
    Ok((flags & libc::AT_SYMLINK_NOFOLLOW == 0, false))
}

impl SystemCalls for Personality {
    fn do_syscall(&self, call: &mut SystemCall) {
        let a = call.args;
        let fd = a[0] as i32;
        let result = match call.number {
            // --- open / create ---
            SYS_OPEN | SYS_OPEN_NOCANCEL => {
                self.open(libc::AT_FDCWD, a[0], a[1] as i32, a[2] as u32)
            }
            SYS_OPENAT | SYS_OPENAT_NOCANCEL => self.open(fd, a[1], a[2] as i32, a[3] as u32),
            // `(path, flags, class, dpflags, mode)`: the data-protection
            // class means nothing on a Mac; the rest is an open.
            SYS_OPEN_DPROTECTED_NP => self.open(libc::AT_FDCWD, a[0], a[1] as i32, a[4] as u32),

            // --- per-fd I/O (virtual only; host fds fall through) ---
            SYS_READ | SYS_READ_NOCANCEL if self.is_virtual(fd) => {
                self.read(fd, a[1], a[2] as usize)
            }
            SYS_WRITE | SYS_WRITE_NOCANCEL if self.is_virtual(fd) => {
                self.write(fd, a[1], a[2] as usize)
            }
            SYS_PREAD | SYS_PREAD_NOCANCEL if self.is_virtual(fd) => {
                self.pread(fd, a[1], a[2] as usize, a[3])
            }
            SYS_PWRITE | SYS_PWRITE_NOCANCEL if self.is_virtual(fd) => {
                self.pwrite(fd, a[1], a[2] as usize, a[3])
            }
            SYS_READV | SYS_READV_NOCANCEL if self.is_virtual(fd) => {
                self.readv(fd, a[1], a[2] as usize)
            }
            SYS_WRITEV | SYS_WRITEV_NOCANCEL if self.is_virtual(fd) => {
                self.writev(fd, a[1], a[2] as usize)
            }
            SYS_LSEEK if self.is_virtual(fd) => self.lseek(fd, a[1] as i64, a[2] as i32),
            SYS_GETDIRENTRIES64 if self.is_virtual(fd) => {
                let basep = a[3];
                self.read_dirents(fd, a[1], a[2] as usize, |entries, start, max| {
                    // `basep` receives the position the read started from.
                    if basep != 0 {
                        unsafe { ptr::write(basep as *mut i64, start as i64) };
                    }
                    let (bytes, next) = encode_dirents(entries, start, max);
                    let len = bytes.len() as i64;
                    Ok((bytes, next, len))
                })
            }
            SYS_GETATTRLISTBULK if self.is_virtual(fd) => {
                self.getattrlistbulk(fd, a[1], a[2], a[3] as usize, a[4] as u32)
            }
            SYS_FSTAT64 if self.is_virtual(fd) => {
                self.stat_fd(fd).map(|s| write_stat(a[1], &s)).map(|()| 0)
            }
            SYS_FSYNC | SYS_FSYNC_NOCANCEL | SYS_FDATASYNC if self.is_virtual(fd) => self.fsync(fd),
            SYS_FTRUNCATE if self.is_virtual(fd) => self.ftruncate(fd, a[1]),
            SYS_CLOSE | SYS_CLOSE_NOCANCEL if self.is_virtual(fd) => {
                self.fds.lock().unwrap().close(fd).map(|_| 0)
            }
            // The close-on-exec ioctls edit table state. Any other ioctl on
            // a special file must reach the real device — `TIOCGETA`
            // (isatty), `TIOCSWINSZ`, the pty controls — or no tty ever
            // works, so those forward to the host on the backing fd.
            // Regular files and directories keep answering ENOTTY: their
            // ioctls can mutate the filesystem itself, which would bypass
            // the read-only mount.
            SYS_IOCTL if self.is_virtual(fd) => match a[1] {
                FIOCLEX => self.set_cloexec(fd, true),
                FIONCLEX => self.set_cloexec(fd, false),
                _ => match self.desc(fd).and_then(|d| d.file.fstat()) {
                    Ok(st)
                        if matches!(
                            st.file_type,
                            FileType::CharDevice
                                | FileType::BlockDevice
                                | FileType::Fifo
                                | FileType::Socket
                        ) =>
                    {
                        self.deny_ro_or_passthrough(call, Ok(()), Some(0));
                        return;
                    }
                    Ok(_) => Err(Errno::ENOTTY),
                    Err(e) => Err(e),
                },
            },

            // --- dup / fcntl (virtual only) ---
            SYS_DUP if self.is_virtual(fd) => self.dup(fd, 0, false),
            SYS_DUP2 if self.is_virtual(fd) => self.dup2(fd, a[1] as i32, 0),
            // dup2 from a host fd forwards, but the kernel's dup2 atomically
            // replaces whatever occupies `newfd` — if that was a virtual
            // descriptor, its reservation is closed by the host call itself
            // and the table entry must go with it, or the table would keep
            // intercepting a number that is now a plain host fd.
            SYS_DUP2 if self.is_virtual(a[1] as i32) => {
                let mut fds = self.fds.lock().unwrap();
                let result = host_syscall(call);
                if let SyscallResult::Ok(_) = result {
                    fds.map.remove(&(a[1] as i32));
                }
                call.set_result(result);
                return;
            }
            SYS_FCNTL | SYS_FCNTL_NOCANCEL if self.is_virtual(fd) => {
                match self.fcntl(fd, a[1] as i32, a[2]) {
                    Some(result) => result,
                    // A command the table has no say in (`F_NOCACHE`,
                    // `F_FULLFSYNC`, `F_PREALLOCATE`, …) reaches the kernel
                    // through the reservation, an alias of the backing file.
                    None => {
                        call.set_result(host_syscall(call));
                        return;
                    }
                }
            }

            // --- path metadata ---
            SYS_STAT64 => self
                .stat_at(libc::AT_FDCWD, &unsafe { read_cstr(a[0]) }, true)
                .map(|s| write_stat(a[1], &s))
                .map(|()| 0),
            SYS_LSTAT64 => self
                .stat_at(libc::AT_FDCWD, &unsafe { read_cstr(a[0]) }, false)
                .map(|s| write_stat(a[1], &s))
                .map(|()| 0),
            SYS_FSTATAT64 => self
                .stat_at(
                    fd,
                    &unsafe { read_cstr(a[1]) },
                    a[3] as i32 & libc::AT_SYMLINK_NOFOLLOW == 0,
                )
                .map(|s| write_stat(a[2], &s))
                .map(|()| 0),
            // The pre-64-bit-inode layouts: nothing on arm64 issues them,
            // but a hand-rolled syscall still names a path the namespace
            // must resolve. The kernel lays out its own answer against the
            // backing file.
            // The `_extended` stats also fetch the ACL, which only the kernel
            // can lay out, so they resolve here and answer from the backing
            // file.
            SYS_STAT | SYS_LSTAT | SYS_STAT64_EXTENDED | SYS_LSTAT64_EXTENDED | SYS_STATFS
            | SYS_STATFS64 | SYS_PATHCONF => {
                let follow = !matches!(call.number, SYS_LSTAT | SYS_LSTAT64_EXTENDED);
                self.forward_at_host_path(call, 0, follow);
                return;
            }
            SYS_ACCESS => self.access(libc::AT_FDCWD, a[0], a[1] as i32),
            SYS_FACCESSAT => self.access(fd, a[1], a[2] as i32),
            SYS_READLINK => self.readlink(libc::AT_FDCWD, a[0], a[1], a[2] as usize),
            SYS_READLINKAT => self.readlink(fd, a[1], a[2], a[3] as usize),
            SYS_TRUNCATE => self.truncate(a[0], a[1]),
            SYS_GETATTRLIST => {
                self.getattrlist(
                    call,
                    libc::AT_FDCWD,
                    0,
                    a[1],
                    a[2],
                    a[3] as usize,
                    a[4] as u32,
                );
                return;
            }
            SYS_GETATTRLISTAT => {
                self.getattrlist(call, fd, 1, a[2], a[3], a[4] as usize, a[5] as u32);
                return;
            }
            SYS_FGETATTRLIST if self.is_virtual(fd) => {
                self.fgetattrlist(call, fd, a[1], a[2], a[3] as usize, a[4] as u32);
                return;
            }

            // --- path mutation ---
            SYS_MKDIR => self.mkdir(libc::AT_FDCWD, a[0], a[1] as u32),
            SYS_MKDIRAT => self.mkdir(fd, a[1], a[2] as u32),
            SYS_UNLINK => self.unlinkat(libc::AT_FDCWD, a[0], 0),
            SYS_RMDIR => self.unlinkat(libc::AT_FDCWD, a[0], libc::AT_REMOVEDIR),
            SYS_UNLINKAT => self.unlinkat(fd, a[1], a[2] as i32),
            SYS_RENAME => self.rename(libc::AT_FDCWD, a[0], libc::AT_FDCWD, a[1], 0),
            SYS_RENAMEAT => self.rename(fd, a[1], a[2] as i32, a[3], 0),
            SYS_RENAMEATX_NP => self.rename(fd, a[1], a[2] as i32, a[3], a[4] as u32),
            SYS_SYMLINK => self.symlink(a[0], libc::AT_FDCWD, a[1]),
            SYS_SYMLINKAT => self.symlink(a[0], a[1] as i32, a[2]),
            SYS_LINK => self.link(libc::AT_FDCWD, a[0], libc::AT_FDCWD, a[1]),
            SYS_LINKAT => self.link(fd, a[1], a[2] as i32, a[3]),
            // The in-kernel copies have no layered form: a caller falls back
            // to reading and writing, which the overlay serves.
            SYS_EXCHANGEDATA | SYS_CLONEFILEAT | SYS_FCLONEFILEAT | SYS_COPYFILE => {
                Err(Errno(libc::ENOTSUP))
            }

            // --- cwd ---
            SYS_CHDIR => self.chdir(a[0]),
            SYS_FCHDIR if self.is_virtual(fd) => self.fchdir(fd),

            // --- attribute mutators (path forms; fd forms virtual only) ---
            SYS_CHMOD => self.chmod_path(libc::AT_FDCWD, a[0], a[1] as u32, 0),
            SYS_FCHMODAT => self.chmod_path(fd, a[1], a[2] as u32, a[3] as i32),
            SYS_FCHMOD if self.is_virtual(fd) => self.fchmod(fd, a[1] as u32),
            SYS_CHOWN => self.chown_path(libc::AT_FDCWD, a[0], a[1] as u32, a[2] as u32, 0),
            SYS_LCHOWN => self.chown_path(
                libc::AT_FDCWD,
                a[0],
                a[1] as u32,
                a[2] as u32,
                libc::AT_SYMLINK_NOFOLLOW,
            ),
            SYS_FCHOWNAT => self.chown_path(fd, a[1], a[2] as u32, a[3] as u32, a[4] as i32),
            SYS_FCHOWN if self.is_virtual(fd) => self.fchown(fd, a[1] as u32, a[2] as u32),
            SYS_UTIMES => self.utimes(libc::AT_FDCWD, a[0], a[1]),
            SYS_FUTIMES if self.is_virtual(fd) => self.utimes(fd, 0, a[1]),
            // `setattrlist` is how libc sets times (`utimensat`, `touch`,
            // `cp -p`) and the rest of the mutable common attributes; the
            // kernel decodes the attribute buffer against the materialized
            // host path. The fd form becomes the path form on the
            // descriptor's own path.
            SYS_SETATTRLIST => {
                let follow = a[4] as u32 & libc::FSOPT_NOFOLLOW == 0;
                self.forward_materialized(call, libc::AT_FDCWD, 0, follow);
                return;
            }
            SYS_SETATTRLISTAT => {
                let follow = a[5] as u32 & libc::FSOPT_NOFOLLOW == 0;
                let mut path_form =
                    SystemCall::new(SYS_SETATTRLIST, [0, a[2], a[3], a[4], a[5], 0, 0, 0]);
                self.forward_materialized_from(&mut path_form, fd, a[1], follow);
                finish_from(call, &path_form);
                return;
            }
            SYS_FSETATTRLIST if self.is_virtual(fd) => {
                let mut path_form =
                    SystemCall::new(SYS_SETATTRLIST, [0, a[1], a[2], a[3], a[4], 0, 0, 0]);
                self.forward_materialized_from(&mut path_form, fd, 0, true);
                finish_from(call, &path_form);
                return;
            }
            SYS_CHFLAGS => {
                self.forward_materialized(call, libc::AT_FDCWD, 0, true);
                return;
            }
            SYS_FCHFLAGS if self.is_virtual(fd) => {
                let mut path_form = SystemCall::new(SYS_CHFLAGS, [0, a[1], 0, 0, 0, 0, 0, 0]);
                self.forward_materialized_from(&mut path_form, fd, 0, true);
                finish_from(call, &path_form);
                return;
            }
            SYS_MKNOD => self.mknod(libc::AT_FDCWD, a[0], a[1] as u32, a[2]),
            SYS_MKNODAT => self.mknod(fd, a[1], a[2] as u32, a[3]),
            SYS_MKFIFO => self.mknod(libc::AT_FDCWD, a[0], a[1] as u32 | libc::S_IFIFO as u32, 0),
            SYS_MKFIFOAT => self.mknod(fd, a[1], a[2] as u32 | libc::S_IFIFO as u32, 0),
            // `(path, name, value, size, position, options)`: the position
            // is for the resource fork and stays zero; `XATTR_NOFOLLOW`
            // decides the final symlink and the rest of the options
            // (`XATTR_CREATE`, `XATTR_REPLACE`) go through to the filesystem.
            SYS_SETXATTR => self.setxattr_path(
                a[0],
                a[1],
                a[2],
                a[3] as usize,
                a[5] as i32 & !libc::XATTR_NOFOLLOW,
                a[5] as i32 & libc::XATTR_NOFOLLOW == 0,
            ),
            SYS_FSETXATTR if self.is_virtual(fd) => {
                self.fsetxattr(fd, a[1], a[2], a[3] as usize, a[5] as i32)
            }
            SYS_REMOVEXATTR => {
                self.removexattr_path(a[0], a[1], a[2] as i32 & libc::XATTR_NOFOLLOW == 0)
            }
            SYS_FREMOVEXATTR if self.is_virtual(fd) => self.fremovexattr(fd, a[1]),
            // Xattr reads carry no write policy, but the path must still
            // resolve through the namespace: the file may live only in an
            // overlay's delta, and the host kernel then serves the backing
            // path. (A virtual fd is a kernel alias of its backing file, so
            // the fd forms forward as-is.) The overlay's own bookkeeping
            // namespace is invisible throughout: a marker name reads as
            // absent, and marker entries vanish from listings.
            SYS_GETXATTR => {
                if marker_name(a[1]) {
                    call.set_result(SyscallResult::Error(fsops::ENOATTR));
                } else {
                    self.forward_at_host_path(call, 0, a[5] as i32 & libc::XATTR_NOFOLLOW == 0);
                }
                return;
            }
            SYS_LISTXATTR => {
                self.forward_at_host_path(call, 0, a[3] as i32 & libc::XATTR_NOFOLLOW == 0);
                let result = call.result().expect("a forwarded call has a result");
                call.set_result(strip_marker_names(result, a[1], a[2] as usize));
                return;
            }
            SYS_FGETXATTR if marker_name(a[1]) => Err(Errno(fsops::ENOATTR)),
            SYS_FLISTXATTR => {
                let result = host_syscall(call);
                call.set_result(strip_marker_names(result, a[1], a[2] as usize));
                return;
            }

            // Everything else — including host-fd I/O and syscalls not modeled
            // here — goes to the host kernel unchanged.
            _ => {
                call.set_result(host_syscall(call));
                return;
            }
        };
        finish(call, result);
    }

    fn resolve_fd(&self, guest_fd: i32) -> Option<std::os::fd::RawFd> {
        self.host_fd_of(guest_fd)
    }

    fn on_execve(&self, path: &Path) {
        self.exec_committed(path)
    }

    fn lock_for_fork(&self) -> Box<dyn ForkHold + '_> {
        self.fork_hold()
    }

    fn resolve_exec(&self, dirfd: i32, path: &[u8], flags: i32) -> Option<Result<PathBuf, i32>> {
        Some(self.exec_target(dirfd, path, flags).map_err(|e| e.raw()))
    }
}

/// Copy a forwarded stand-in call's result onto the guest's call.
fn finish_from(call: &mut SystemCall, forwarded: &SystemCall) {
    call.set_result(forwarded.result().expect("a forwarded call has a result"));
}

impl Personality {
    /// The `fcntl` commands the table answers for; `None` for one the kernel
    /// should see through the reservation.
    fn fcntl(&self, fd: i32, cmd: i32, arg: u64) -> Option<Result<i64, Errno>> {
        Some(match cmd {
            libc::F_DUPFD => self.dup(fd, arg as i32, false),
            libc::F_DUPFD_CLOEXEC => self.dup(fd, arg as i32, true),
            libc::F_GETFD => self
                .cloexec(fd)
                .map(|c| if c { libc::FD_CLOEXEC as i64 } else { 0 }),
            libc::F_SETFD => self.set_cloexec(fd, arg as i32 & libc::FD_CLOEXEC != 0),
            libc::F_GETFL => self.status_flags(fd),
            libc::F_SETFL => {
                self.set_status_flags(fd, arg as i32, libc::O_APPEND | libc::O_NONBLOCK)
            }
            // libc's `getcwd` and `realpath` ask the kernel for the path
            // behind an open `.`; the answer must be the guest's path, not
            // the backing file's — which for an upper entry would name the
            // delta.
            libc::F_GETPATH | libc::F_GETPATH_NOFIRMLINK => self.getpath(fd, arg),
            libc::F_GETLK
            | libc::F_SETLK
            | libc::F_SETLKW
            | libc::F_OFD_GETLK
            | libc::F_OFD_SETLK
            | libc::F_OFD_SETLKW => self.lock(fd, cmd, arg),
            _ => return None,
        })
    }

    /// `F_GETPATH`: the path the descriptor was opened by, into the caller's
    /// `MAXPATHLEN` buffer.
    fn getpath(&self, fd: i32, buf: u64) -> Result<i64, Errno> {
        let path = self.desc(fd)?.path.clone();
        let bytes = path.as_os_str().as_bytes();
        if bytes.len() + 1 > libc::PATH_MAX as usize {
            return Err(Errno(libc::ENAMETOOLONG));
        }
        let out = guest_mut(buf, bytes.len() + 1);
        out[..bytes.len()].copy_from_slice(bytes);
        out[bytes.len()] = 0;
        Ok(0)
    }

    /// Forward a path-keyed mutator the Personality does not model to the
    /// host, against the host path a mutation of the guest path may be
    /// applied to (see [`crate::Vfs::materialize`]): the read-only check
    /// applies as for any write, and a layered filesystem copies the entry
    /// up first so the change lands on the copy the guest sees. `dirfd`
    /// anchors a relative path at `call.args[index]`.
    fn forward_materialized(&self, call: &mut SystemCall, dirfd: i32, index: usize, follow: bool) {
        let pathptr = call.args[index];
        self.forward_materialized_from(call, dirfd, pathptr, follow);
    }

    /// [`Self::forward_materialized`] with the guest path read from `pathptr`
    /// rather than from the call: for the fd forms recast as path forms, where
    /// a zero `pathptr` names the descriptor's own file.
    fn forward_materialized_from(
        &self,
        call: &mut SystemCall,
        dirfd: i32,
        pathptr: u64,
        follow: bool,
    ) {
        let host = (|| -> Result<CString, Errno> {
            let abs: PathBuf = if pathptr == 0 {
                self.writable_desc(dirfd)?.path.clone()
            } else {
                let raw = unsafe { read_cstr(pathptr) };
                self.abs_path(dirfd, &raw)?.into_owned()
            };
            let r = self.ns.resolve(&abs, follow)?;
            if !r.writable {
                return Err(Errno::EROFS);
            }
            let host = r.fs.materialize(&r.rel)?;
            CString::new(host.as_os_str().as_bytes()).map_err(|_| Errno::EINVAL)
        })();
        self.forward_rewritten(call, 0, host);
    }

    /// `getattrlist`/`getattrlistat`: answer from the symbolic stat when every
    /// requested attribute can be, so the guest sees the same identity `stat`
    /// reports; otherwise let the kernel answer for the backing host path.
    #[allow(clippy::too_many_arguments)]
    fn getattrlist(
        &self,
        call: &mut SystemCall,
        dirfd: i32,
        index: usize,
        alist: u64,
        buf: u64,
        size: usize,
        options: u32,
    ) {
        let follow = options & libc::FSOPT_NOFOLLOW == 0;
        let req = match AttrRequest::read(alist) {
            Ok(req) => req,
            Err(e) => return call.set_result(SyscallResult::Error(e.raw())),
        };
        if !req.synthesizable() {
            if index == 0 {
                return self.forward_at_host_path(call, 0, follow);
            }
            // The `at` form has no host-path counterpart the kernel
            // accepts with a dirfd of ours; recast it against the absolute
            // host path.
            let mut path_form = SystemCall::new(
                SYS_GETATTRLIST,
                [0, alist, buf, size as u64, options as u64, 0, 0, 0],
            );
            let host = (|| -> Result<CString, Errno> {
                let raw = unsafe { read_cstr(call.args[index]) };
                let abs = self.abs_path(dirfd, &raw)?;
                let r = self.ns.resolve(&abs, follow)?;
                let host = r.fs.host_path(&r.rel).ok_or(Errno::ENOENT)?;
                CString::new(host.as_os_str().as_bytes()).map_err(|_| Errno::EINVAL)
            })();
            self.forward_rewritten(&mut path_form, 0, host);
            return finish_from(call, &path_form);
        }
        let result = (|| -> Result<i64, Errno> {
            let raw = unsafe { read_cstr(call.args[index]) };
            let abs = self.abs_path(dirfd, &raw)?;
            let r = self.ns.resolve(&abs, follow)?;
            let st = match r.stat {
                Some(st) => st,
                None => r.fs.stat(&r.rel, follow)?,
            };
            let subject = self.subject(&r.abs, &st)?;
            let record = pack_record(&req, &subject, options, 4);
            write_record(buf, size, &record, options)
        })();
        finish(call, result);
    }

    /// `fgetattrlist` on a virtual descriptor: as [`Self::getattrlist`], for
    /// the open file.
    fn fgetattrlist(
        &self,
        call: &mut SystemCall,
        fd: i32,
        alist: u64,
        buf: u64,
        size: usize,
        options: u32,
    ) {
        let req = match AttrRequest::read(alist) {
            Ok(req) => req,
            Err(e) => return call.set_result(SyscallResult::Error(e.raw())),
        };
        if !req.synthesizable() {
            return call.set_result(host_syscall(call));
        }
        let result = (|| -> Result<i64, Errno> {
            let desc = self.desc(fd)?;
            let st = desc.file.fstat()?;
            let subject = self.subject(&desc.path, &st)?;
            let record = pack_record(&req, &subject, options, 4);
            write_record(buf, size, &record, options)
        })();
        finish(call, result);
    }

    /// `getattrlistbulk` on a virtual directory: the merged entries from the
    /// directory snapshot, each packed as [`pack_record`] lays it out. The
    /// snapshot is the one `getdirentries64` reads, cursor included; the
    /// synthesized `.` and `..` are skipped, since bulk reads never report
    /// them. An entry that cannot be stat'd (removed between the snapshot
    /// and now) is skipped the way the kernel skips a vanished entry.
    fn getattrlistbulk(
        &self,
        fd: i32,
        alist: u64,
        buf: u64,
        size: usize,
        options: u32,
    ) -> Result<i64, Errno> {
        let req = AttrRequest::read(alist)?;
        if !req.synthesizable() {
            return Err(Errno::EINVAL);
        }
        let dir = self.desc(fd)?.path.clone();
        let dir_stat = self.desc(fd)?.file.fstat()?;
        self.read_dirents(fd, buf, size, |entries, start, max| {
            let mut out = Vec::new();
            let mut count = 0i64;
            let mut idx = start;
            while idx < entries.len() {
                let e = &entries[idx];
                let name = e.name.as_bytes();
                if name == b"." || name == b".." {
                    idx += 1;
                    continue;
                }
                let abs = dir.join(&e.name);
                let subject = if req.needs_stat() {
                    match self.ns.stat_path(&abs, false) {
                        Ok(st) => AttrSubject {
                            st,
                            name: name.to_vec(),
                            full_path: abs.as_os_str().as_bytes().to_vec(),
                            parent_ino: dir_stat.ino,
                        },
                        Err(_) => {
                            idx += 1;
                            continue;
                        }
                    }
                } else {
                    // Name, type, and inode come straight from the entry.
                    let mut st = dir_stat;
                    st.ino = e.ino;
                    st.file_type = e.file_type;
                    AttrSubject {
                        st,
                        name: name.to_vec(),
                        full_path: abs.as_os_str().as_bytes().to_vec(),
                        parent_ino: dir_stat.ino,
                    }
                };
                let record = pack_record(&req, &subject, options, 8);
                if out.len() + record.len() > max {
                    if count == 0 {
                        return Err(Errno::ERANGE);
                    }
                    break;
                }
                out.extend_from_slice(&record);
                count += 1;
                idx += 1;
            }
            Ok((out, idx, count))
        })
    }

    /// What an attribute request is answered about: the stat, the entry's
    /// name, its absolute guest path, and its parent's inode.
    fn subject(&self, abs: &Path, st: &Stat) -> Result<AttrSubject, Errno> {
        let name = match abs.components().next_back() {
            Some(Component::Normal(n)) => n.as_bytes().to_vec(),
            _ => Vec::new(),
        };
        let parent_ino = match abs.parent() {
            Some(parent) if parent != abs => {
                self.ns.stat_path(parent, true).map(|p| p.ino).unwrap_or(0)
            }
            _ => st.ino,
        };
        Ok(AttrSubject {
            st: *st,
            name,
            full_path: abs.as_os_str().as_bytes().to_vec(),
            parent_ino,
        })
    }
}

// --- struct stat ---

/// Fill a guest Darwin `struct stat` (the 64-bit-inode layout every arm64
/// binary uses) from a symbolic [`Stat`].
fn write_stat(ptr: u64, s: &Stat) {
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    st.st_dev = s.dev as libc::dev_t;
    st.st_mode = s.mode.0 as libc::mode_t;
    st.st_nlink = s.nlink as libc::nlink_t;
    st.st_ino = s.ino;
    st.st_uid = s.uid;
    st.st_gid = s.gid;
    st.st_rdev = s.rdev as libc::dev_t;
    st.st_atime = s.atime.sec;
    st.st_atime_nsec = s.atime.nsec;
    st.st_mtime = s.mtime.sec;
    st.st_mtime_nsec = s.mtime.nsec;
    st.st_ctime = s.ctime.sec;
    st.st_ctime_nsec = s.ctime.nsec;
    st.st_birthtime = s.btime.sec;
    st.st_birthtime_nsec = s.btime.nsec;
    st.st_size = s.size as libc::off_t;
    st.st_blocks = s.blocks as libc::blkcnt_t;
    st.st_blksize = s.blksize as libc::blksize_t;
    st.st_flags = s.flags;
    unsafe { ptr::write(ptr as *mut libc::stat, st) };
}

// --- getdirentries64 ---

/// Encode directory `entries` from index `start` into Darwin's 64-bit
/// `struct dirent` records (`d_ino`, `d_seekoff`, `d_reclen`, `d_namlen`,
/// `d_type`, name), stopping before the byte total would exceed `max`.
/// Returns the encoded bytes and the index to resume from. `d_seekoff` is
/// the position after the entry, which is what `telldir` reports.
fn encode_dirents(entries: &[DirEntry], start: usize, max: usize) -> (Vec<u8>, usize) {
    const HDR: usize = 21;
    let mut out = Vec::new();
    let mut idx = start;
    while idx < entries.len() {
        let e = &entries[idx];
        let name = e.name.as_bytes();
        let reclen = (HDR + name.len() + 1).div_ceil(8) * 8;
        if out.len() + reclen > max {
            break;
        }
        let mut rec = vec![0u8; reclen];
        rec[0..8].copy_from_slice(&e.ino.to_ne_bytes());
        rec[8..16].copy_from_slice(&((idx as u64) + 1).to_ne_bytes());
        rec[16..18].copy_from_slice(&(reclen as u16).to_ne_bytes());
        rec[18..20].copy_from_slice(&(name.len() as u16).to_ne_bytes());
        rec[20] = dirent_type(e.file_type);
        rec[21..21 + name.len()].copy_from_slice(name);
        out.extend_from_slice(&rec);
        idx += 1;
    }
    (out, idx)
}

fn dirent_type(t: FileType) -> u8 {
    match t {
        FileType::Regular => libc::DT_REG,
        FileType::Directory => libc::DT_DIR,
        FileType::Symlink => libc::DT_LNK,
        FileType::CharDevice => libc::DT_CHR,
        FileType::BlockDevice => libc::DT_BLK,
        FileType::Fifo => libc::DT_FIFO,
        FileType::Socket => libc::DT_SOCK,
    }
}

// --- getattrlist ---
//
// A request is five bitmaps (`struct attrlist`). The answer packs the
// requested attributes in a fixed order — the bit order within each group,
// common, then directory or file by the object's type — each fixed-size
// field at 4-byte alignment, with the variable-length ones (names, paths)
// placed as `attrreference_t {offset, length}` whose data follows the fixed
// fields. `ATTR_CMN_RETURNED_ATTRS` comes first and says which of the
// requested attributes were actually packed. Every record starts with its
// own length.

const ATTR_BIT_MAP_COUNT: u16 = 5;

const ATTR_CMN_NAME: u32 = 0x0000_0001;
const ATTR_CMN_DEVID: u32 = 0x0000_0002;
const ATTR_CMN_OBJTYPE: u32 = 0x0000_0008;
const ATTR_CMN_OBJID: u32 = 0x0000_0020;
const ATTR_CMN_OBJPERMANENTID: u32 = 0x0000_0040;
const ATTR_CMN_PAROBJID: u32 = 0x0000_0080;
const ATTR_CMN_CRTIME: u32 = 0x0000_0200;
const ATTR_CMN_MODTIME: u32 = 0x0000_0400;
const ATTR_CMN_CHGTIME: u32 = 0x0000_0800;
const ATTR_CMN_ACCTIME: u32 = 0x0000_1000;
const ATTR_CMN_BKUPTIME: u32 = 0x0000_2000;
const ATTR_CMN_OWNERID: u32 = 0x0000_8000;
const ATTR_CMN_GRPID: u32 = 0x0001_0000;
const ATTR_CMN_ACCESSMASK: u32 = 0x0002_0000;
const ATTR_CMN_FLAGS: u32 = 0x0004_0000;
const ATTR_CMN_GEN_COUNT: u32 = 0x0008_0000;
const ATTR_CMN_DOCUMENT_ID: u32 = 0x0010_0000;
const ATTR_CMN_USERACCESS: u32 = 0x0020_0000;
const ATTR_CMN_FILEID: u32 = 0x0200_0000;
const ATTR_CMN_PARENTID: u32 = 0x0400_0000;
const ATTR_CMN_FULLPATH: u32 = 0x0800_0000;
const ATTR_CMN_ADDEDTIME: u32 = 0x1000_0000;
const ATTR_CMN_ERROR: u32 = 0x2000_0000;
const ATTR_CMN_DATA_PROTECT_FLAGS: u32 = 0x4000_0000;
const ATTR_CMN_RETURNED_ATTRS: u32 = 0x8000_0000;

const ATTR_DIR_LINKCOUNT: u32 = 0x0000_0001;
const ATTR_DIR_MOUNTSTATUS: u32 = 0x0000_0004;
const ATTR_DIR_ALLOCSIZE: u32 = 0x0000_0008;
const ATTR_DIR_IOBLOCKSIZE: u32 = 0x0000_0010;
const ATTR_DIR_DATALENGTH: u32 = 0x0000_0020;

const ATTR_FILE_LINKCOUNT: u32 = 0x0000_0001;
const ATTR_FILE_TOTALSIZE: u32 = 0x0000_0002;
const ATTR_FILE_ALLOCSIZE: u32 = 0x0000_0004;
const ATTR_FILE_IOBLOCKSIZE: u32 = 0x0000_0008;
const ATTR_FILE_CLUMPSIZE: u32 = 0x0000_0010;
const ATTR_FILE_DEVTYPE: u32 = 0x0000_0020;
const ATTR_FILE_FILETYPE: u32 = 0x0000_0040;
const ATTR_FILE_FORKCOUNT: u32 = 0x0000_0080;
const ATTR_FILE_DATALENGTH: u32 = 0x0000_0200;
const ATTR_FILE_DATAALLOCSIZE: u32 = 0x0000_0400;
const ATTR_FILE_RSRCLENGTH: u32 = 0x0000_1000;
const ATTR_FILE_RSRCALLOCSIZE: u32 = 0x0000_2000;

/// The attributes a [`Stat`] answers. Anything else — volume attributes,
/// Finder info, ACLs, UUIDs — the kernel answers for the backing file.
const SYNTH_CMN: u32 = ATTR_CMN_RETURNED_ATTRS
    | ATTR_CMN_ERROR
    | ATTR_CMN_NAME
    | ATTR_CMN_DEVID
    | ATTR_CMN_OBJTYPE
    | ATTR_CMN_OBJID
    | ATTR_CMN_OBJPERMANENTID
    | ATTR_CMN_PAROBJID
    | ATTR_CMN_CRTIME
    | ATTR_CMN_MODTIME
    | ATTR_CMN_CHGTIME
    | ATTR_CMN_ACCTIME
    | ATTR_CMN_BKUPTIME
    | ATTR_CMN_OWNERID
    | ATTR_CMN_GRPID
    | ATTR_CMN_ACCESSMASK
    | ATTR_CMN_FLAGS
    | ATTR_CMN_GEN_COUNT
    | ATTR_CMN_DOCUMENT_ID
    | ATTR_CMN_USERACCESS
    | ATTR_CMN_FILEID
    | ATTR_CMN_PARENTID
    | ATTR_CMN_FULLPATH
    | ATTR_CMN_ADDEDTIME
    | ATTR_CMN_DATA_PROTECT_FLAGS;
const SYNTH_DIR: u32 = ATTR_DIR_LINKCOUNT
    | ATTR_DIR_MOUNTSTATUS
    | ATTR_DIR_ALLOCSIZE
    | ATTR_DIR_IOBLOCKSIZE
    | ATTR_DIR_DATALENGTH;
const SYNTH_FILE: u32 = ATTR_FILE_LINKCOUNT
    | ATTR_FILE_TOTALSIZE
    | ATTR_FILE_ALLOCSIZE
    | ATTR_FILE_IOBLOCKSIZE
    | ATTR_FILE_CLUMPSIZE
    | ATTR_FILE_DEVTYPE
    | ATTR_FILE_FILETYPE
    | ATTR_FILE_FORKCOUNT
    | ATTR_FILE_DATALENGTH
    | ATTR_FILE_DATAALLOCSIZE
    | ATTR_FILE_RSRCLENGTH
    | ATTR_FILE_RSRCALLOCSIZE;

/// The attributes a directory entry answers without a stat: what the
/// snapshot already carries.
const ENTRY_CMN: u32 = ATTR_CMN_RETURNED_ATTRS
    | ATTR_CMN_ERROR
    | ATTR_CMN_NAME
    | ATTR_CMN_OBJTYPE
    | ATTR_CMN_OBJID
    | ATTR_CMN_OBJPERMANENTID
    | ATTR_CMN_FILEID
    | ATTR_CMN_FULLPATH
    | ATTR_CMN_PAROBJID
    | ATTR_CMN_PARENTID;

/// A guest `struct attrlist`.
struct AttrRequest {
    cmn: u32,
    vol: u32,
    dir: u32,
    file: u32,
    fork: u32,
}

impl AttrRequest {
    fn read(ptr: u64) -> Result<AttrRequest, Errno> {
        if ptr == 0 {
            return Err(Errno(libc::EFAULT));
        }
        let words = unsafe { ptr::read_unaligned(ptr as *const [u32; 6]) };
        if words[0] as u16 != ATTR_BIT_MAP_COUNT {
            return Err(Errno::EINVAL);
        }
        Ok(AttrRequest {
            cmn: words[1],
            vol: words[2],
            dir: words[3],
            file: words[4],
            fork: words[5],
        })
    }

    fn synthesizable(&self) -> bool {
        self.vol == 0
            && self.fork == 0
            && self.cmn & !SYNTH_CMN == 0
            && self.dir & !SYNTH_DIR == 0
            && self.file & !SYNTH_FILE == 0
    }

    fn needs_stat(&self) -> bool {
        self.cmn & !ENTRY_CMN != 0 || self.dir != 0 || self.file != 0
    }
}

struct AttrSubject {
    st: Stat,
    name: Vec<u8>,
    full_path: Vec<u8>,
    parent_ino: u64,
}

/// `fsobj_type_t` for a file type (`<sys/vnode.h>` `enum vtype`).
fn vtype(t: FileType) -> u32 {
    match t {
        FileType::Regular => 1,
        FileType::Directory => 2,
        FileType::BlockDevice => 3,
        FileType::CharDevice => 4,
        FileType::Symlink => 5,
        FileType::Socket => 6,
        FileType::Fifo => 7,
    }
}

/// The caller's own access rights to the object (`ATTR_CMN_USERACCESS`):
/// the permission class the effective ids fall in, as `X_OK|W_OK|R_OK` bits.
fn user_access(st: &Stat) -> u32 {
    let (uid, gid) = unsafe { (libc::geteuid(), libc::getegid()) };
    if uid == 0 {
        return 7;
    }
    let shift = if uid == st.uid {
        6
    } else if gid == st.gid {
        3
    } else {
        0
    };
    (st.mode.0 >> shift) & 7
}

/// One packed record: the length word, the fixed fields, then the variable
/// data, the whole padded to `align`.
fn pack_record(req: &AttrRequest, s: &AttrSubject, options: u32, align: usize) -> Vec<u8> {
    let st = &s.st;
    let is_dir = st.file_type == FileType::Directory;
    let mut fixed: Vec<u8> = Vec::with_capacity(160);
    let mut strings: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut returned = [0u32; 5];
    let pack_inval = options & libc::FSOPT_PACK_INVAL_ATTRS != 0;

    // The returned-attributes set is filled in once the record is packed.
    if req.cmn & ATTR_CMN_RETURNED_ATTRS != 0 {
        fixed.extend_from_slice(&[0u8; 20]);
        returned[0] |= ATTR_CMN_RETURNED_ATTRS;
    }
    let ts = |t: super::Timespec| {
        let mut v = Vec::with_capacity(16);
        v.extend_from_slice(&t.sec.to_ne_bytes());
        v.extend_from_slice(&t.nsec.to_ne_bytes());
        v
    };
    let objid = |ino: u64| {
        let mut v = Vec::with_capacity(8);
        v.extend_from_slice(&(ino as u32).to_ne_bytes());
        v.extend_from_slice(&0u32.to_ne_bytes());
        v
    };
    let cmn: [(u32, Vec<u8>); 24] = [
        (ATTR_CMN_ERROR, 0u32.to_ne_bytes().to_vec()),
        (ATTR_CMN_NAME, Vec::new()),
        (ATTR_CMN_DEVID, (st.dev as u32).to_ne_bytes().to_vec()),
        (ATTR_CMN_OBJTYPE, vtype(st.file_type).to_ne_bytes().to_vec()),
        (ATTR_CMN_OBJID, objid(st.ino)),
        (ATTR_CMN_OBJPERMANENTID, objid(st.ino)),
        (ATTR_CMN_PAROBJID, objid(s.parent_ino)),
        (ATTR_CMN_CRTIME, ts(st.btime)),
        (ATTR_CMN_MODTIME, ts(st.mtime)),
        (ATTR_CMN_CHGTIME, ts(st.ctime)),
        (ATTR_CMN_ACCTIME, ts(st.atime)),
        (ATTR_CMN_BKUPTIME, vec![0u8; 16]),
        (ATTR_CMN_OWNERID, st.uid.to_ne_bytes().to_vec()),
        (ATTR_CMN_GRPID, st.gid.to_ne_bytes().to_vec()),
        (ATTR_CMN_ACCESSMASK, st.mode.0.to_ne_bytes().to_vec()),
        (ATTR_CMN_FLAGS, st.flags.to_ne_bytes().to_vec()),
        (ATTR_CMN_GEN_COUNT, 0u32.to_ne_bytes().to_vec()),
        (ATTR_CMN_DOCUMENT_ID, 0u32.to_ne_bytes().to_vec()),
        (ATTR_CMN_USERACCESS, user_access(st).to_ne_bytes().to_vec()),
        (ATTR_CMN_FILEID, st.ino.to_ne_bytes().to_vec()),
        (ATTR_CMN_PARENTID, s.parent_ino.to_ne_bytes().to_vec()),
        (ATTR_CMN_FULLPATH, Vec::new()),
        (ATTR_CMN_ADDEDTIME, ts(st.btime)),
        (ATTR_CMN_DATA_PROTECT_FLAGS, 0u32.to_ne_bytes().to_vec()),
    ];
    for (bit, value) in cmn {
        if req.cmn & bit == 0 {
            continue;
        }
        returned[0] |= bit;
        match bit {
            ATTR_CMN_NAME => {
                strings.push((fixed.len(), s.name.clone()));
                fixed.extend_from_slice(&[0u8; 8]);
            }
            ATTR_CMN_FULLPATH => {
                strings.push((fixed.len(), s.full_path.clone()));
                fixed.extend_from_slice(&[0u8; 8]);
            }
            _ => fixed.extend_from_slice(&value),
        }
    }
    let alloc = st.blocks * 512;
    if is_dir {
        let dir: [(u32, Vec<u8>); 5] = [
            (ATTR_DIR_LINKCOUNT, (st.nlink as u32).to_ne_bytes().to_vec()),
            (ATTR_DIR_MOUNTSTATUS, 0u32.to_ne_bytes().to_vec()),
            (ATTR_DIR_ALLOCSIZE, alloc.to_ne_bytes().to_vec()),
            (
                ATTR_DIR_IOBLOCKSIZE,
                (st.blksize as u32).to_ne_bytes().to_vec(),
            ),
            (ATTR_DIR_DATALENGTH, st.size.to_ne_bytes().to_vec()),
        ];
        for (bit, value) in dir {
            if req.dir & bit != 0 {
                returned[2] |= bit;
                fixed.extend_from_slice(&value);
            }
        }
        // File attributes do not apply to a directory and are left out of
        // the packing, whatever `FSOPT_PACK_INVAL_ATTRS` says.
    } else {
        let file: [(u32, Vec<u8>); 12] = [
            (
                ATTR_FILE_LINKCOUNT,
                (st.nlink as u32).to_ne_bytes().to_vec(),
            ),
            (ATTR_FILE_TOTALSIZE, st.size.to_ne_bytes().to_vec()),
            (ATTR_FILE_ALLOCSIZE, alloc.to_ne_bytes().to_vec()),
            (
                ATTR_FILE_IOBLOCKSIZE,
                (st.blksize as u32).to_ne_bytes().to_vec(),
            ),
            (
                ATTR_FILE_CLUMPSIZE,
                (st.blksize as u32).to_ne_bytes().to_vec(),
            ),
            (ATTR_FILE_DEVTYPE, (st.rdev as u32).to_ne_bytes().to_vec()),
            (ATTR_FILE_FILETYPE, 0u32.to_ne_bytes().to_vec()),
            (ATTR_FILE_FORKCOUNT, 1u32.to_ne_bytes().to_vec()),
            (ATTR_FILE_DATALENGTH, st.size.to_ne_bytes().to_vec()),
            (ATTR_FILE_DATAALLOCSIZE, alloc.to_ne_bytes().to_vec()),
            (ATTR_FILE_RSRCLENGTH, 0u64.to_ne_bytes().to_vec()),
            (ATTR_FILE_RSRCALLOCSIZE, 0u64.to_ne_bytes().to_vec()),
        ];
        for (bit, value) in file {
            if req.file & bit != 0 {
                returned[3] |= bit;
                fixed.extend_from_slice(&value);
            }
        }
        if pack_inval {
            // Directory attributes do not apply to a file; a caller that
            // asked for the invalid ones packed gets zeros in their place.
            for (bit, width) in [
                (ATTR_DIR_LINKCOUNT, 4),
                (ATTR_DIR_MOUNTSTATUS, 4),
                (ATTR_DIR_ALLOCSIZE, 8),
                (ATTR_DIR_IOBLOCKSIZE, 4),
                (ATTR_DIR_DATALENGTH, 8),
            ] {
                if req.dir & bit != 0 {
                    fixed.extend_from_slice(&vec![0u8; width]);
                }
            }
        }
    }

    const HEADER: usize = 4;
    let mut record = Vec::with_capacity(HEADER + fixed.len() + 64);
    record.extend_from_slice(&[0u8; HEADER]);
    record.extend_from_slice(&fixed);
    for (ref_pos, data) in strings {
        let field = HEADER + ref_pos;
        let offset = (record.len() - field) as i32;
        let length = (data.len() + 1) as u32;
        record[field..field + 4].copy_from_slice(&offset.to_ne_bytes());
        record[field + 4..field + 8].copy_from_slice(&length.to_ne_bytes());
        record.extend_from_slice(&data);
        record.push(0);
        while record.len() % 4 != 0 {
            record.push(0);
        }
    }
    while record.len() % align != 0 {
        record.push(0);
    }
    let len = record.len() as u32;
    record[..4].copy_from_slice(&len.to_ne_bytes());
    if req.cmn & ATTR_CMN_RETURNED_ATTRS != 0 {
        for (i, word) in returned.iter().enumerate() {
            record[HEADER + i * 4..HEADER + i * 4 + 4].copy_from_slice(&word.to_ne_bytes());
        }
    }
    record
}

/// Deliver one `getattrlist` record into the caller's buffer: as much as
/// fits, with the length word reporting the whole record's size when the
/// caller asked (`FSOPT_REPORT_FULLSIZE`) or when it all fit.
fn write_record(buf: u64, size: usize, record: &[u8], options: u32) -> Result<i64, Errno> {
    if size < 4 {
        return Err(Errno::ERANGE);
    }
    let n = record.len().min(size);
    let out = guest_mut(buf, n);
    out.copy_from_slice(&record[..n]);
    if n < record.len() && options & libc::FSOPT_REPORT_FULLSIZE == 0 {
        out[..4].copy_from_slice(&(n as u32).to_ne_bytes());
    }
    Ok(0)
}

/// Decoded by the parent for the `AT_*` mutators; Darwin's forms never name
/// the dirfd with an empty path, so the fd arm is unreachable here.
#[allow(dead_code)]
fn _at_target_shape(_t: AtTarget) {}
