//! The Linux decoder: x86-64 syscall numbers in, Linux struct layouts out.
//! Every method body here is one of the parent's operations behind the
//! kernel's own spelling of it.

use std::{
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    ptr,
};

use crate::{
    ForkHold, SyscallResult, SystemCall, SystemCalls, host_syscall, sys::hostfs::BACKING_FLOOR,
    sys::mmap::copy_from_guest,
};

use super::{
    AtTarget, DirEntry, Errno, FileType, Personality, Stat, Timespec, check_host, finish,
    guest_mut, marker_name, read_cstr, strip_marker_names,
};
use crate::sys::fs::vfs::StatFs;

/// The fd links the kernel resolves to the open file itself.
pub const FD_LINK_PREFIXES: &[&[u8]] = &[b"/dev/fd/", b"/proc/self/fd/", b"/proc/thread-self/fd/"];

/// `true` for a path the kernel resolves to the calling process's executed
/// image rather than through the filesystem.
pub fn is_magic_exe(p: &Path) -> bool {
    let s = p.as_os_str().as_bytes();
    s == b"/proc/self/exe" || s == b"/proc/thread-self/exe"
}

/// The path a `sockaddr_un` of `len` bytes at `addr` names, or `None` for
/// an address the kernel should see as the guest built it: not `AF_UNIX`,
/// unnamed, or in Linux's abstract namespace (a leading NUL), which no
/// filesystem path can reach. Linux's `sockaddr_un` is `sa_family_t` then
/// `sun_path[108]`.
pub fn unix_socket_path(addr: u64, len: usize) -> Option<Vec<u8>> {
    const HEADER: usize = 2;
    const SUN_PATH: usize = 108;
    let len = len.min(HEADER + SUN_PATH);
    if len <= HEADER {
        return None;
    }
    let mut buf = vec![0u8; len];
    if !copy_from_guest(addr, &mut buf)
        || u16::from_ne_bytes([buf[0], buf[1]]) as i32 != libc::AF_UNIX
    {
        return None;
    }
    let path = &buf[HEADER..];
    let end = path.iter().position(|&b| b == 0).unwrap_or(path.len());
    (end > 0).then(|| path[..end].to_vec())
}

/// A `sockaddr_un` naming the host path `path`, NUL-terminated, or `None`
/// when the path does not fit `sun_path`.
pub fn unix_sockaddr(path: &[u8]) -> Option<Vec<u8>> {
    const SUN_PATH: usize = 108;
    if path.len() >= SUN_PATH {
        return None;
    }
    let mut out = Vec::with_capacity(2 + path.len() + 1);
    out.extend_from_slice(&(libc::AF_UNIX as u16).to_ne_bytes());
    out.extend_from_slice(path);
    out.push(0);
    Some(out)
}

/// `execveat(fd, "", AT_EMPTY_PATH)`: the fd names the file.
pub fn exec_names_fd(path: &[u8], flags: i32) -> bool {
    path.is_empty() && flags & libc::AT_EMPTY_PATH != 0
}

/// `timeval` microseconds as nanoseconds.
pub fn usec_to_nsec(usec: libc::suseconds_t) -> i64 {
    usec * 1000
}

/// The `AT_*` flags an attribute mutator accepts, decoded to whether the
/// final symlink is named itself and whether an empty path names the dirfd.
pub fn decode_at_flags(flags: i32) -> Result<(bool, bool), Errno> {
    const ALLOWED: i32 = libc::AT_EMPTY_PATH | libc::AT_SYMLINK_NOFOLLOW;
    if flags & !ALLOWED != 0 {
        return Err(Errno::EINVAL);
    }
    Ok((
        flags & libc::AT_SYMLINK_NOFOLLOW == 0,
        flags & libc::AT_EMPTY_PATH != 0,
    ))
}

/// Apply the kernel's `open(2)` rules to the raw flags and mode before any
/// filesystem sees them. open(2) ignores `mode` unless the call creates a
/// file (O_CREAT or O_TMPFILE) — the third argument is variadic and callers
/// routinely pass a stale value on plain reads — and even on a creating open
/// it masks everything outside the permission bits, so the common idiom
/// open(dst, O_CREAT, src_stat.st_mode) works with S_IFREG included (libuv's
/// uv_fs_copyfile does exactly that). openat2, which HostFs resolves through,
/// instead rejects a nonzero mode on a non-creating open and unknown mode
/// bits with EINVAL. The same treatment for O_PATH companions: open(2)
/// silently masks everything but O_CLOEXEC, O_DIRECTORY, and O_NOFOLLOW out
/// of an O_PATH open, while openat2 rejects the extras with EINVAL — and
/// callers do pass them (Bun's existence probes open O_PATH|O_LARGEFILE).
pub fn sanitize_open(flags: i32, mode: u32) -> (i32, u32) {
    let mode = if flags & libc::O_CREAT != 0 || flags & libc::O_TMPFILE == libc::O_TMPFILE {
        mode & 0o7777
    } else {
        0
    };
    let flags = if flags & libc::O_PATH != 0 {
        flags & (libc::O_PATH | libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
    } else {
        flags
    };
    (flags, mode)
}

/// Reopen the object behind a host descriptor through its magic link.
pub fn reopen_host_fd(fd: i32, flags: i32, mode: u32) -> Result<i64, Errno> {
    let path = format!("/proc/self/fd/{fd}\0");
    check_host(unsafe { libc::open(path.as_ptr() as *const libc::c_char, flags, mode) })
}

impl SystemCalls for Personality {
    fn do_syscall(&self, call: &mut SystemCall) {
        let nr = call.number as i64;
        let a = call.args;
        let result = match nr {
            // --- open / create ---
            libc::SYS_openat => self.open(a[0] as i32, a[1], a[2] as i32, a[3] as u32),
            libc::SYS_open => self.open(libc::AT_FDCWD, a[0], a[1] as i32, a[2] as u32),
            // openat2 carries its flags/mode in a `struct open_how { u64 flags;
            // u64 mode; u64 resolve; }`; route it like openat so it cannot dodge
            // the Vfs and the read-only check. The `resolve` field's extra
            // restrictions are not yet honored (the namespace already confines).
            libc::SYS_openat2 => {
                let how = a[2];
                let flags = unsafe { ptr::read(how as *const u64) } as i32;
                let mode = unsafe { ptr::read((how as *const u64).add(1)) } as u32;
                self.open(a[0] as i32, a[1], flags, mode)
            }
            libc::SYS_creat => self.open(
                libc::AT_FDCWD,
                a[0],
                libc::O_CREAT | libc::O_WRONLY | libc::O_TRUNC,
                a[1] as u32,
            ),

            // --- per-fd I/O (virtual only; host fds fall through) ---
            libc::SYS_read if self.is_virtual(a[0] as i32) => {
                self.read(a[0] as i32, a[1], a[2] as usize)
            }
            libc::SYS_write if self.is_virtual(a[0] as i32) => {
                self.write(a[0] as i32, a[1], a[2] as usize)
            }
            libc::SYS_pread64 if self.is_virtual(a[0] as i32) => {
                self.pread(a[0] as i32, a[1], a[2] as usize, a[3])
            }
            libc::SYS_pwrite64 if self.is_virtual(a[0] as i32) => {
                self.pwrite(a[0] as i32, a[1], a[2] as usize, a[3])
            }
            libc::SYS_readv if self.is_virtual(a[0] as i32) => {
                self.readv(a[0] as i32, a[1], a[2] as usize)
            }
            libc::SYS_writev if self.is_virtual(a[0] as i32) => {
                self.writev(a[0] as i32, a[1], a[2] as usize)
            }
            libc::SYS_lseek if self.is_virtual(a[0] as i32) => {
                self.lseek(a[0] as i32, a[1] as i64, a[2] as i32)
            }
            libc::SYS_getdents64 if self.is_virtual(a[0] as i32) => {
                self.read_dirents(a[0] as i32, a[1], a[2] as usize, |entries, start, max| {
                    let (bytes, next) = encode_dirents(entries, start, max);
                    let len = bytes.len() as i64;
                    Ok((bytes, next, len))
                })
            }
            libc::SYS_fstat if self.is_virtual(a[0] as i32) => self
                .stat_fd(a[0] as i32)
                .map(|s| write_stat(a[1], &s))
                .map(|()| 0),
            libc::SYS_fstatfs if self.is_virtual(a[0] as i32) => self
                .fstatfs(a[0] as i32)
                .map(|s| write_statfs(a[1], &s))
                .map(|()| 0),
            libc::SYS_fsync | libc::SYS_fdatasync if self.is_virtual(a[0] as i32) => {
                self.fsync(a[0] as i32)
            }
            libc::SYS_ftruncate if self.is_virtual(a[0] as i32) => {
                self.ftruncate(a[0] as i32, a[1])
            }
            libc::SYS_close if self.is_virtual(a[0] as i32) => {
                self.fds.lock().unwrap().close(a[0] as i32).map(|_| 0)
            }
            // An ioctl on a special file must reach the real device —
            // `grantpt`/`unlockpt`, `TCGETS` (isatty), `TIOCSWINSZ` — or no
            // pty or tty ever works, so those forward to the host on the
            // backing fd. Regular files and directories keep answering
            // ENOTTY: their ioctls can mutate the filesystem itself
            // (`FS_IOC_SETFLAGS`, `FICLONE`), which would bypass the
            // read-only mount.
            libc::SYS_ioctl if self.is_virtual(a[0] as i32) => {
                match self.desc(a[0] as i32).and_then(|d| d.file.fstat()) {
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
                }
            }

            // --- dup / fcntl (virtual only) ---
            libc::SYS_dup if self.is_virtual(a[0] as i32) => self.dup(a[0] as i32, 0, false),
            libc::SYS_dup2 if self.is_virtual(a[0] as i32) => {
                self.dup2(a[0] as i32, a[1] as i32, 0)
            }
            libc::SYS_dup3 if self.is_virtual(a[0] as i32) => {
                self.dup2(a[0] as i32, a[1] as i32, a[2] as i32)
            }
            // dup2/dup3 from a host fd forwards, but the kernel's dup2
            // atomically replaces whatever occupies `newfd` — if that was a
            // virtual descriptor (bash restoring a saved stdin over a
            // redirect, `dup2(saved, 0)`), its reservation is closed by the
            // host call itself and the table entry must go with it, or the
            // table would keep intercepting a number that is now a plain
            // host fd.
            libc::SYS_dup2 | libc::SYS_dup3 if self.is_virtual(a[1] as i32) => {
                // Kernel replacement and table removal under one hold of the
                // lock; see [`Personality::dup`] for why the pair is atomic.
                let mut fds = self.fds.lock().unwrap();
                let result = host_syscall(call);
                if let SyscallResult::Ok(_) = result {
                    fds.map.remove(&(a[1] as i32));
                }
                call.set_result(result);
                return;
            }
            libc::SYS_fcntl if self.is_virtual(a[0] as i32) => {
                self.fcntl(a[0] as i32, a[1] as i32, a[2])
            }

            // --- path metadata ---
            libc::SYS_newfstatat => self.newfstatat(a[0] as i32, a[1], a[2], a[3] as i32),
            libc::SYS_stat => self
                .stat_at(libc::AT_FDCWD, &unsafe { read_cstr(a[0]) }, true)
                .map(|s| write_stat(a[1], &s))
                .map(|()| 0),
            libc::SYS_lstat => self
                .stat_at(libc::AT_FDCWD, &unsafe { read_cstr(a[0]) }, false)
                .map(|s| write_stat(a[1], &s))
                .map(|()| 0),
            libc::SYS_statx => self.statx(a[0] as i32, a[1], a[2] as i32, a[4]),
            libc::SYS_statfs => self
                .statfs_path(a[0])
                .map(|s| write_statfs(a[1], &s))
                .map(|()| 0),
            libc::SYS_access => self.access(libc::AT_FDCWD, a[0], a[1] as i32),
            libc::SYS_faccessat | libc::SYS_faccessat2 => {
                self.access(a[0] as i32, a[1], a[2] as i32)
            }
            libc::SYS_readlinkat => self.readlink(a[0] as i32, a[1], a[2], a[3] as usize),
            libc::SYS_readlink => self.readlink(libc::AT_FDCWD, a[0], a[1], a[2] as usize),
            libc::SYS_truncate => self.truncate(a[0], a[1]),

            // --- path mutation ---
            libc::SYS_mkdirat => self.mkdir(a[0] as i32, a[1], a[2] as u32),
            libc::SYS_mkdir => self.mkdir(libc::AT_FDCWD, a[0], a[1] as u32),
            libc::SYS_unlinkat => self.unlinkat(a[0] as i32, a[1], a[2] as i32),
            libc::SYS_unlink => self.unlinkat(libc::AT_FDCWD, a[0], 0),
            libc::SYS_rmdir => self.unlinkat(libc::AT_FDCWD, a[0], libc::AT_REMOVEDIR),
            libc::SYS_renameat2 => self.rename(a[0] as i32, a[1], a[2] as i32, a[3], a[4] as u32),
            libc::SYS_renameat => self.rename(a[0] as i32, a[1], a[2] as i32, a[3], 0),
            libc::SYS_rename => self.rename(libc::AT_FDCWD, a[0], libc::AT_FDCWD, a[1], 0),
            libc::SYS_symlinkat => self.symlink(a[0], a[1] as i32, a[2]),
            libc::SYS_symlink => self.symlink(a[0], libc::AT_FDCWD, a[1]),
            libc::SYS_linkat => self.link(a[0] as i32, a[1], a[2] as i32, a[3]),
            libc::SYS_link => self.link(libc::AT_FDCWD, a[0], libc::AT_FDCWD, a[1]),

            // close_range: pre-exec hygiene ("close or CLOEXEC-mark every fd
            // from N up" — Bun marks 3.. before every spawn) sweeps over fd
            // numbers the guest cannot see: the runtime's backing descriptors
            // and the HostFs root, which live at or above BACKING_FLOOR.
            // Letting the sweep reach them closes the VFS out from under the
            // process (the next emulated execve then cannot even load ld.so's
            // libraries). Apply the guest's request to what the guest owns:
            // table entries in the range are closed or CLOEXEC-marked in the
            // table, and the host range is clamped below the floor.
            libc::SYS_close_range => {
                let first = a[0] as i64;
                let last = (a[1] as i64).min(i32::MAX as i64);
                let cloexec = a[2] as u32 & libc::CLOSE_RANGE_CLOEXEC != 0;
                {
                    let mut fds = self.fds.lock().unwrap();
                    let in_range: Vec<i32> = fds
                        .map
                        .keys()
                        .copied()
                        .filter(|&n| (n as i64) >= first && (n as i64) <= last)
                        .collect();
                    for n in in_range {
                        if cloexec {
                            fds.map.get_mut(&n).unwrap().cloexec = true;
                        } else {
                            let _ = fds.close(n);
                        }
                    }
                }
                let host_last = last.min(BACKING_FLOOR as i64 - 1);
                if first <= host_last {
                    check_host(unsafe {
                        libc::syscall(libc::SYS_close_range, first, host_last, a[2]) as libc::c_int
                    })
                } else {
                    Ok(0)
                }
            }

            // --- cwd ---
            libc::SYS_getcwd => self.getcwd(a[0], a[1] as usize),
            libc::SYS_chdir => self.chdir(a[0]),
            libc::SYS_fchdir if self.is_virtual(a[0] as i32) => self.fchdir(a[0] as i32),

            // --- attribute mutators (path forms; fd forms virtual only) ---
            libc::SYS_chmod => self.chmod_path(libc::AT_FDCWD, a[0], a[1] as u32, 0),
            // The raw fchmodat syscall has no flags argument; fchmodat2 adds it.
            libc::SYS_fchmodat => self.chmod_path(a[0] as i32, a[1], a[2] as u32, 0),
            libc::SYS_fchmodat2 => self.chmod_path(a[0] as i32, a[1], a[2] as u32, a[3] as i32),
            libc::SYS_fchmod if self.is_virtual(a[0] as i32) => {
                self.fchmod(a[0] as i32, a[1] as u32)
            }
            libc::SYS_chown => self.chown_path(libc::AT_FDCWD, a[0], a[1] as u32, a[2] as u32, 0),
            libc::SYS_lchown => self.chown_path(
                libc::AT_FDCWD,
                a[0],
                a[1] as u32,
                a[2] as u32,
                libc::AT_SYMLINK_NOFOLLOW,
            ),
            libc::SYS_fchownat => {
                self.chown_path(a[0] as i32, a[1], a[2] as u32, a[3] as u32, a[4] as i32)
            }
            libc::SYS_fchown if self.is_virtual(a[0] as i32) => {
                self.fchown(a[0] as i32, a[1] as u32, a[2] as u32)
            }
            libc::SYS_utime => self.utime(a[0], a[1]),
            libc::SYS_utimes => self.utimes(libc::AT_FDCWD, a[0], a[1]),
            // futimesat(fd, NULL, …) targets the descriptor itself; that form
            // on a host fd falls through to the host below.
            libc::SYS_futimesat if a[1] != 0 || self.is_virtual(a[0] as i32) => {
                self.utimes(a[0] as i32, a[1], a[2])
            }
            // utimensat(fd, NULL, …) targets the descriptor itself; that form
            // on a host fd falls through to the host below.
            libc::SYS_utimensat if a[1] != 0 || self.is_virtual(a[0] as i32) => {
                self.utimensat(a[0] as i32, a[1], a[2], a[3] as i32)
            }
            libc::SYS_mknod => self.mknod(libc::AT_FDCWD, a[0], a[1] as u32, a[2]),
            libc::SYS_mknodat => self.mknod(a[0] as i32, a[1], a[2] as u32, a[3]),
            libc::SYS_setxattr => {
                self.setxattr_path(a[0], a[1], a[2], a[3] as usize, a[4] as i32, true)
            }
            libc::SYS_lsetxattr => {
                self.setxattr_path(a[0], a[1], a[2], a[3] as usize, a[4] as i32, false)
            }
            libc::SYS_fsetxattr if self.is_virtual(a[0] as i32) => {
                self.fsetxattr(a[0] as i32, a[1], a[2], a[3] as usize, a[4] as i32)
            }
            libc::SYS_removexattr => self.removexattr_path(a[0], a[1], true),
            libc::SYS_lremovexattr => self.removexattr_path(a[0], a[1], false),
            libc::SYS_fremovexattr if self.is_virtual(a[0] as i32) => {
                self.fremovexattr(a[0] as i32, a[1])
            }
            // Xattr reads carry no write policy, but the path must still
            // resolve through the namespace: the file may live only in an
            // overlay's delta, and the host kernel then serves the backing
            // path. (A virtual fd is a kernel alias of its backing file, so
            // the fd forms forward as-is.) The overlay's own bookkeeping
            // namespace is invisible throughout: a marker name reads as
            // absent, and marker entries vanish from listings.
            libc::SYS_getxattr | libc::SYS_lgetxattr => {
                if marker_name(a[1]) {
                    call.set_result(SyscallResult::Error(libc::ENODATA));
                } else {
                    self.forward_at_host_path(call, 0, nr == libc::SYS_getxattr);
                }
                return;
            }
            libc::SYS_listxattr | libc::SYS_llistxattr => {
                self.forward_at_host_path(call, 0, nr == libc::SYS_listxattr);
                let result = call.result().expect("a forwarded call has a result");
                call.set_result(strip_marker_names(result, a[1], a[2] as usize));
                return;
            }
            libc::SYS_fgetxattr if marker_name(a[1]) => Err(Errno(libc::ENODATA)),
            libc::SYS_flistxattr => {
                let result = host_syscall(call);
                call.set_result(strip_marker_names(result, a[1], a[2] as usize));
                return;
            }
            libc::SYS_fallocate if self.is_virtual(a[0] as i32) => {
                self.fallocate(a[0] as i32, a[1] as i32, a[2], a[3])
            }

            // Everything else — including host-fd I/O and syscalls not modeled
            // here — goes to the host kernel unchanged.
            // A Unix-domain socket's name is a path in the namespace.
            libc::SYS_bind => {
                self.bind_unix(call);
                return;
            }
            libc::SYS_connect => {
                self.connect_unix(call);
                return;
            }

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

    // Phase 1 does not drop close-on-exec descriptors across a successful
    // `execve`: the runtime reads the new image (possibly *from* such a
    // descriptor, as `execveat(fd, "", AT_EMPTY_PATH)` does) after the syscall
    // is observed but with no hook back here once the load succeeds, so closing
    // them here would race ahead of that read. The cost is a leaked cloexec fd
    // across exec — benign and revisited when exec reads through the Vfs.
}

impl Personality {
    /// `statfs`: the statistics of the filesystem serving a path.
    fn statfs_path(&self, pathptr: u64) -> Result<StatFs, Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        let abs = self.abs_path(libc::AT_FDCWD, &raw)?;
        let r = self.ns.resolve(&abs, true)?;
        r.fs.statfs(&r.rel)
    }

    /// `fstatfs`: the statistics of the filesystem behind an open file.
    fn fstatfs(&self, fd: i32) -> Result<StatFs, Errno> {
        self.desc(fd)?.file.fstatfs()
    }

    fn getcwd(&self, buf: u64, size: usize) -> Result<i64, Errno> {
        let mut bytes = self.cwd.lock().unwrap().as_os_str().as_bytes().to_vec();
        bytes.push(0);
        if bytes.len() > size {
            return Err(Errno::ERANGE);
        }
        guest_mut(buf, bytes.len()).copy_from_slice(&bytes);
        Ok(bytes.len() as i64) // getcwd returns the length including the NUL
    }

    fn fallocate(&self, fd: i32, mode: i32, offset: u64, len: u64) -> Result<i64, Errno> {
        self.writable_desc(fd)?.file.fallocate(mode, offset, len)?;
        Ok(0)
    }

    fn newfstatat(&self, dirfd: i32, pathptr: u64, buf: u64, flags: i32) -> Result<i64, Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        // AT_EMPTY_PATH with an empty path stats the dirfd itself.
        let s = if raw.is_empty() && flags & libc::AT_EMPTY_PATH != 0 && self.is_virtual(dirfd) {
            self.stat_fd(dirfd)?
        } else {
            self.stat_at(dirfd, &raw, flags & libc::AT_SYMLINK_NOFOLLOW == 0)?
        };
        write_stat(buf, &s);
        Ok(0)
    }

    fn statx(&self, dirfd: i32, pathptr: u64, flags: i32, buf: u64) -> Result<i64, Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        let s = if raw.is_empty() && flags & libc::AT_EMPTY_PATH != 0 && self.is_virtual(dirfd) {
            self.stat_fd(dirfd)?
        } else {
            self.stat_at(dirfd, &raw, flags & libc::AT_SYMLINK_NOFOLLOW == 0)?
        };
        write_statx(buf, &s);
        Ok(0)
    }

    fn fcntl(&self, fd: i32, cmd: i32, arg: u64) -> Result<i64, Errno> {
        match cmd {
            libc::F_DUPFD => self.dup(fd, arg as i32, false),
            libc::F_DUPFD_CLOEXEC => self.dup(fd, arg as i32, true),
            libc::F_GETFD => Ok(if self.cloexec(fd)? {
                libc::FD_CLOEXEC as i64
            } else {
                0
            }),
            libc::F_SETFD => self.set_cloexec(fd, arg as i32 & libc::FD_CLOEXEC != 0),
            libc::F_GETFL => self.status_flags(fd),
            libc::F_SETFL => self.set_status_flags(
                fd,
                arg as i32,
                libc::O_APPEND | libc::O_NONBLOCK | libc::O_DIRECT | libc::O_NOATIME,
            ),
            libc::F_SETLK
            | libc::F_SETLKW
            | libc::F_GETLK
            | libc::F_OFD_SETLK
            | libc::F_OFD_SETLKW
            | libc::F_OFD_GETLK => self.lock(fd, cmd, arg),
            _ => Err(Errno::EINVAL),
        }
    }

    fn utime(&self, pathptr: u64, bufptr: u64) -> Result<i64, Errno> {
        // struct utimbuf: [actime, modtime] in whole seconds; NULL = both now.
        let times = if bufptr == 0 {
            None
        } else {
            let b = unsafe { ptr::read(bufptr as *const [libc::time_t; 2]) };
            Some(b.map(|sec| Timespec { sec, nsec: 0 }))
        };
        self.utimens_path(libc::AT_FDCWD, pathptr, times, true)
    }

    fn utimensat(&self, dirfd: i32, pathptr: u64, tsptr: u64, flags: i32) -> Result<i64, Errno> {
        // A NULL pointer (not an empty string) is glibc's futimens: the fd
        // names the file and flags do not apply. Only the empty-string form
        // goes through at_target's AT_EMPTY_PATH decode.
        if pathptr == 0 && flags != 0 {
            return Err(Errno::EINVAL);
        }
        let times = if tsptr == 0 {
            None
        } else {
            let ts = unsafe { ptr::read(tsptr as *const [libc::timespec; 2]) };
            Some(ts.map(|t| Timespec {
                sec: t.tv_sec,
                nsec: t.tv_nsec,
            }))
        };
        if pathptr == 0 {
            self.writable_desc(dirfd)?.file.futimens(times)?;
            return Ok(0);
        }
        match self.at_target(dirfd, pathptr, flags)? {
            AtTarget::Fd(desc) => desc.file.utimensat_empty(times)?,
            AtTarget::Path { resolved, follow } => {
                resolved.fs.utimens(&resolved.rel, follow, times)?
            }
        }
        Ok(0)
    }
}

/// Fill a guest `struct stat` from a symbolic [`Stat`]. On x86-64 the kernel and
/// glibc `struct stat` share a layout, so a `libc::stat` written here is exactly
/// what the guest's `fstat`/`newfstatat` expects.
fn write_stat(ptr: u64, s: &Stat) {
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    st.st_dev = s.dev;
    st.st_ino = s.ino;
    st.st_mode = s.mode.0;
    st.st_nlink = s.nlink;
    st.st_uid = s.uid;
    st.st_gid = s.gid;
    st.st_rdev = s.rdev;
    st.st_size = s.size as i64;
    st.st_blksize = s.blksize as i64;
    st.st_blocks = s.blocks as i64;
    st.st_atime = s.atime.sec;
    st.st_atime_nsec = s.atime.nsec;
    st.st_mtime = s.mtime.sec;
    st.st_mtime_nsec = s.mtime.nsec;
    st.st_ctime = s.ctime.sec;
    st.st_ctime_nsec = s.ctime.nsec;
    unsafe { ptr::write(ptr as *mut libc::stat, st) };
}

/// Fill a guest `struct statfs` from a symbolic [`StatFs`].
fn write_statfs(ptr: u64, s: &StatFs) {
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

    let st = RawStatFs {
        f_type: s.fs_type as libc::c_long,
        f_bsize: s.block_size as libc::c_long,
        f_blocks: s.blocks,
        f_bfree: s.blocks_free,
        f_bavail: s.blocks_available,
        f_files: s.files,
        f_ffree: s.files_free,
        f_fsid: s.fsid,
        f_namelen: s.name_max as libc::c_long,
        f_frsize: s.fragment_size as libc::c_long,
        f_flags: s.flags as libc::c_long,
        f_spare: [0; 4],
    };
    unsafe { ptr::write(ptr as *mut RawStatFs, st) };
}

/// Fill a guest `struct statx` from a symbolic [`Stat`], reporting the basic
/// stat fields (the only ones a [`Stat`] carries).
fn write_statx(ptr: u64, s: &Stat) {
    let ts = |t: Timespec| {
        let mut v: libc::statx_timestamp = unsafe { std::mem::zeroed() };
        v.tv_sec = t.sec;
        v.tv_nsec = t.nsec as u32;
        v
    };
    let mut stx: libc::statx = unsafe { std::mem::zeroed() };
    stx.stx_mask = libc::STATX_BASIC_STATS;
    stx.stx_blksize = s.blksize as u32;
    stx.stx_nlink = s.nlink as u32;
    stx.stx_uid = s.uid;
    stx.stx_gid = s.gid;
    stx.stx_mode = s.mode.0 as u16;
    stx.stx_ino = s.ino;
    stx.stx_size = s.size;
    stx.stx_blocks = s.blocks;
    stx.stx_atime = ts(s.atime);
    stx.stx_mtime = ts(s.mtime);
    stx.stx_ctime = ts(s.ctime);
    stx.stx_rdev_major = libc::major(s.rdev);
    stx.stx_rdev_minor = libc::minor(s.rdev);
    stx.stx_dev_major = libc::major(s.dev);
    stx.stx_dev_minor = libc::minor(s.dev);
    unsafe { ptr::write(ptr as *mut libc::statx, stx) };
}

/// Encode directory `entries` from index `start` into `linux_dirent64` records,
/// stopping before the byte total would exceed `max`. Returns the encoded bytes
/// and the index to resume from.
fn encode_dirents(entries: &[DirEntry], start: usize, max: usize) -> (Vec<u8>, usize) {
    // struct linux_dirent64 { u64 d_ino; i64 d_off; u16 d_reclen; u8 d_type; char d_name[]; }
    const HDR: usize = 19;
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
        rec[8..16].copy_from_slice(&((idx as i64) + 1).to_ne_bytes());
        rec[16..18].copy_from_slice(&(reclen as u16).to_ne_bytes());
        rec[18] = dirent_type(e.file_type);
        rec[19..19 + name.len()].copy_from_slice(name);
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

#[cfg(test)]
mod tests {
    use std::{
        os::unix::fs::{MetadataExt, PermissionsExt},
        sync::{
            Arc,
            atomic::{AtomicU32, Ordering},
        },
    };

    use super::*;
    use crate::{HostFs, MountFlags, sys::fs::namespace::Namespace};

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("chimera-personality-{}-{n}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn personality(&self, flags: MountFlags) -> Personality {
            let root = Arc::new(HostFs::new(&self.0).unwrap());
            Personality::new(Namespace::with_root(root, flags))
        }

        fn open(&self, personality: &Personality, path: &std::ffi::CStr, flags: i32) -> i32 {
            personality
                .open(libc::AT_FDCWD, path.as_ptr() as u64, flags, 0)
                .unwrap() as i32
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn fd_write_guard_survives_rename() {
        let scratch = Scratch::new();
        let personality = scratch.personality(MountFlags::NONE);
        std::fs::write(scratch.0.join("target"), b"x").unwrap();

        let fd = scratch.open(&personality, c"/target", libc::O_RDONLY);
        std::fs::rename(scratch.0.join("target"), scratch.0.join("moved")).unwrap();

        assert!(personality.writable_desc(fd).is_ok());
    }

    #[test]
    fn fd_write_guard_survives_unlink() {
        let scratch = Scratch::new();
        let personality = scratch.personality(MountFlags::NONE);
        std::fs::write(scratch.0.join("target"), b"x").unwrap();

        let fd = scratch.open(&personality, c"/target", libc::O_RDONLY);
        std::fs::remove_file(scratch.0.join("target")).unwrap();

        assert!(personality.writable_desc(fd).is_ok());
    }

    #[test]
    fn fd_write_guard_denies_read_only_mount() {
        let scratch = Scratch::new();
        std::fs::write(scratch.0.join("target"), b"x").unwrap();
        let personality = scratch.personality(MountFlags::RDONLY);

        let fd = scratch.open(&personality, c"/target", libc::O_RDONLY);

        assert_eq!(personality.writable_desc(fd).err(), Some(Errno::EROFS));
    }

    /// The `AT_EMPTY_PATH` forms target the open inode, so they must land on
    /// the file the descriptor was opened on even after the path is renamed
    /// away and replaced — not on the path's new occupant.
    #[test]
    fn empty_path_operations_target_opath_file_after_replacement() {
        let scratch = Scratch::new();
        let personality = scratch.personality(MountFlags::NONE);
        let target = scratch.0.join("target");
        let moved = scratch.0.join("moved");
        std::fs::write(&target, b"old").unwrap();

        let fd = scratch.open(&personality, c"/target", libc::O_PATH);
        let fd_ro = scratch.open(&personality, c"/target", libc::O_RDONLY);
        std::fs::rename(&target, &moved).unwrap();
        std::fs::write(&target, b"replacement").unwrap();
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o600)).unwrap();

        personality
            .chmod_path(fd, c"".as_ptr() as u64, 0o640, libc::AT_EMPTY_PATH)
            .unwrap();
        personality
            .chown_path(
                fd,
                c"".as_ptr() as u64,
                u32::MAX,
                u32::MAX,
                libc::AT_EMPTY_PATH,
            )
            .unwrap();
        let empty_times = [libc::timespec {
            tv_sec: 111,
            tv_nsec: 0,
        }; 2];
        personality
            .utimensat(
                fd,
                c"".as_ptr() as u64,
                empty_times.as_ptr() as u64,
                libc::AT_EMPTY_PATH,
            )
            .unwrap();

        // utimes(fd, NULL-path, …) likewise names the descriptor itself — but
        // only through a regular descriptor: the kernel's NULL-path form
        // rejects an O_PATH handle with EBADF (fdget refuses FMODE_PATH),
        // unlike the AT_EMPTY_PATH form above.
        let null_times = [
            libc::timeval {
                tv_sec: 222,
                tv_usec: 0,
            },
            libc::timeval {
                tv_sec: 333,
                tv_usec: 0,
            },
        ];
        assert_eq!(
            personality.utimes(fd, 0, null_times.as_ptr() as u64),
            Err(Errno(libc::EBADF))
        );
        personality
            .utimes(fd_ro, 0, null_times.as_ptr() as u64)
            .unwrap();

        let moved_stat = std::fs::metadata(&moved).unwrap();
        assert_eq!(moved_stat.permissions().mode() & 0o7777, 0o640);
        assert_eq!(moved_stat.mtime(), 333);
        assert_eq!(
            std::fs::metadata(&target).unwrap().permissions().mode() & 0o7777,
            0o600
        );
    }

    #[test]
    fn chmod_by_path_is_erofs_on_read_only_mount() {
        let scratch = Scratch::new();
        std::fs::write(scratch.0.join("target"), b"x").unwrap();
        let personality = scratch.personality(MountFlags::RDONLY);

        assert_eq!(
            personality.chmod_path(libc::AT_FDCWD, c"/target".as_ptr() as u64, 0o600, 0),
            Err(Errno::EROFS)
        );
    }

    /// A host fd is kept away from the guard by the dispatch arms'
    /// `is_virtual` gates; the guard itself answers EBADF for any
    /// descriptor it does not own.
    #[test]
    fn fd_write_guard_rejects_unknown_fds() {
        let scratch = Scratch::new();
        let personality = scratch.personality(MountFlags::RDONLY);

        assert_eq!(
            personality.writable_desc(libc::STDIN_FILENO).err(),
            Some(Errno(libc::EBADF))
        );
    }
}
