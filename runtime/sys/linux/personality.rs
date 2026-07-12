//! The [`Personality`]: the [`SystemCalls`] handler that turns a guest's Linux
//! filesystem syscalls into calls on a [`Vfs`], and owns the descriptor table,
//! current directory, and Linux-ABI encoding that the `Vfs`/`File` traits are
//! kept free of.
//!
//! It joins a guest path against the cwd or a dirfd, hands it to the
//! [`Namespace`] for confining, symlink-resolving, mount-crossing resolution,
//! and dispatches the resulting `(Vfs, rel)` — denying writes to a read-only
//! mount with `EROFS`. Descriptors a filesystem hands out are *virtual* —
//! each one holds a reserved kernel descriptor number (see [`reserve_fd`]) so
//! the guest sees the low, dense, lowest-available numbers POSIX promises —
//! and any syscall on a descriptor not in the table, or any syscall not
//! handled here, is forwarded to the host unchanged. `execve` targets resolve through the namespace too
//! ([`SystemCalls::resolve_exec`]), so a confined guest cannot exec outside its
//! root.

use std::{
    borrow::Cow,
    collections::HashMap,
    ffi::{CStr, OsStr, OsString},
    os::{
        fd::{IntoRawFd, RawFd},
        unix::ffi::OsStrExt,
    },
    path::{Path, PathBuf},
    ptr, slice,
    sync::{Arc, Mutex},
};

use crate::{ForkHold, SyscallResult, SystemCall, SystemCalls, host_syscall};

use super::{
    hostfs::BACKING_FLOOR,
    namespace::{Namespace, Resolved, normalize},
    vfs::{DirEntry, Errno, File, FileType, Mode, OpenFlags, RenameFlags, Stat, StatFs, Timespec},
};

/// Reserve a real kernel descriptor number for a virtual descriptor.
///
/// Guest programs legitimately assume POSIX's "lowest available number" for
/// new descriptors — bash indexes a per-fd array by the number (a sparse fd
/// like `0x4000_0000` once made it allocate and zero 8 GiB per redirect), and
/// `FD_SET` corrupts memory beyond 1023 — so virtual descriptors must live in
/// the same low, dense namespace as host ones. Duplicating the backing host
/// descriptor both claims a kernel-allocated number (which therefore can never
/// collide with any host fd) and keeps the number aliased to the real file, so
/// unmodeled passthrough syscalls on it still reach the right object. A
/// synthetic file with no host backing reserves its number on `/dev/null`.
///
/// The reservation carries no host `FD_CLOEXEC`: Chimera emulates `execve` in
/// place, so the runtime sweeps host close-on-exec fds by hand at exec (see
/// `close_cloexec_fds` in `super::exec`), and a flagged reservation would be
/// swept out from under the table. The guest's close-on-exec flag lives in
/// the [`FdTable`] instead and is honored by [`SystemCalls::on_execve`].
fn reserve_fd(backing: Option<RawFd>) -> Result<i32, Errno> {
    let fd = match backing {
        Some(b) => unsafe { libc::fcntl(b, libc::F_DUPFD, 0) },
        None => unsafe { libc::open(c"/dev/null".as_ptr(), libc::O_RDONLY) },
    };
    if fd < 0 {
        return Err(Errno::from_io(&std::io::Error::last_os_error()));
    }
    Ok(fd)
}

/// The filesystem Personality: a namespace, a descriptor table, and a cwd.
///
/// One handler serves every guest thread ([`SystemCalls`] is `&self` +
/// `Send + Sync`), and Linux shares the descriptor table and cwd process-wide
/// across threads, so both live behind a `Mutex` — the lock *is* the sharing
/// semantics, not a workaround. Each is locked for the few instructions of one
/// table or path operation, never across a `Vfs` call.
pub struct Personality {
    ns: Namespace,
    fds: Mutex<FdTable>,
    cwd: Mutex<PathBuf>,
    /// The guest's executed image — what `/proc/self/exe` names. Chimera
    /// emulates `execve` in place, so the kernel's own link points at the
    /// runtime binary; leaking that gives a self-respawning guest (Bun's
    /// `execPath`) the runtime to exec instead of itself. Seeded by the
    /// embedder ([`Personality::set_exe`]), updated on every committed exec
    /// ([`SystemCalls::on_execve`]).
    exe: Mutex<Option<PathBuf>>,
}

impl Personality {
    /// Build a Personality over `ns`, inheriting the host process's current
    /// directory as the guest's initial cwd.
    pub fn new(ns: Namespace) -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        Self {
            ns,
            fds: Mutex::new(FdTable::new()),
            cwd: Mutex::new(normalize(&cwd)),
            exe: Mutex::new(None),
        }
    }

    /// Record the guest image the sandbox launches, for `/proc/self/exe`.
    /// Canonicalized so the guest sees the absolute path the kernel would
    /// store.
    pub fn set_exe(&self, path: impl Into<PathBuf>) {
        let path = path.into();
        let path = std::fs::canonicalize(&path).unwrap_or(path);
        *self.exe.lock().unwrap() = Some(path);
    }
}

/// `true` for a path the kernel resolves to the calling process's executed
/// image rather than through the filesystem.
fn is_magic_exe(p: &Path) -> bool {
    let s = p.as_os_str().as_bytes();
    s == b"/proc/self/exe" || s == b"/proc/thread-self/exe"
}

/// The descriptor table: virtual guest fd → open file. Mirrors Linux's split of
/// a descriptor (with its own `FD_CLOEXEC`) from the open file description it
/// points at (the shared offset and status flags), so `dup` shares an offset.
struct FdTable {
    map: HashMap<i32, Fd>,
}

struct Fd {
    desc: Arc<OpenFileDescription>,
    cloexec: bool,
}

struct OpenFileDescription {
    file: Arc<dyn File>,
    /// The absolute guest path this was opened by — the anchor for `openat`
    /// against this fd and `fchdir`.
    path: PathBuf,
    /// Open status flags (`O_APPEND`, `O_NONBLOCK`, …); the access mode lives
    /// here too. Shared across `dup`, mutable via `fcntl(F_SETFL)`.
    state: Mutex<DescState>,
    /// Mount writability at open time. Fd operations target the open inode, so
    /// this cannot be recovered by resolving `path` after a rename or unlink.
    writable: bool,
}

struct DescState {
    /// Byte offset for a file; entry index for a directory being walked.
    offset: u64,
    /// Status flags, seeded from the open flags.
    status: i32,
    /// Cached directory snapshot (with synthesized `.`/`..`), filled on the
    /// first `getdents64` and dropped on `lseek`.
    dirents: Option<Vec<DirEntry>>,
}

impl FdTable {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Install an open file at the kernel number [`reserve_fd`] (or the
    /// kernel's `dup` family) handed out for it.
    fn insert_at(&mut self, fd: i32, desc: Arc<OpenFileDescription>, cloexec: bool) {
        self.map.insert(fd, Fd { desc, cloexec });
    }

    fn get(&self, fd: i32) -> Option<&Fd> {
        self.map.get(&fd)
    }

    fn close(&mut self, fd: i32) -> Result<(), Errno> {
        self.map.remove(&fd).ok_or(Errno::EBADF)?;
        // Release the reserved kernel number so the guest sees it reused,
        // lowest-first, the way a real close behaves.
        unsafe { libc::close(fd) };
        Ok(())
    }
}

/// The descriptor named by a magic fd link — `/dev/fd/N`, `/proc/self/fd/N`,
/// or `/proc/thread-self/fd/N` — or `None` for any other path. The kernel
/// resolves these links to the open file itself, not through the filesystem
/// (their readlink target, `pipe:[…]` or `socket:[…]`, is not a path), so the
/// namespace walker cannot follow them; [`Personality::open`] services them
/// as fd reopens instead.
fn magic_fd_link(abs: &Path) -> Option<i32> {
    let s = abs.as_os_str().as_bytes();
    // The /dev/std* symlinks point into /proc/self/fd and would hit the
    // magic link one hop later; naming their well-known descriptors here
    // keeps the shell idioms (`< /dev/stdin`, `> /dev/stderr`) working.
    match s {
        b"/dev/stdin" => return Some(0),
        b"/dev/stdout" => return Some(1),
        b"/dev/stderr" => return Some(2),
        _ => {}
    }
    let n = s
        .strip_prefix(b"/dev/fd/")
        .or_else(|| s.strip_prefix(b"/proc/self/fd/"))
        .or_else(|| s.strip_prefix(b"/proc/thread-self/fd/"))?;
    std::str::from_utf8(n).ok()?.parse().ok()
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
                self.getdents64(a[0] as i32, a[1], a[2] as usize)
            }
            libc::SYS_fstat if self.is_virtual(a[0] as i32) => self.fstat(a[0] as i32, a[1]),
            libc::SYS_fstatfs if self.is_virtual(a[0] as i32) => self.fstatfs(a[0] as i32, a[1]),
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
            libc::SYS_stat => {
                self.fstatat_path(libc::AT_FDCWD, &unsafe { read_cstr(a[0]) }, a[1], true)
            }
            libc::SYS_lstat => {
                self.fstatat_path(libc::AT_FDCWD, &unsafe { read_cstr(a[0]) }, a[1], false)
            }
            libc::SYS_statx => self.statx(a[0] as i32, a[1], a[2] as i32, a[4]),
            libc::SYS_statfs => self.statfs(a[0], a[1]),
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

            // --- metadata mutators not yet on the Vfs trait ---
            //
            // These change a file's attributes rather than its path or contents.
            // The Vfs trait has no methods for them yet, but they must still
            // honor the read-only mount: deny on a read-only mount with EROFS,
            // and otherwise let the host serve them (the only mount today is the
            // host root, so the guest path is the host path). Left unhandled they
            // would fall through to the host and mutate it under `--readonly`.
            libc::SYS_mknod
            | libc::SYS_setxattr
            | libc::SYS_lsetxattr
            | libc::SYS_removexattr
            | libc::SYS_lremovexattr => {
                let guard = self.guard_write_path(libc::AT_FDCWD, a[0]);
                self.deny_ro_or_passthrough(call, guard, None);
                return;
            }
            libc::SYS_mknodat => {
                let guard = self.guard_write_path(a[0] as i32, a[1]);
                // Translate the dirfd if it is one of ours before forwarding.
                self.deny_ro_or_passthrough(call, guard, Some(0));
                return;
            }
            libc::SYS_fsetxattr | libc::SYS_fremovexattr | libc::SYS_fallocate => {
                let guard = self.guard_write_fd(a[0] as i32);
                self.deny_ro_or_passthrough(call, guard, Some(0));
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

    fn resolve_fd(&self, guest_fd: i32) -> Option<RawFd> {
        // A descriptor not in the table is already a host fd; leave it.
        self.fds
            .lock()
            .unwrap()
            .get(guest_fd)
            .and_then(|fd| fd.desc.file.host_fd())
    }

    fn on_execve(&self, path: &Path) {
        // The link target of /proc/self/exe follows the exec.
        *self.exe.lock().unwrap() = Some(path.to_path_buf());
        // POSIX exec semantics for the table: close-on-exec entries go, the
        // rest survive into the new image with their numbers intact. Closing
        // through `FdTable::close` releases each reserved kernel number, so
        // the runtime's host-fd sweep (which runs after this) finds it gone
        // and skips it.
        let mut fds = self.fds.lock().unwrap();
        let doomed: Vec<i32> = fds
            .map
            .iter()
            .filter(|(_, fd)| fd.cloexec)
            .map(|(&n, _)| n)
            .collect();
        for fd in doomed {
            let _ = fds.close(fd);
        }
    }

    fn lock_for_fork(&self) -> Box<dyn ForkHold + '_> {
        // The descriptor table pairs memory state with kernel state (the
        // reserved numbers and backing fds), so a fork's copy must not land
        // mid-update; cwd and exe just must not be copied mid-write.
        struct Held<'a> {
            _fds: std::sync::MutexGuard<'a, FdTable>,
            _cwd: std::sync::MutexGuard<'a, PathBuf>,
            _exe: std::sync::MutexGuard<'a, Option<PathBuf>>,
        }
        Box::new(Held {
            _fds: self.fds.lock().unwrap(),
            _cwd: self.cwd.lock().unwrap(),
            _exe: self.exe.lock().unwrap(),
        })
    }

    fn resolve_exec(&self, dirfd: i32, path: &[u8], flags: i32) -> Option<Result<PathBuf, i32>> {
        // Resolve the target through the namespace, then hand the loader the
        // host path that backs it — so a confined guest cannot exec outside its
        // root, and a synthetic filesystem (no host path) is reported as EACCES.
        Some(
            (|| {
                let abs: Cow<Path> = if path.is_empty() && flags & libc::AT_EMPTY_PATH != 0 {
                    // execveat(fd, "", AT_EMPTY_PATH): the fd names the file.
                    Cow::Owned(self.desc(dirfd)?.path.clone())
                } else {
                    self.abs_path(dirfd, path)?
                };
                let r = self.ns.resolve(&abs, true)?;
                r.fs.host_path(&r.rel).ok_or(Errno::EACCES)
            })()
            .map_err(|e: Errno| e.raw()),
        )
    }

    // Phase 1 does not drop close-on-exec descriptors across a successful
    // `execve`: the runtime reads the new image (possibly *from* such a
    // descriptor, as `execveat(fd, "", AT_EMPTY_PATH)` does) after the syscall
    // is observed but with no hook back here once the load succeeds, so closing
    // them here would race ahead of that read. The cost is a leaked cloexec fd
    // across exec — benign and revisited when exec reads through the Vfs.
}

impl Personality {
    /// `true` for a descriptor this Personality owns; `false` for any other
    /// (host) descriptor, which passes straight through.
    fn is_virtual(&self, fd: i32) -> bool {
        self.fds.lock().unwrap().map.contains_key(&fd)
    }

    /// Turn a `dirfd` + raw guest path into an absolute, normalized guest path.
    /// An already-absolute path — the common case — is borrowed, not copied.
    fn abs_path<'a>(&self, dirfd: i32, raw: &'a [u8]) -> Result<Cow<'a, Path>, Errno> {
        let p = Path::new(OsStr::from_bytes(raw));
        let joined: Cow<'a, Path> = if p.is_absolute() {
            Cow::Borrowed(p)
        } else if dirfd == libc::AT_FDCWD {
            Cow::Owned(self.cwd.lock().unwrap().join(p))
        } else {
            match self.fds.lock().unwrap().get(dirfd) {
                Some(fd) => Cow::Owned(fd.desc.path.join(p)),
                // A relative path against a host dirfd has no place in the
                // namespace.
                None => return Err(Errno::EBADF),
            }
        };
        // /proc/self/exe names the running image, which under an emulated
        // exec is the guest's binary, not Chimera's: leaking the runtime's
        // own path here hands a self-respawning guest the wrong program.
        if is_magic_exe(&joined)
            && let Some(exe) = self.exe.lock().unwrap().clone()
        {
            return Ok(Cow::Owned(exe));
        }
        // Not normalized here: the resolver must apply `..` *after* following
        // symlinks, not lexically, so it walks the joined path itself.
        Ok(joined)
    }

    /// Resolve a path mutator's target and reject it with `EROFS` if it lands on
    /// a read-only mount. Path-resolution errors (`ENOENT`, `ELOOP`, …) carry
    /// through unchanged.
    fn resolve_write(&self, dirfd: i32, pathptr: u64, follow: bool) -> Result<Resolved, Errno> {
        if pathptr == 0 {
            return Err(Errno(libc::EFAULT));
        }
        let raw = unsafe { read_cstr(pathptr) };
        if raw.is_empty() {
            return Err(Errno::ENOENT);
        }
        self.resolve_write_raw(dirfd, &raw, follow)
    }

    fn resolve_write_raw(&self, dirfd: i32, raw: &[u8], follow: bool) -> Result<Resolved, Errno> {
        let abs = self.abs_path(dirfd, raw)?;
        let r = self.ns.resolve(&abs, follow)?;
        if !r.writable {
            return Err(Errno::EROFS);
        }
        Ok(r)
    }

    /// [`resolve_write`] for the passthrough arms, which need only the verdict.
    ///
    /// [`resolve_write`]: Self::resolve_write
    fn guard_write_path(&self, dirfd: i32, pathptr: u64) -> Result<(), Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        self.resolve_write_raw(dirfd, &raw, true).map(|_| ())
    }

    /// The same check for an fd mutator, using the mount policy captured when
    /// the open file description was created; a host fd is not ours to police.
    fn guard_write_fd(&self, fd: i32) -> Result<(), Errno> {
        if !self.is_virtual(fd) {
            return Ok(());
        }
        if self.desc(fd)?.writable {
            Ok(())
        } else {
            Err(Errno::EROFS)
        }
    }

    /// Apply a write guard to a passthrough syscall: a denied guard becomes the
    /// guest's result; a permitted one forwards to the host, first swapping any
    /// virtual fd argument (`fd_arg` names its index) for its host fd.
    fn deny_ro_or_passthrough(
        &self,
        call: &mut SystemCall,
        guard: Result<(), Errno>,
        fd_arg: Option<usize>,
    ) {
        if let Err(e) = guard {
            call.set_result(SyscallResult::Error(e.raw()));
            return;
        }
        if let Some(i) = fd_arg {
            let fd = call.args[i] as i32;
            if self.is_virtual(fd) {
                match self.resolve_fd(fd) {
                    Some(h) => call.args[i] = h as u64,
                    None => {
                        call.set_result(SyscallResult::Error(Errno::EBADF.raw()));
                        return;
                    }
                }
            }
        }
        call.set_result(host_syscall(call));
    }

    fn open(&self, dirfd: i32, pathptr: u64, flags: i32, mode: u32) -> Result<i64, Errno> {
        // open(2) ignores `mode` unless the call creates a file (O_CREAT or
        // O_TMPFILE) — the third argument is variadic and callers routinely
        // pass a stale value on plain reads. openat2, which HostFs resolves
        // through, instead rejects a nonzero mode on a non-creating open
        // with EINVAL, so apply the kernel's open(2) rule before any
        // filesystem sees it.
        let mode = if flags & libc::O_CREAT != 0 || flags & libc::O_TMPFILE == libc::O_TMPFILE {
            mode
        } else {
            0
        };
        // The same treatment for O_PATH companions: open(2) silently masks
        // everything but O_CLOEXEC, O_DIRECTORY, and O_NOFOLLOW out of an
        // O_PATH open, while openat2 rejects the extras with EINVAL — and
        // callers do pass them (Bun's existence probes open O_PATH|O_LARGEFILE).
        let flags = if flags & libc::O_PATH != 0 {
            flags & (libc::O_PATH | libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        } else {
            flags
        };
        let raw = unsafe { read_cstr(pathptr) };
        let abs = self.abs_path(dirfd, &raw)?;
        if let Some(fd) = magic_fd_link(&abs) {
            return self.reopen_fd(fd, flags, mode);
        }
        self.open_abs(&abs, flags, mode)
    }

    fn open_abs(&self, abs: &Path, flags: i32, mode: u32) -> Result<i64, Errno> {
        let follow = flags & libc::O_NOFOLLOW == 0;
        let r = self.ns.resolve(abs, follow)?;
        // Any write intent against a read-only mount is EROFS — but a plain
        // read keeps working, so stdio/sockets are unaffected. Special files
        // are exempt, as they are on a kernel `mount -o ro`: writing a
        // character or block device, FIFO, or socket sends bytes to the
        // object behind the node, not to the filesystem holding it, so
        // `/dev/null` and pty opens keep working in a read-only sandbox.
        // (`O_CREAT` on an existing node is a no-op, and `O_TRUNC` is ignored
        // for devices, so neither drags a special file back under the check.)
        let writes = flags & libc::O_ACCMODE != libc::O_RDONLY
            || flags & (libc::O_CREAT | libc::O_TRUNC) != 0;
        if writes && !r.writable {
            let special = matches!(
                r.fs.stat(&r.rel, follow).map(|st| st.file_type),
                Ok(FileType::CharDevice
                    | FileType::BlockDevice
                    | FileType::Fifo
                    | FileType::Socket)
            );
            if !special {
                return Err(Errno::EROFS);
            }
        }
        // The host open must not carry O_CLOEXEC: the fd it returns becomes the
        // guest-visible reservation, and a reservation with host FD_CLOEXEC
        // would be swept at the next emulated exec (see [`reserve_fd`]). The
        // guest's close-on-exec flag lives in the fd table instead.
        let mut file: Box<dyn File> =
            r.fs.open(&r.rel, OpenFlags(flags & !libc::O_CLOEXEC), Mode(mode))?;
        // Reservation and insert under one hold of the lock; see
        // [`Personality::dup`] for why the pair is atomic.
        let mut fds = self.fds.lock().unwrap();
        // The kernel already handed the open the lowest available number —
        // detach it as the reservation rather than closing it and dup'ing the
        // same number back.
        let guest_fd = match file.detach_reservation() {
            Some(low) => low.into_raw_fd(),
            None => reserve_fd(file.host_fd())?,
        };
        let file: Arc<dyn File> = Arc::from(file);
        let desc = Arc::new(OpenFileDescription {
            file,
            path: r.abs,
            writable: r.writable,
            state: Mutex::new(DescState {
                offset: 0,
                status: flags,
                dirents: None,
            }),
        });
        let cloexec = flags & libc::O_CLOEXEC != 0;
        fds.insert_at(guest_fd, desc, cloexec);
        Ok(guest_fd as i64)
    }

    /// Reopen the object behind one of the guest's descriptors, the way the
    /// kernel services an open of a magic fd link (see [`magic_fd_link`]). A
    /// virtual descriptor reopens by the path it was opened with, so the
    /// namespace and the read-only policy apply as usual. A host descriptor
    /// forwards to the host kernel when the object has no filesystem identity
    /// — a pipe, socket, or tty, which is what process substitution and
    /// `/dev/fd` redirection name. A write reopen of a regular host file is
    /// refused with EROFS: the walker cannot reach the object to consult the
    /// mount policy, and reopening through the magic link is the classic way
    /// to escalate a read-only descriptor to a writable one.
    fn reopen_fd(&self, fd: i32, flags: i32, mode: u32) -> Result<i64, Errno> {
        if flags & libc::O_NOFOLLOW != 0 {
            // The magic link is a symlink; the kernel answers O_NOFOLLOW on
            // it with ELOOP.
            return Err(Errno::ELOOP);
        }
        if self.is_virtual(fd) {
            let path = self.desc(fd)?.path.clone();
            return self.open_abs(&path, flags, mode);
        }
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd, &mut st) } != 0 {
            return Err(Errno::from_io(&std::io::Error::last_os_error()));
        }
        let special = matches!(
            st.st_mode & libc::S_IFMT,
            libc::S_IFIFO | libc::S_IFSOCK | libc::S_IFCHR | libc::S_IFBLK
        );
        let writes = flags & libc::O_ACCMODE != libc::O_RDONLY
            || flags & (libc::O_CREAT | libc::O_TRUNC) != 0;
        if !special && writes {
            return Err(Errno::EROFS);
        }
        let path = format!("/proc/self/fd/{fd}\0");
        check_host(unsafe { libc::open(path.as_ptr() as *const libc::c_char, flags, mode) })
    }

    fn desc(&self, fd: i32) -> Result<Arc<OpenFileDescription>, Errno> {
        Ok(Arc::clone(
            &self.fds.lock().unwrap().get(fd).ok_or(Errno::EBADF)?.desc,
        ))
    }

    fn read(&self, fd: i32, buf: u64, len: usize) -> Result<i64, Errno> {
        let desc = self.desc(fd)?;
        let mut st = desc.state.lock().unwrap();
        let off = st.offset;
        let n = desc.file.pread(guest_mut(buf, len), off)?;
        st.offset += n as u64;
        Ok(n as i64)
    }

    fn write(&self, fd: i32, buf: u64, len: usize) -> Result<i64, Errno> {
        let desc = self.desc(fd)?;
        let mut st = desc.state.lock().unwrap();
        let n = if st.status & libc::O_APPEND != 0 {
            let write = desc.file.append(guest(buf, len))?;
            st.offset = write.offset;
            write.count
        } else {
            let off = st.offset;
            let n = desc.file.pwrite(guest(buf, len), off)?;
            st.offset = off + n as u64;
            n
        };
        Ok(n as i64)
    }

    fn pread(&self, fd: i32, buf: u64, len: usize, off: u64) -> Result<i64, Errno> {
        Ok(self.desc(fd)?.file.pread(guest_mut(buf, len), off)? as i64)
    }

    fn pwrite(&self, fd: i32, buf: u64, len: usize, off: u64) -> Result<i64, Errno> {
        Ok(self.desc(fd)?.file.pwrite(guest(buf, len), off)? as i64)
    }

    fn readv(&self, fd: i32, iov: u64, cnt: usize) -> Result<i64, Errno> {
        let desc = self.desc(fd)?;
        let mut st = desc.state.lock().unwrap();
        let mut total = 0i64;
        for (base, len) in iovecs(iov, cnt) {
            if len == 0 {
                continue;
            }
            let n = desc.file.pread(guest_mut(base, len), st.offset)?;
            st.offset += n as u64;
            total += n as i64;
            if n < len {
                break; // short read ends the gather
            }
        }
        Ok(total)
    }

    fn writev(&self, fd: i32, iov: u64, cnt: usize) -> Result<i64, Errno> {
        let desc = self.desc(fd)?;
        let mut st = desc.state.lock().unwrap();
        let mut total = 0i64;
        for (base, len) in iovecs(iov, cnt) {
            if len == 0 {
                continue;
            }
            let n = if st.status & libc::O_APPEND != 0 {
                let write = desc.file.append(guest(base, len))?;
                st.offset = write.offset;
                write.count
            } else {
                let off = st.offset;
                let n = desc.file.pwrite(guest(base, len), off)?;
                st.offset = off + n as u64;
                n
            };
            total += n as i64;
            if n < len {
                break;
            }
        }
        Ok(total)
    }

    fn lseek(&self, fd: i32, off: i64, whence: i32) -> Result<i64, Errno> {
        let desc = self.desc(fd)?;
        let mut st = desc.state.lock().unwrap();
        let base = match whence {
            libc::SEEK_SET => 0,
            libc::SEEK_CUR => st.offset as i64,
            libc::SEEK_END => desc.file.fstat()?.size as i64,
            _ => return Err(Errno::EINVAL),
        };
        let next = base.checked_add(off).ok_or(Errno::EINVAL)?;
        if next < 0 {
            return Err(Errno::EINVAL);
        }
        st.offset = next as u64;
        st.dirents = None; // a seek restarts any directory walk
        Ok(next)
    }

    fn fstat(&self, fd: i32, buf: u64) -> Result<i64, Errno> {
        let s = self.desc(fd)?.file.fstat()?;
        write_stat(buf, &s);
        Ok(0)
    }

    fn newfstatat(&self, dirfd: i32, pathptr: u64, buf: u64, flags: i32) -> Result<i64, Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        // AT_EMPTY_PATH with an empty path stats the dirfd itself.
        if raw.is_empty() && flags & libc::AT_EMPTY_PATH != 0 && self.is_virtual(dirfd) {
            return self.fstat(dirfd, buf);
        }
        self.fstatat_path(dirfd, &raw, buf, flags & libc::AT_SYMLINK_NOFOLLOW == 0)
    }

    fn fstatat_path(&self, dirfd: i32, raw: &[u8], buf: u64, follow: bool) -> Result<i64, Errno> {
        let abs = self.abs_path(dirfd, raw)?;
        let s = self.ns.stat_path(&abs, follow)?;
        write_stat(buf, &s);
        Ok(0)
    }

    fn statx(&self, dirfd: i32, pathptr: u64, flags: i32, buf: u64) -> Result<i64, Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        let s = if raw.is_empty() && flags & libc::AT_EMPTY_PATH != 0 && self.is_virtual(dirfd) {
            self.desc(dirfd)?.file.fstat()?
        } else {
            let follow = flags & libc::AT_SYMLINK_NOFOLLOW == 0;
            let abs = self.abs_path(dirfd, &raw)?;
            self.ns.stat_path(&abs, follow)?
        };
        write_statx(buf, &s);
        Ok(0)
    }

    fn access(&self, dirfd: i32, pathptr: u64, mode: i32) -> Result<i64, Errno> {
        // Existence and write-against-read-only are enforced; finer mode/uid/gid
        // permission checks are deferred.
        let raw = unsafe { read_cstr(pathptr) };
        let abs = self.abs_path(dirfd, &raw)?;
        let r = self.ns.resolve(&abs, true)?;
        // Confirm the target exists (the resolver may already have stat'd it).
        if r.stat.is_none() {
            r.fs.stat(&r.rel, true)?;
        }
        if mode & libc::W_OK != 0 && !r.writable {
            return Err(Errno::EROFS);
        }
        Ok(0)
    }

    fn readlink(&self, dirfd: i32, pathptr: u64, buf: u64, size: usize) -> Result<i64, Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        // readlink of /proc/self/exe answers with the link's target — the
        // guest's image — before path rewriting would try to readlink the
        // binary itself (EINVAL: it is not a symlink).
        if is_magic_exe(Path::new(OsStr::from_bytes(&raw)))
            && let Some(exe) = self.exe.lock().unwrap().clone()
        {
            let bytes = exe.as_os_str().as_bytes();
            let n = bytes.len().min(size);
            guest_mut(buf, n).copy_from_slice(&bytes[..n]);
            return Ok(n as i64);
        }
        let abs = self.abs_path(dirfd, &raw)?;
        // readlink names the link itself: never follow the final component.
        let r = self.ns.resolve(&abs, false)?;
        let target = r.fs.readlink(&r.rel)?;
        let bytes = target.as_os_str().as_bytes();
        let n = bytes.len().min(size); // readlink truncates, never NUL-terminates
        guest_mut(buf, n).copy_from_slice(&bytes[..n]);
        Ok(n as i64)
    }

    fn mkdir(&self, dirfd: i32, pathptr: u64, mode: u32) -> Result<i64, Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        let abs = self.abs_path(dirfd, &raw)?;
        let r = self.ns.resolve(&abs, true)?;
        if !r.writable {
            return Err(Errno::EROFS);
        }
        r.fs.mkdir(&r.rel, Mode(mode))?;
        Ok(0)
    }

    fn unlinkat(&self, dirfd: i32, pathptr: u64, flags: i32) -> Result<i64, Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        let abs = self.abs_path(dirfd, &raw)?;
        // Removal names the entry itself, symlinks included: do not follow.
        let r = self.ns.resolve(&abs, false)?;
        if !r.writable {
            return Err(Errno::EROFS);
        }
        if flags & libc::AT_REMOVEDIR != 0 {
            r.fs.rmdir(&r.rel)?;
        } else {
            r.fs.unlink(&r.rel)?;
        }
        Ok(0)
    }

    fn rename(&self, odir: i32, optr: u64, ndir: i32, nptr: u64, flags: u32) -> Result<i64, Errno> {
        let (oraw, nraw) = (unsafe { read_cstr(optr) }, unsafe { read_cstr(nptr) });
        let from = self.abs_path(odir, &oraw)?;
        let to = self.abs_path(ndir, &nraw)?;
        let rf = self.ns.resolve(&from, false)?;
        let rt = self.ns.resolve(&to, false)?;
        if !rf.writable || !rt.writable {
            return Err(Errno::EROFS);
        }
        if !Arc::ptr_eq(&rf.fs, &rt.fs) {
            return Err(Errno::EXDEV); // cross-filesystem rename
        }
        rf.fs.rename(&rf.rel, &rt.rel, RenameFlags(flags))?;
        Ok(0)
    }

    fn symlink(&self, targetptr: u64, dirfd: i32, linkptr: u64) -> Result<i64, Errno> {
        let target = unsafe { read_cstr(targetptr) };
        let link = unsafe { read_cstr(linkptr) };
        let abs = self.abs_path(dirfd, &link)?;
        let r = self.ns.resolve(&abs, false)?;
        if !r.writable {
            return Err(Errno::EROFS);
        }
        r.fs.symlink(Path::new(OsStr::from_bytes(&target)), &r.rel)?;
        Ok(0)
    }

    fn link(&self, odir: i32, optr: u64, ndir: i32, nptr: u64) -> Result<i64, Errno> {
        let (oraw, nraw) = (unsafe { read_cstr(optr) }, unsafe { read_cstr(nptr) });
        let old = self.abs_path(odir, &oraw)?;
        let new = self.abs_path(ndir, &nraw)?;
        // The hard link names the existing entry, not its symlink target.
        let ro = self.ns.resolve(&old, false)?;
        let rn = self.ns.resolve(&new, false)?;
        if !rn.writable {
            return Err(Errno::EROFS);
        }
        if !Arc::ptr_eq(&ro.fs, &rn.fs) {
            return Err(Errno::EXDEV);
        }
        ro.fs.link(&ro.rel, &rn.rel)?;
        Ok(0)
    }

    fn truncate(&self, pathptr: u64, len: u64) -> Result<i64, Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        let abs = self.abs_path(libc::AT_FDCWD, &raw)?;
        let r = self.ns.resolve(&abs, true)?;
        if !r.writable {
            return Err(Errno::EROFS);
        }
        let file = r.fs.open(&r.rel, OpenFlags(libc::O_WRONLY), Mode(0))?;
        file.ftruncate(len)?;
        Ok(0)
    }

    fn ftruncate(&self, fd: i32, len: u64) -> Result<i64, Errno> {
        self.desc(fd)?.file.ftruncate(len)?;
        Ok(0)
    }

    fn fsync(&self, fd: i32) -> Result<i64, Errno> {
        self.desc(fd)?.file.fsync()?;
        Ok(0)
    }

    fn statfs(&self, pathptr: u64, buf: u64) -> Result<i64, Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        let abs = self.abs_path(libc::AT_FDCWD, &raw)?;
        let r = self.ns.resolve(&abs, true)?;
        write_statfs(buf, &r.fs.statfs(&r.rel)?);
        Ok(0)
    }

    fn fstatfs(&self, fd: i32, buf: u64) -> Result<i64, Errno> {
        write_statfs(buf, &self.desc(fd)?.file.fstatfs()?);
        Ok(0)
    }

    fn getdents64(&self, fd: i32, buf: u64, count: usize) -> Result<i64, Errno> {
        let desc = self.desc(fd)?;
        let mut st = desc.state.lock().unwrap();
        if st.dirents.is_none() {
            // Synthesize `.` and `..` up front (the kernel reports them but a
            // Vfs does not); `..` borrows this directory's inode for Phase 1.
            let self_ino = desc.file.fstat().map(|s| s.ino).unwrap_or(0);
            let mut entries = vec![
                DirEntry {
                    ino: self_ino,
                    file_type: FileType::Directory,
                    name: OsString::from("."),
                },
                DirEntry {
                    ino: self_ino,
                    file_type: FileType::Directory,
                    name: OsString::from(".."),
                },
            ];
            entries.extend(desc.file.getdents()?);
            st.offset = 0;
            st.dirents = Some(entries);
        }
        let entries = st.dirents.as_ref().unwrap();
        let (bytes, next) = encode_dirents(entries, st.offset as usize, count);
        guest_mut(buf, bytes.len()).copy_from_slice(&bytes);
        st.offset = next as u64;
        Ok(bytes.len() as i64)
    }

    /// `dup`/`F_DUPFD`: the guest fd is itself the reserved kernel number, so
    /// duplicating it hands the kernel the numbering (lowest free at or above
    /// `minfd`, exactly the contract) and keeps the new number reserved.
    ///
    /// Here and in [`Personality::dup2`], the kernel call runs under the same
    /// hold of the table lock as the map update. The pair is one compound
    /// state change: another thread's table op (or a fork snapshot, which
    /// takes this lock — see `SystemCalls::lock_for_fork`) landing between
    /// the halves would see a number the kernel has already reassigned still
    /// tracked, or an allocated number untracked — and a stale close-on-exec
    /// entry, closed at the next emulated exec, takes down whatever innocent
    /// descriptor has since been handed that number.
    fn dup(&self, fd: i32, minfd: i32, cloexec: bool) -> Result<i64, Errno> {
        let mut fds = self.fds.lock().unwrap();
        let desc = Arc::clone(&fds.get(fd).ok_or(Errno::EBADF)?.desc);
        let newfd = check_host(unsafe { libc::fcntl(fd, libc::F_DUPFD, minfd) })? as i32;
        fds.insert_at(newfd, desc, cloexec);
        Ok(newfd as i64)
    }

    fn dup2(&self, oldfd: i32, newfd: i32, flags: i32) -> Result<i64, Errno> {
        let mut fds = self.fds.lock().unwrap();
        let desc = Arc::clone(&fds.get(oldfd).ok_or(Errno::EBADF)?.desc);
        if oldfd == newfd {
            return Ok(newfd as i64); // dup2 no-op; dup3 would EINVAL, but is rare
        }
        // Claim `newfd` in the kernel as an alias of the same open file.
        // `dup2` atomically closes whatever occupied the number — a previous
        // reservation or a plain host fd (e.g. dup2(fd, 1) redirecting
        // stdout) — exactly the dup2 the guest asked for.
        check_host(unsafe { libc::dup2(oldfd, newfd) })?;
        let cloexec = flags & libc::O_CLOEXEC != 0;
        fds.map.insert(newfd, Fd { desc, cloexec });
        Ok(newfd as i64)
    }

    fn fcntl(&self, fd: i32, cmd: i32, arg: u64) -> Result<i64, Errno> {
        match cmd {
            libc::F_DUPFD => self.dup(fd, arg as i32, false),
            libc::F_DUPFD_CLOEXEC => self.dup(fd, arg as i32, true),
            libc::F_GETFD => Ok(
                if self
                    .fds
                    .lock()
                    .unwrap()
                    .get(fd)
                    .ok_or(Errno::EBADF)?
                    .cloexec
                {
                    libc::FD_CLOEXEC as i64
                } else {
                    0
                },
            ),
            libc::F_SETFD => {
                self.fds
                    .lock()
                    .unwrap()
                    .map
                    .get_mut(&fd)
                    .ok_or(Errno::EBADF)?
                    .cloexec = arg as i32 & libc::FD_CLOEXEC != 0;
                Ok(0)
            }
            libc::F_GETFL => Ok(self.desc(fd)?.state.lock().unwrap().status as i64),
            libc::F_SETFL => {
                // Only the changeable status bits; access mode and creation
                // flags are fixed at open time.
                const SETTABLE: i32 =
                    libc::O_APPEND | libc::O_NONBLOCK | libc::O_DIRECT | libc::O_NOATIME;
                let desc = self.desc(fd)?;
                let mut st = desc.state.lock().unwrap();
                st.status = (st.status & !SETTABLE) | (arg as i32 & SETTABLE);
                // The flags must reach the real open file description too:
                // O_NONBLOCK decides whether a read on a tty or FIFO served
                // through the backing fd blocks or reports EAGAIN, and an
                // event loop lives or dies by that.
                if let Some(host) = desc.file.host_fd() {
                    unsafe { libc::fcntl(host, libc::F_SETFL, st.status) };
                }
                Ok(0)
            }
            // Byte-range record locks — SQLite's locking protocol (cargo's
            // global-cache database is the first customer). Forwarded on the
            // guest number itself: the reservation is an `F_DUPFD` alias of
            // the backing open file description, so the kernel locks the real
            // file for both families — classic locks key on (process, inode),
            // OFD locks on the description, and the alias shares both. The
            // `struct flock` pointer is guest memory, which the kernel
            // validates itself. A synthetic file reserves its number on
            // `/dev/null`, where every synthetic file's locks would alias one
            // inode and conflict with each other; those stay refused.
            libc::F_SETLK
            | libc::F_SETLKW
            | libc::F_GETLK
            | libc::F_OFD_SETLK
            | libc::F_OFD_SETLKW
            | libc::F_OFD_GETLK => {
                if self.desc(fd)?.file.host_fd().is_none() {
                    return Err(Errno::EINVAL);
                }
                Ok(check_host(unsafe { libc::fcntl(fd, cmd, arg) })? as i64)
            }
            _ => Err(Errno::EINVAL),
        }
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

    fn chdir(&self, pathptr: u64) -> Result<i64, Errno> {
        let raw = unsafe { read_cstr(pathptr) };
        let abs = self.abs_path(libc::AT_FDCWD, &raw)?;
        let r = self.ns.resolve(&abs, true)?;
        if r.fs.stat(&r.rel, true)?.file_type != FileType::Directory {
            return Err(Errno::ENOTDIR);
        }
        *self.cwd.lock().unwrap() = r.abs; // store the canonical, symlink-resolved path
        Ok(0)
    }

    fn fchdir(&self, fd: i32) -> Result<i64, Errno> {
        let path = self.desc(fd)?.path.clone();
        *self.cwd.lock().unwrap() = path;
        Ok(0)
    }

    fn chmod_path(&self, dirfd: i32, pathptr: u64, mode: u32, flags: i32) -> Result<i64, Errno> {
        const ALLOWED: i32 = libc::AT_EMPTY_PATH | libc::AT_SYMLINK_NOFOLLOW;
        if flags & !ALLOWED != 0 {
            return Err(Errno::EINVAL);
        }
        if pathptr == 0 {
            return Err(Errno(libc::EFAULT));
        }
        let raw = unsafe { read_cstr(pathptr) };
        if raw.is_empty() {
            if flags & libc::AT_EMPTY_PATH == 0 {
                return Err(Errno::ENOENT);
            }
            if self.is_virtual(dirfd) {
                self.guard_write_fd(dirfd)?;
                self.desc(dirfd)?.file.fchmodat_empty(Mode(mode))?;
                return Ok(0);
            }
        }
        let follow = flags & libc::AT_SYMLINK_NOFOLLOW == 0;
        let r = self.resolve_write_raw(dirfd, &raw, follow)?;
        r.fs.chmod(&r.rel, follow, Mode(mode))?;
        Ok(0)
    }

    fn fchmod(&self, fd: i32, mode: u32) -> Result<i64, Errno> {
        self.guard_write_fd(fd)?;
        self.desc(fd)?.file.fchmod(Mode(mode))?;
        Ok(0)
    }

    fn chown_path(
        &self,
        dirfd: i32,
        pathptr: u64,
        uid: u32,
        gid: u32,
        flags: i32,
    ) -> Result<i64, Errno> {
        const ALLOWED: i32 = libc::AT_EMPTY_PATH | libc::AT_SYMLINK_NOFOLLOW;
        if flags & !ALLOWED != 0 {
            return Err(Errno::EINVAL);
        }
        if pathptr == 0 {
            return Err(Errno(libc::EFAULT));
        }
        let raw = unsafe { read_cstr(pathptr) };
        // fchownat(fd, "", AT_EMPTY_PATH, …) targets the descriptor itself.
        if flags & libc::AT_EMPTY_PATH != 0 && raw.is_empty() && self.is_virtual(dirfd) {
            self.guard_write_fd(dirfd)?;
            self.desc(dirfd)?.file.fchownat_empty(uid, gid)?;
            return Ok(0);
        }
        if raw.is_empty() && flags & libc::AT_EMPTY_PATH == 0 {
            return Err(Errno::ENOENT);
        }
        let follow = flags & libc::AT_SYMLINK_NOFOLLOW == 0;
        let r = self.resolve_write_raw(dirfd, &raw, follow)?;
        r.fs.chown(&r.rel, follow, uid, gid)?;
        Ok(0)
    }

    fn fchown(&self, fd: i32, uid: u32, gid: u32) -> Result<i64, Errno> {
        self.guard_write_fd(fd)?;
        self.desc(fd)?.file.fchown(uid, gid)?;
        Ok(0)
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

    fn utimes(&self, dirfd: i32, pathptr: u64, tvptr: u64) -> Result<i64, Errno> {
        let times = if tvptr == 0 {
            None
        } else {
            let tv = unsafe { ptr::read(tvptr as *const [libc::timeval; 2]) };
            if tv.iter().any(|t| !(0..1_000_000).contains(&t.tv_usec)) {
                return Err(Errno::EINVAL);
            }
            Some(tv.map(|t| Timespec {
                sec: t.tv_sec,
                nsec: t.tv_usec * 1000,
            }))
        };
        if pathptr == 0 {
            self.guard_write_fd(dirfd)?;
            self.desc(dirfd)?.file.futimens(times)?;
            return Ok(0);
        }
        self.utimens_path(dirfd, pathptr, times, true)
    }

    fn utimensat(&self, dirfd: i32, pathptr: u64, tsptr: u64, flags: i32) -> Result<i64, Errno> {
        const ALLOWED: i32 = libc::AT_EMPTY_PATH | libc::AT_SYMLINK_NOFOLLOW;
        if flags & !ALLOWED != 0 || pathptr == 0 && flags != 0 {
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
            // utimensat(fd, NULL, …): the fd names the file.
            self.guard_write_fd(dirfd)?;
            self.desc(dirfd)?.file.futimens(times)?;
            return Ok(0);
        }
        let raw = unsafe { read_cstr(pathptr) };
        if raw.is_empty() {
            if flags & libc::AT_EMPTY_PATH == 0 {
                return Err(Errno::ENOENT);
            }
            if self.is_virtual(dirfd) {
                self.guard_write_fd(dirfd)?;
                self.desc(dirfd)?.file.utimensat_empty(times)?;
                return Ok(0);
            }
        }
        let follow = flags & libc::AT_SYMLINK_NOFOLLOW == 0;
        let r = self.resolve_write_raw(dirfd, &raw, follow)?;
        r.fs.utimens(&r.rel, follow, times)?;
        Ok(0)
    }

    fn utimens_path(
        &self,
        dirfd: i32,
        pathptr: u64,
        times: Option<[Timespec; 2]>,
        follow: bool,
    ) -> Result<i64, Errno> {
        let r = self.resolve_write(dirfd, pathptr, follow)?;
        r.fs.utimens(&r.rel, follow, times)?;
        Ok(0)
    }
}

/// Write a `SyscallResult` into `call` from a `Result<i64, Errno>`.
fn finish(call: &mut SystemCall, r: Result<i64, Errno>) {
    call.set_result(match r {
        Ok(v) => SyscallResult::Ok(v),
        Err(e) => SyscallResult::Error(e.raw()),
    });
}

/// Map a libc `-1`/value return into a `Result`, reading errno on failure.
fn check_host(ret: libc::c_int) -> Result<i64, Errno> {
    if ret < 0 {
        Err(Errno::from_io(&std::io::Error::last_os_error()))
    } else {
        Ok(ret as i64)
    }
}

/// A guest buffer as a shared byte slice. Chimera and the guest share one
/// address space, so a guest pointer is a host pointer.
fn guest(ptr: u64, len: usize) -> &'static [u8] {
    if len == 0 {
        return &[];
    }
    unsafe { slice::from_raw_parts(ptr as *const u8, len) }
}

/// A guest buffer as a mutable byte slice.
fn guest_mut(ptr: u64, len: usize) -> &'static mut [u8] {
    if len == 0 {
        return &mut [];
    }
    unsafe { slice::from_raw_parts_mut(ptr as *mut u8, len) }
}

/// Read a NUL-terminated guest string (without the terminator).
unsafe fn read_cstr(ptr: u64) -> Vec<u8> {
    if ptr == 0 {
        return Vec::new();
    }
    // The guest shares the address space, so this is an in-process C string:
    // libc's strlen + one exact-size copy.
    unsafe { CStr::from_ptr(ptr as *const libc::c_char) }
        .to_bytes()
        .to_vec()
}

/// Iterate a guest `struct iovec[]` as `(base, len)` pairs.
fn iovecs(iov: u64, cnt: usize) -> Vec<(u64, usize)> {
    let mut out = Vec::with_capacity(cnt);
    for i in 0..cnt {
        let v = unsafe { ptr::read((iov as *const libc::iovec).add(i)) };
        out.push((v.iov_base as u64, v.iov_len));
    }
    out
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
    use crate::sys::linux::{HostFs, MountFlags};

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

        assert_eq!(personality.guard_write_fd(fd), Ok(()));
    }

    #[test]
    fn fd_write_guard_survives_unlink() {
        let scratch = Scratch::new();
        let personality = scratch.personality(MountFlags::NONE);
        std::fs::write(scratch.0.join("target"), b"x").unwrap();

        let fd = scratch.open(&personality, c"/target", libc::O_RDONLY);
        std::fs::remove_file(scratch.0.join("target")).unwrap();

        assert_eq!(personality.guard_write_fd(fd), Ok(()));
    }

    #[test]
    fn fd_write_guard_denies_read_only_mount() {
        let scratch = Scratch::new();
        std::fs::write(scratch.0.join("target"), b"x").unwrap();
        let personality = scratch.personality(MountFlags::RDONLY);

        let fd = scratch.open(&personality, c"/target", libc::O_RDONLY);

        assert_eq!(personality.guard_write_fd(fd), Err(Errno::EROFS));
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

    #[test]
    fn fd_write_guard_ignores_host_fds() {
        let scratch = Scratch::new();
        let personality = scratch.personality(MountFlags::RDONLY);

        assert_eq!(personality.guard_write_fd(libc::STDIN_FILENO), Ok(()));
    }
}
