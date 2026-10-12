//! The [`Personality`]: the [`SystemCalls`] handler that turns a guest's
//! filesystem syscalls into calls on a [`Vfs`], and owns the descriptor table,
//! current directory, and guest-ABI encoding that the `Vfs`/`File` traits are
//! kept free of.
//!
//! It joins a guest path against the cwd or a dirfd, hands it to the
//! [`Namespace`] for confining, symlink-resolving, mount-crossing resolution,
//! and dispatches the resulting `(Vfs, rel)` — denying writes to a read-only
//! mount with `EROFS`. Descriptors a filesystem hands out are *virtual* —
//! each one holds a reserved kernel descriptor number (see [`reserve_fd`]) so
//! the guest sees the low, dense, lowest-available numbers POSIX promises —
//! and any syscall on a descriptor not in the table, or any syscall not
//! handled here, is forwarded to the host unchanged. `execve` targets resolve
//! through the namespace too ([`SystemCalls::resolve_exec`]), so a confined
//! guest cannot exec outside its root.
//!
//! This module is the kernel-neutral half: the table, the resolver, and one
//! method per operation, speaking symbolic [`Stat`]s and raw guest buffers.
//! The guest's own ABI — which syscall numbers mean what, how `struct stat`
//! and a directory entry are laid out, which `fcntl` command is which — is
//! the business of one submodule per host ([`linux`], [`darwin`]), which
//! implements [`SystemCalls`] by decoding the call and encoding the answer.

use std::{
    borrow::Cow,
    collections::HashMap,
    ffi::{CStr, CString, OsStr, OsString},
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
    delta::XATTR_NAMESPACE,
    namespace::{Namespace, Resolved, normalize},
    vfs::{DirEntry, Errno, File, FileType, Mode, OpenFlags, RenameFlags, Stat, Timespec},
};

#[cfg(target_os = "macos")]
mod darwin;
#[cfg(target_os = "linux")]
mod linux;

/// The host's decoder, which also supplies the few per-kernel facts the
/// shared code branches on.
#[cfg(target_os = "macos")]
use darwin as os;
#[cfg(target_os = "linux")]
use linux as os;

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
    /// `execPath`) the runtime to exec instead of itself. Recorded at every
    /// image install ([`SystemCalls::on_execve`]), the initial one included.
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
/// The descriptor named by a magic fd link — `/dev/fd/N`, or on Linux
/// `/proc/self/fd/N` and `/proc/thread-self/fd/N` — or `None` for any other
/// path. The kernel resolves these links to the open file itself, not through
/// the filesystem (their readlink target, `pipe:[…]` or `socket:[…]`, is not a
/// path), so the namespace walker cannot follow them; [`Personality::open`]
/// services them as fd reopens instead.
fn magic_fd_link(abs: &Path) -> Option<i32> {
    let s = abs.as_os_str().as_bytes();
    // The /dev/std* symlinks point at the fd links and would hit the magic
    // link one hop later; naming their well-known descriptors here keeps the
    // shell idioms (`< /dev/stdin`, `> /dev/stderr`) working.
    match s {
        b"/dev/stdin" => return Some(0),
        b"/dev/stdout" => return Some(1),
        b"/dev/stderr" => return Some(2),
        _ => {}
    }
    let n = os::FD_LINK_PREFIXES
        .iter()
        .find_map(|prefix| s.strip_prefix(*prefix))?;
    std::str::from_utf8(n).ok()?.parse().ok()
}

impl Personality {
    /// [`SystemCalls::resolve_fd`]: the host descriptor behind one of ours.
    fn host_fd_of(&self, guest_fd: i32) -> Option<RawFd> {
        // A descriptor not in the table is already a host fd; leave it.
        self.fds
            .lock()
            .unwrap()
            .get(guest_fd)
            .and_then(|fd| fd.desc.file.host_fd())
    }

    /// [`SystemCalls::on_execve`]: an image install committed.
    fn exec_committed(&self, path: &Path) {
        // The link target of the executed image follows the exec.
        // Canonicalized so the guest sees the absolute, symlink-resolved path
        // the kernel would store.
        let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        *self.exe.lock().unwrap() = Some(path);
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

    /// [`SystemCalls::lock_for_fork`]: every lock this handler owns.
    fn fork_hold(&self) -> Box<dyn ForkHold + '_> {
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

    /// [`SystemCalls::resolve_exec`]: the host path behind an exec target.
    fn exec_target(&self, dirfd: i32, path: &[u8], flags: i32) -> Result<PathBuf, Errno> {
        // Resolve the target through the namespace, then hand the loader the
        // host path that backs it — so a confined guest cannot exec outside its
        // root, and a synthetic filesystem (no host path) is reported as EACCES.
        let abs: Cow<Path> = if os::exec_names_fd(path, flags) {
            // execveat(fd, "", AT_EMPTY_PATH): the fd names the file.
            Cow::Owned(self.desc(dirfd)?.path.clone())
        } else {
            self.abs_path(dirfd, path)?
        };
        let r = self.ns.resolve(&abs, true)?;
        r.fs.host_path(&r.rel).ok_or(Errno::EACCES)
    }
}

/// What an `AT_*` attribute mutator's `(dirfd, pathptr, flags)` names, as
/// decoded by [`Personality::at_target`].
enum AtTarget {
    /// `AT_EMPTY_PATH` with an empty path on one of our descriptors: the
    /// operation targets the open description, already write-guarded.
    Fd(Arc<OpenFileDescription>),
    /// A path resolved on a writable mount; `follow` is the final-symlink
    /// disposition the operation must carry through to its `Vfs` call.
    Path {
        resolved: Box<Resolved>,
        follow: bool,
    },
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
        // The executed-image link names the running image, which under an
        // emulated exec is the guest's binary, not Chimera's: leaking the
        // runtime's own path here hands a self-respawning guest the wrong
        // program.
        if os::is_magic_exe(&joined)
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

    /// Decode an `AT_*` attribute mutator's `(dirfd, pathptr, flags)` into its
    /// target. The host's decoder validates `flags` and says whether they ask
    /// for the final symlink itself and whether an empty path names the
    /// dirfd (Linux's `AT_EMPTY_PATH`); a NULL path is `EFAULT`, an empty path
    /// is either that fd form (on a virtual dirfd) or `ENOENT`. The
    /// `utimensat` NULL-pointer form (the fd names the file) is not an
    /// `AT_EMPTY_PATH` case and is handled by its caller before this.
    fn at_target(&self, dirfd: i32, pathptr: u64, flags: i32) -> Result<AtTarget, Errno> {
        let (follow, empty_path) = os::decode_at_flags(flags)?;
        if pathptr == 0 {
            return Err(Errno(libc::EFAULT));
        }
        let raw = unsafe { read_cstr(pathptr) };
        if raw.is_empty() {
            if !empty_path {
                return Err(Errno::ENOENT);
            }
            if self.is_virtual(dirfd) {
                return Ok(AtTarget::Fd(self.writable_desc(dirfd)?));
            }
            // AT_EMPTY_PATH with an empty path on a host dirfd falls through to
            // resolution, which answers EBADF for the unknown dirfd.
        }
        let resolved = Box::new(self.resolve_write_raw(dirfd, &raw, follow)?);
        Ok(AtTarget::Path { resolved, follow })
    }

    /// The mutator target behind an fd, rejected with `EROFS` if the mount was
    /// read-only when it was opened — the fd-form counterpart of
    /// [`resolve_write`], returning the descriptor so the caller reuses the
    /// one lookup. Host fds never reach here: the fd-form dispatch arms are
    /// gated on [`is_virtual`], since a host fd is not ours to police, so an
    /// fd absent from the table is `EBADF`.
    ///
    /// [`resolve_write`]: Self::resolve_write
    /// [`is_virtual`]: Self::is_virtual
    fn writable_desc(&self, fd: i32) -> Result<Arc<OpenFileDescription>, Errno> {
        let desc = self.desc(fd)?;
        if desc.writable {
            Ok(desc)
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
        let (flags, mode) = os::sanitize_open(flags, mode);
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
        os::reopen_host_fd(fd, flags, mode)
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
            // SEEK_DATA/SEEK_HOLE ask about the file's extents, which only
            // the kernel knows: ask it through the reservation, an alias of
            // the backing open file, and adopt the absolute offset it names.
            _ => {
                let next = unsafe { libc::lseek(fd, off, whence) };
                if next < 0 {
                    return Err(Errno::from_io(&std::io::Error::last_os_error()));
                }
                st.offset = next as u64;
                st.dirents = None;
                return Ok(next);
            }
        };
        let next = base.checked_add(off).ok_or(Errno::EINVAL)?;
        if next < 0 {
            return Err(Errno::EINVAL);
        }
        st.offset = next as u64;
        st.dirents = None; // a seek restarts any directory walk
        Ok(next)
    }

    /// `fstat`: the open file's identity.
    fn stat_fd(&self, fd: i32) -> Result<Stat, Errno> {
        self.desc(fd)?.file.fstat()
    }

    /// `stat`/`lstat`/`fstatat`: a path's identity, the final symlink
    /// followed or named per `follow`.
    fn stat_at(&self, dirfd: i32, raw: &[u8], follow: bool) -> Result<Stat, Errno> {
        let abs = self.abs_path(dirfd, raw)?;
        self.ns.stat_path(&abs, follow)
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
        // readlink of the executed-image link answers with the link's target
        // — the guest's image — before path rewriting would try to readlink
        // the binary itself (EINVAL: it is not a symlink).
        if os::is_magic_exe(Path::new(OsStr::from_bytes(&raw)))
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

    /// Serve a directory read on `fd`: snapshot the entries on the first read
    /// (with `.` and `..` synthesized up front — the kernel reports them but a
    /// `Vfs` does not; `..` borrows this directory's inode), then let `encode`
    /// lay out the records from the fd's cursor into at most `count` bytes,
    /// returning the bytes, the index to resume from, and the value the
    /// syscall answers with. The cursor lives in the fd table, so a `dup`
    /// shares the walk and `lseek` restarts it.
    fn read_dirents(
        &self,
        fd: i32,
        buf: u64,
        count: usize,
        encode: impl FnOnce(&[DirEntry], usize, usize) -> Result<(Vec<u8>, usize, i64), Errno>,
    ) -> Result<i64, Errno> {
        let desc = self.desc(fd)?;
        let mut st = desc.state.lock().unwrap();
        if st.dirents.is_none() {
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
        let (bytes, next, result) = encode(entries, st.offset as usize, count)?;
        guest_mut(buf, bytes.len()).copy_from_slice(&bytes);
        st.offset = next as u64;
        Ok(result)
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

    /// `F_GETFD`: the descriptor's close-on-exec flag, kept in the table.
    fn cloexec(&self, fd: i32) -> Result<bool, Errno> {
        Ok(self
            .fds
            .lock()
            .unwrap()
            .get(fd)
            .ok_or(Errno::EBADF)?
            .cloexec)
    }

    /// `F_SETFD`: set the descriptor's close-on-exec flag.
    fn set_cloexec(&self, fd: i32, cloexec: bool) -> Result<i64, Errno> {
        self.fds
            .lock()
            .unwrap()
            .map
            .get_mut(&fd)
            .ok_or(Errno::EBADF)?
            .cloexec = cloexec;
        Ok(0)
    }

    /// `F_GETFL`: the open file description's status flags.
    fn status_flags(&self, fd: i32) -> Result<i64, Errno> {
        Ok(self.desc(fd)?.state.lock().unwrap().status as i64)
    }

    /// `F_SETFL`: replace the changeable status bits (`settable`); access mode
    /// and creation flags are fixed at open time.
    fn set_status_flags(&self, fd: i32, arg: i32, settable: i32) -> Result<i64, Errno> {
        let desc = self.desc(fd)?;
        let mut st = desc.state.lock().unwrap();
        st.status = (st.status & !settable) | (arg & settable);
        // The flags must reach the real open file description too:
        // O_NONBLOCK decides whether a read on a tty or FIFO served
        // through the backing fd blocks or reports EAGAIN, and an
        // event loop lives or dies by that.
        if let Some(host) = desc.file.host_fd() {
            unsafe { libc::fcntl(host, libc::F_SETFL, st.status) };
        }
        Ok(0)
    }

    /// Byte-range record locks — SQLite's locking protocol (cargo's global
    /// cache database is the first customer). Forwarded on the guest number
    /// itself: the reservation is an `F_DUPFD` alias of the backing open
    /// file description, so the kernel locks the real file for both
    /// families — classic locks key on (process, inode), OFD locks on the
    /// description, and the alias shares both. The `struct flock` pointer is
    /// guest memory, which the kernel validates itself. A synthetic file
    /// reserves its number on `/dev/null`, where every synthetic file's locks
    /// would alias one inode and conflict with each other; those stay
    /// refused.
    fn lock(&self, fd: i32, cmd: i32, arg: u64) -> Result<i64, Errno> {
        if self.desc(fd)?.file.host_fd().is_none() {
            return Err(Errno::EINVAL);
        }
        Ok(check_host(unsafe { libc::fcntl(fd, cmd, arg) })? as i64)
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
        match self.at_target(dirfd, pathptr, flags)? {
            AtTarget::Fd(desc) => desc.file.fchmodat_empty(Mode(mode))?,
            AtTarget::Path { resolved, follow } => {
                resolved.fs.chmod(&resolved.rel, follow, Mode(mode))?
            }
        }
        Ok(0)
    }

    fn fchmod(&self, fd: i32, mode: u32) -> Result<i64, Errno> {
        self.writable_desc(fd)?.file.fchmod(Mode(mode))?;
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
        match self.at_target(dirfd, pathptr, flags)? {
            AtTarget::Fd(desc) => desc.file.fchownat_empty(uid, gid)?,
            AtTarget::Path { resolved, follow } => {
                resolved.fs.chown(&resolved.rel, follow, uid, gid)?
            }
        }
        Ok(0)
    }

    fn fchown(&self, fd: i32, uid: u32, gid: u32) -> Result<i64, Errno> {
        self.writable_desc(fd)?.file.fchown(uid, gid)?;
        Ok(0)
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
                nsec: os::usec_to_nsec(t.tv_usec),
            }))
        };
        // The descriptor form (`futimesat(fd, NULL, …)`) is taken only when the
        // dirfd is a real descriptor, matching the kernel's `filename == NULL
        // && dfd != AT_FDCWD` rule; a NULL path against `AT_FDCWD` is a bad
        // address, so it falls through to the path form and answers `EFAULT`.
        if pathptr == 0 && dirfd != libc::AT_FDCWD {
            self.writable_desc(dirfd)?.file.futimens(times)?;
            return Ok(0);
        }
        self.utimens_path(dirfd, pathptr, times, true)
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

    fn mknod(&self, dirfd: i32, pathptr: u64, mode: u32, dev: u64) -> Result<i64, Errno> {
        // mknod names the entry it creates; a trailing symlink is EEXIST, so
        // never follow the final component.
        let r = self.resolve_write(dirfd, pathptr, false)?;
        r.fs.mknod(&r.rel, Mode(mode), dev)?;
        Ok(0)
    }

    fn setxattr_path(
        &self,
        pathptr: u64,
        nameptr: u64,
        valptr: u64,
        len: usize,
        flags: i32,
        follow: bool,
    ) -> Result<i64, Errno> {
        if marker_name(nameptr) {
            return Err(Errno(libc::EPERM));
        }
        let r = self.resolve_write(libc::AT_FDCWD, pathptr, follow)?;
        let name = unsafe { read_cstr(nameptr) };
        r.fs.setxattr(
            &r.rel,
            follow,
            OsStr::from_bytes(&name),
            guest(valptr, len),
            flags,
        )?;
        Ok(0)
    }

    fn fsetxattr(
        &self,
        fd: i32,
        nameptr: u64,
        valptr: u64,
        len: usize,
        flags: i32,
    ) -> Result<i64, Errno> {
        if marker_name(nameptr) {
            return Err(Errno(libc::EPERM));
        }
        let desc = self.writable_desc(fd)?;
        let name = unsafe { read_cstr(nameptr) };
        desc.file
            .fsetxattr(OsStr::from_bytes(&name), guest(valptr, len), flags)?;
        Ok(0)
    }

    fn removexattr_path(&self, pathptr: u64, nameptr: u64, follow: bool) -> Result<i64, Errno> {
        if marker_name(nameptr) {
            return Err(Errno(libc::EPERM));
        }
        let r = self.resolve_write(libc::AT_FDCWD, pathptr, follow)?;
        let name = unsafe { read_cstr(nameptr) };
        r.fs.removexattr(&r.rel, follow, OsStr::from_bytes(&name))?;
        Ok(0)
    }

    fn fremovexattr(&self, fd: i32, nameptr: u64) -> Result<i64, Errno> {
        if marker_name(nameptr) {
            return Err(Errno(libc::EPERM));
        }
        let desc = self.writable_desc(fd)?;
        let name = unsafe { read_cstr(nameptr) };
        desc.file.fremovexattr(OsStr::from_bytes(&name))?;
        Ok(0)
    }

    /// Forward a path-keyed syscall to the host against the backing host path
    /// the namespace resolves `call.args[index]` to — for the reads the
    /// Personality does not model but whose path must still resolve through
    /// the namespace (`getxattr`, `statfs`, `pathconf`, …): the file may live
    /// only in an overlay's delta, and the host kernel then serves the
    /// backing path.
    fn forward_at_host_path(&self, call: &mut SystemCall, index: usize, follow: bool) {
        let host = (|| -> Result<CString, Errno> {
            let raw = unsafe { read_cstr(call.args[index]) };
            let abs = self.abs_path(libc::AT_FDCWD, &raw)?;
            let r = self.ns.resolve(&abs, follow)?;
            let host = r.fs.host_path(&r.rel).ok_or(Errno::ENOENT)?;
            CString::new(host.as_os_str().as_bytes()).map_err(|_| Errno::EINVAL)
        })();
        self.forward_rewritten(call, index, host);
    }

    /// Forward `call` with `args[index]` swapped for `host` while the kernel
    /// sees it, restoring the guest's own argument afterwards so observers
    /// read the call as the guest made it. A failed resolution is the call's
    /// result.
    fn forward_rewritten(&self, call: &mut SystemCall, index: usize, host: Result<CString, Errno>) {
        let result = match host {
            Ok(host) => {
                let saved = call.args[index];
                call.args[index] = host.as_ptr() as u64;
                let result = host_syscall(call);
                call.args[index] = saved;
                result
            }
            Err(e) => SyscallResult::Error(e.raw()),
        };
        call.set_result(result);
    }
}

/// `true` when the guest names an attribute inside the overlay's bookkeeping
/// namespace, which it may neither read nor forge.
fn marker_name(nameptr: u64) -> bool {
    unsafe { read_cstr(nameptr) }.starts_with(XATTR_NAMESPACE)
}

/// Drop bookkeeping names from a successful `listxattr`-family result: the
/// guest buffer holds NUL-terminated names, compacted in place. A pure size
/// query (`size == 0`) passes through and may over-report by the hidden
/// names — callers treat it as an allocation hint, not a promise.
fn strip_marker_names(result: SyscallResult, buf: u64, size: usize) -> SyscallResult {
    let SyscallResult::Ok(len) = result else {
        return result;
    };
    if size == 0 || len <= 0 {
        return result;
    }
    let names = guest_mut(buf, (len as usize).min(size));
    let mut out = 0;
    let mut i = 0;
    while i < names.len() {
        let end = match names[i..].iter().position(|&b| b == 0) {
            Some(p) => i + p + 1,
            None => names.len(),
        };
        if !names[i..end].starts_with(XATTR_NAMESPACE) {
            names.copy_within(i..end, out);
            out += end - i;
        }
        i = end;
    }
    SyscallResult::Ok(out as i64)
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
