//! The userspace-filesystem seam: the [`Vfs`] (one filesystem instance, a
//! superblock) and [`File`] (one open file or directory handle) traits a
//! filesystem implements, plus the symbolic types they speak.
//!
//! This sits *behind* a [`crate::SystemCalls`] handler, not in place of one. A
//! filesystem author never touches a guest register, decodes a `struct statx`,
//! or scatters an iovec: that Linux-ABI grind — pulling paths and structs out
//! of guest memory, translating open-flag bits and errno numbers, laying out
//! `struct stat`/`linux_dirent64` — belongs to the Personality (the
//! `SystemCalls` impl) and the mount-table resolver that wrap these traits.
//! Path resolution (`AT_FDCWD`/dirfd → absolute, `..`, symlink following, root
//! confinement, crossing mountpoints, read-only mounts) belongs to the
//! resolver too, so the `*at` dirfd syscall forms are gone before a `Vfs` ever
//! runs.
//!
//! The model mirrors Linux: a `Vfs` is a filesystem *instance* (a superblock),
//! held as `Arc<dyn Vfs>` and mounted into a namespace; a guest fd backed by a
//! filesystem holds an `Arc<dyn File>` plus its read/write offset (the offset
//! lives in the fd table, not the handle, so `dup` shares it and `pread`/
//! `pwrite` stay positional unless the handle itself owns semantics such as
//! `O_APPEND` or `fstatfs`). Methods are named for the Linux system call they
//! service. Identity the guest can observe — `ls -i`, hardlinks reporting one
//! inode for two names, `fstat` after `link` — rides inside [`Stat`] as the
//! real `st_ino`/`st_dev`, so the traits stay path-addressed without an
//! inode-keyed `lookup`/`forget` protocol.

// Scaffolding: the Personality that drives these traits does not exist yet, so
// nothing here is constructed or called. Drop this once it is wired in.
#![allow(dead_code)]

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

/// One filesystem instance — a Linux superblock. Mounted into a namespace and
/// held as `Arc<dyn Vfs>`, so every method takes `&self`; a mutable filesystem
/// carries its own interior mutability.
///
/// Paths are **mount-relative**: the resolver has already turned the guest's
/// `dirfd`/`AT_FDCWD`-anchored, possibly-symlinked path into a path rooted at
/// this mount, and crossed any nested mountpoints before reaching here.
///
/// `Send + Sync` because a filesystem instance is shared as `Arc<dyn Vfs>` and,
/// once Chimera runs guest threads, will be reached from more than one.
pub trait Vfs: Send + Sync {
    /// Open (or create) a file or directory, yielding a positional handle.
    fn open(&self, path: &Path, flags: OpenFlags, mode: Mode) -> Result<Box<dyn File>, Errno>;

    /// Stat a path. `follow == false` is `lstat` semantics (do not dereference a
    /// final symlink); this one method backs `stat`, `lstat`, `newfstatat`, and
    /// `statx` once the Personality has folded in `AT_SYMLINK_NOFOLLOW`.
    fn stat(&self, path: &Path, follow: bool) -> Result<Stat, Errno>;

    /// Read the target of a symbolic link.
    fn readlink(&self, path: &Path) -> Result<PathBuf, Errno>;

    /// Create a directory.
    fn mkdir(&self, path: &Path, mode: Mode) -> Result<(), Errno>;

    /// Remove a non-directory entry.
    fn unlink(&self, path: &Path) -> Result<(), Errno>;

    /// Remove an empty directory. (`unlinkat(AT_REMOVEDIR)` lands here.)
    fn rmdir(&self, path: &Path) -> Result<(), Errno>;

    /// Create a symbolic link at `link` pointing to `target`.
    fn symlink(&self, target: &Path, link: &Path) -> Result<(), Errno>;

    /// Create a hard link `new` to the existing entry `old`. Both paths are
    /// relative to this same instance; a cross-filesystem `link` is rejected as
    /// `EXDEV` by the Personality and never reaches a `Vfs`.
    fn link(&self, old: &Path, new: &Path) -> Result<(), Errno>;

    /// Rename `from` to `to`. `flags` carries `renameat2` extensions
    /// (`RENAME_NOREPLACE`, `RENAME_EXCHANGE`, …); a cross-filesystem rename is
    /// `EXDEV` and handled above.
    fn rename(&self, from: &Path, to: &Path, flags: RenameFlags) -> Result<(), Errno>;

    /// Change a file's permission bits. `follow == false` atomically refuses a
    /// final symlink with `EOPNOTSUPP`, as `fchmodat2(AT_SYMLINK_NOFOLLOW)` does.
    fn chmod(&self, path: &Path, follow: bool, mode: Mode) -> Result<(), Errno>;

    /// Change a file's ownership. `follow == false` is `lchown` semantics. An
    /// id of `u32::MAX` (the kernel's `-1`) leaves that id unchanged.
    fn chown(&self, path: &Path, follow: bool, uid: u32, gid: u32) -> Result<(), Errno>;

    /// Set access and modification times. `None` sets both to now; otherwise
    /// the pair is `[atime, mtime]`, whose `nsec` fields may carry the
    /// kernel's `UTIME_NOW`/`UTIME_OMIT` specials, honored as `utimensat(2)`
    /// does. `follow == false` names a final symlink itself.
    fn utimens(&self, path: &Path, follow: bool, times: Option<[Timespec; 2]>)
    -> Result<(), Errno>;

    /// Report filesystem-wide statistics for `statfs` by path.
    fn statfs(&self, path: &Path) -> Result<StatFs, Errno>;

    /// The host path backing a mount-relative path, if this filesystem is host
    /// passthrough. The runtime uses it to load an `execve` target through its
    /// ELF loader without reaching back through this trait byte by byte; a
    /// synthetic filesystem returns `None` and is not (yet) executable-from.
    fn host_path(&self, _path: &Path) -> Option<PathBuf> {
        None
    }

    /// Whether this filesystem resolves and confines paths itself (e.g.
    /// [`HostFs`] via `openat2(RESOLVE_IN_ROOT)`). When the namespace is a single
    /// mount of a confining filesystem, the resolver skips its per-component walk
    /// and hands the raw path straight in, since the filesystem will follow
    /// symlinks and `..` scoped to its root. A filesystem that returns `false`
    /// relies on the namespace walker to resolve and confine on its behalf.
    ///
    /// [`HostFs`]: super::hostfs::HostFs
    fn confines(&self) -> bool {
        false
    }
}

/// One open file or directory. Positional: there is no internal cursor, so the
/// fd table owns the offset and a `dup`'d fd shares this handle. Held as
/// `Arc<dyn File>`, hence `&self` throughout — and `Send + Sync` for the same
/// sharing reason as [`Vfs`].
pub trait File: Send + Sync {
    /// Read at an absolute offset (`pread`/`read` after the fd table supplies
    /// the cursor; the `readv` scatter is the Personality's job).
    fn pread(&self, buf: &mut [u8], offset: u64) -> Result<usize, Errno>;

    /// Write at an absolute offset (`pwrite`/`write`/`writev`, likewise).
    fn pwrite(&self, buf: &[u8], offset: u64) -> Result<usize, Errno>;

    /// Append one write atomically at the current EOF, as Linux requires for
    /// `write`/`writev` on an `O_APPEND` open file description. Returns the
    /// file position after the write, so the shared descriptor state can track
    /// the kernel's per-description offset without re-stat'ing the path.
    fn append(&self, buf: &[u8]) -> Result<WriteResult, Errno>;

    /// Stat the open handle, for `fstat`/`fstatat(AT_EMPTY_PATH)`.
    fn fstat(&self) -> Result<Stat, Errno>;

    /// Report filesystem-wide statistics for this open handle, backing
    /// `fstatfs` even after rename or unlink.
    fn fstatfs(&self) -> Result<StatFs, Errno>;

    /// Truncate the file to `len` bytes.
    fn ftruncate(&self, len: u64) -> Result<(), Errno>;

    /// Flush the file to durable storage.
    fn fsync(&self) -> Result<(), Errno>;

    /// `chmod` through the open handle (`fchmod`).
    fn fchmod(&self, mode: Mode) -> Result<(), Errno>;

    /// `fchmodat2(fd, "", mode, AT_EMPTY_PATH)`, which also accepts an
    /// `O_PATH` handle.
    fn fchmodat_empty(&self, mode: Mode) -> Result<(), Errno> {
        self.fchmod(mode)
    }

    /// `chown` through the open handle (`fchown`). `u32::MAX` leaves an id
    /// unchanged, as in [`Vfs::chown`].
    fn fchown(&self, uid: u32, gid: u32) -> Result<(), Errno>;

    /// `fchownat(fd, "", uid, gid, AT_EMPTY_PATH)`, including `O_PATH`.
    fn fchownat_empty(&self, uid: u32, gid: u32) -> Result<(), Errno> {
        self.fchown(uid, gid)
    }

    /// Set times through the open handle (`futimens`); `times` as in
    /// [`Vfs::utimens`].
    fn futimens(&self, times: Option<[Timespec; 2]>) -> Result<(), Errno>;

    /// `utimensat(fd, "", times, AT_EMPTY_PATH)`, including `O_PATH`.
    fn utimensat_empty(&self, times: Option<[Timespec; 2]>) -> Result<(), Errno> {
        self.futimens(times)
    }

    /// Snapshot of a directory's entries, backing `getdents64` on a dirfd. The
    /// Personality encodes the snapshot into `linux_dirent64` records and keeps
    /// the read position in the fd table. A non-directory handle returns
    /// [`Errno::ENOTDIR`].
    fn getdents(&self) -> Result<Vec<DirEntry>, Errno> {
        Err(Errno::ENOTDIR)
    }

    /// The host descriptor backing this handle, if any. A passthrough handle
    /// (like [`HostFile`]) returns its real fd so a file-backed `mmap` can map
    /// it directly; a synthetic file returns `None`, leaving the Personality to
    /// emulate the mapping. Used only for `mmap`, never for ordinary I/O.
    ///
    /// [`HostFile`]: super::hostfs::HostFile
    fn host_fd(&self) -> Option<std::os::fd::RawFd> {
        None
    }

    /// Detach this file's kernel-allocated descriptor number so the caller can
    /// use it as the guest-visible reservation, moving the file's own backing
    /// out of the low fd space first. The number a passthrough open receives
    /// from the kernel is already the lowest available — exactly what
    /// `reserve_fd` would claim — so handing it over saves the close + re-dup
    /// round trip. `None` when there is no detachable number (no host backing,
    /// or the relocation failed); the caller falls back to `reserve_fd`.
    fn detach_reservation(&mut self) -> Option<std::os::fd::OwnedFd> {
        None
    }
}

/// The outcome of one write-like operation that advances a file position.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct WriteResult {
    pub count: usize,
    pub offset: u64,
}

/// A Linux errno, carried positive. The Personality encodes it as `-errno` in
/// the guest's return register, the way the kernel reports a failed syscall.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Errno(pub i32);

impl Errno {
    pub const EPERM: Errno = Errno(libc::EPERM);
    pub const ENOENT: Errno = Errno(libc::ENOENT);
    pub const EIO: Errno = Errno(libc::EIO);
    pub const EBADF: Errno = Errno(libc::EBADF);
    pub const EACCES: Errno = Errno(libc::EACCES);
    pub const EEXIST: Errno = Errno(libc::EEXIST);
    pub const EXDEV: Errno = Errno(libc::EXDEV);
    pub const ENOTDIR: Errno = Errno(libc::ENOTDIR);
    pub const EISDIR: Errno = Errno(libc::EISDIR);
    pub const EINVAL: Errno = Errno(libc::EINVAL);
    pub const ENOSPC: Errno = Errno(libc::ENOSPC);
    pub const EROFS: Errno = Errno(libc::EROFS);
    pub const ENOTEMPTY: Errno = Errno(libc::ENOTEMPTY);
    pub const ELOOP: Errno = Errno(libc::ELOOP);
    pub const ENOSYS: Errno = Errno(libc::ENOSYS);
    pub const ENOTTY: Errno = Errno(libc::ENOTTY);
    pub const ERANGE: Errno = Errno(libc::ERANGE);

    /// The raw positive errno value.
    pub fn raw(self) -> i32 {
        self.0
    }

    /// Wrap the errno from a failed `std::io` operation, falling back to `EIO`
    /// when the error carries no OS code (a `HostFs` will lean on this).
    pub fn from_io(err: &std::io::Error) -> Self {
        Errno(err.raw_os_error().unwrap_or(libc::EIO))
    }
}

/// The open-mode bits of a guest `open`/`openat`, kept raw (`O_RDONLY` … with
/// `O_CREAT`, `O_TRUNC`, `O_DIRECTORY`, `O_NOFOLLOW`, …). Helpers name the bits
/// a filesystem actually branches on; the rest stay accessible via [`raw`].
///
/// [`raw`]: OpenFlags::raw
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct OpenFlags(pub i32);

impl OpenFlags {
    pub fn raw(self) -> i32 {
        self.0
    }

    fn access(self) -> i32 {
        self.0 & libc::O_ACCMODE
    }

    pub fn readable(self) -> bool {
        self.access() == libc::O_RDONLY || self.access() == libc::O_RDWR
    }

    pub fn writable(self) -> bool {
        self.access() == libc::O_WRONLY || self.access() == libc::O_RDWR
    }

    pub fn create(self) -> bool {
        self.0 & libc::O_CREAT != 0
    }

    pub fn excl(self) -> bool {
        self.0 & libc::O_EXCL != 0
    }

    pub fn truncate(self) -> bool {
        self.0 & libc::O_TRUNC != 0
    }

    pub fn append(self) -> bool {
        self.0 & libc::O_APPEND != 0
    }

    pub fn directory(self) -> bool {
        self.0 & libc::O_DIRECTORY != 0
    }

    pub fn nofollow(self) -> bool {
        self.0 & libc::O_NOFOLLOW != 0
    }
}

/// The permission/type bits passed to a creating syscall, raw as the guest gave
/// them (`mode_t`, already masked by the guest's umask).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Mode(pub u32);

/// The `renameat2` flag set. `EMPTY` is a plain `rename`.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub struct RenameFlags(pub u32);

impl RenameFlags {
    pub const EMPTY: RenameFlags = RenameFlags(0);

    pub fn noreplace(self) -> bool {
        self.0 & libc::RENAME_NOREPLACE != 0
    }

    pub fn exchange(self) -> bool {
        self.0 & libc::RENAME_EXCHANGE != 0
    }
}

/// The kind of a directory entry / stat target.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum FileType {
    Regular,
    Directory,
    Symlink,
    CharDevice,
    BlockDevice,
    Fifo,
    Socket,
}

/// A wall-clock timestamp, split as `struct timespec` is.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Timespec {
    pub sec: i64,
    pub nsec: i64,
}

/// A symbolic `struct stat`. The Personality lays this out into the guest's
/// `stat`/`statx` buffer. `ino`/`dev` are the *real* backing identity (so
/// hardlinks and `fstat`-after-`link` agree); a synthetic filesystem mints its
/// own stable values.
#[derive(Copy, Clone, Debug)]
pub struct Stat {
    pub dev: u64,
    pub ino: u64,
    pub file_type: FileType,
    pub mode: Mode,
    pub nlink: u64,
    pub uid: u32,
    pub gid: u32,
    pub rdev: u64,
    pub size: u64,
    pub blksize: u64,
    pub blocks: u64,
    pub atime: Timespec,
    pub mtime: Timespec,
    pub ctime: Timespec,
}

/// One entry from a directory snapshot, the raw material for `getdents64`.
#[derive(Clone, Debug)]
pub struct DirEntry {
    pub ino: u64,
    pub file_type: FileType,
    pub name: OsString,
}

/// A symbolic `struct statfs`, laid out by the Personality for `statfs`/
/// `fstatfs`.
#[derive(Copy, Clone, Debug)]
pub struct StatFs {
    pub fs_type: i64,
    pub block_size: u64,
    pub fragment_size: u64,
    pub blocks: u64,
    pub blocks_free: u64,
    pub blocks_available: u64,
    pub files: u64,
    pub files_free: u64,
    pub fsid: [i32; 2],
    pub name_max: u64,
    pub flags: u64,
}
