//! `chimera mount`: serve a filesystem's merged view through FUSE.
//!
//! The mount presents exactly what a `chimera run` resuming the filesystem
//! would see — the live host below, the filesystem's change-set above — and
//! writes through it land in the change-set, so the host stays untouched
//! until `fs apply`. The bridge drives the same [`Vfs`] object a run's
//! Personality does; only the front end differs: the kernel's FUSE protocol
//! stands where the guest's syscalls would.
//!
//! FUSE speaks inodes, the [`Vfs`] speaks paths, so the bridge keeps a
//! bidirectional ino⇄path table, minting a number the first time a path is
//! seen. The kernel resolves symlinks and `..` itself and looks up one
//! component at a time, so every path handed to the [`Vfs`] is normalized
//! and symlink-free — the same contract the namespace walker gives it.
//! Entries and attributes are advertised with a zero TTL by default: the
//! lower layer is the live host, which can change under the mount at any
//! moment, so the kernel is never allowed to cache a resolution. `--cache`
//! raises the TTL for workloads that prefer fewer round trips over
//! host-coherence. One consequence of path identity: two hard links to one
//! file report two inode numbers (their `nlink` still counts links);
//! programs that deduplicate by inode see them as distinct files.

use std::{
    collections::HashMap,
    ffi::{CString, OsStr, OsString},
    io,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use chimera::{
    Errno, FileType as VfsFileType, HostFs, Mode, OpenFlags, OverlayFs, RenameFlags, Stat,
    Timespec, Vfs,
};

use crate::fuser::{
    self, FileAttr, FileType, MountOption, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory,
    ReplyEmpty, ReplyEntry, ReplyOpen, ReplyStatfs, ReplyWrite, ReplyXattr, Request, TimeOrNow,
};
use crate::{fs, opts::MountCmd};

/// Entry point for `chimera mount`.
pub fn command(cmd: MountCmd) -> ExitCode {
    match mount(&cmd) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("chimera: {err}");
            ExitCode::FAILURE
        }
    }
}

fn mount(cmd: &MountCmd) -> io::Result<()> {
    if cmd.filesystem == "host" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the live host is already mounted; name a filesystem",
        ));
    }
    if let Some(scheme) = fs::scheme(OsStr::new(&cmd.filesystem)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "unknown filesystem scheme \"{scheme}:\" (a path whose first component contains a colon needs a leading ./)"
            ),
        ));
    }
    // The selector reads as `--in` does: an id names a kept filesystem, a
    // path names a change-set directory, created on first use. The session
    // share keeps `fs rm`/`fs prune` from deleting the tree while mounted.
    let fsys = if cmd.filesystem.contains('/') {
        fs::attach(OsStr::new(&cmd.filesystem))
    } else {
        fs::resume(&cmd.filesystem)
    }
    .map_err(|e| io::Error::new(e.kind(), format!("cannot open filesystem: {e}")))?;

    let mountpoint = PathBuf::from(&cmd.mountpoint);
    if !mountpoint.is_dir() {
        fsys.finish(false);
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("mountpoint {} is not a directory", mountpoint.display()),
        ));
    }

    let host = Arc::new(HostFs::new("/").expect("host root / is a directory"));
    let root: Arc<dyn Vfs> = match OverlayFs::new(host, &fsys.root) {
        Ok(overlay) => Arc::new(overlay),
        Err(err) => {
            let err = io::Error::from_raw_os_error(err.raw());
            let err = io::Error::new(
                err.kind(),
                format!("cannot open filesystem {}: {err}", fsys.root.display()),
            );
            fsys.finish(false);
            return Err(err);
        }
    };

    // `DefaultPermissions` puts permission checking in the kernel, judged
    // from the merged view's own uid/gid/mode — the check a run's guest
    // would face at the host boundary.
    let mut options = vec![
        MountOption::FSName(format!("chimera:{}", fsys.id)),
        MountOption::Subtype("chimera".to_string()),
        MountOption::DefaultPermissions,
    ];
    if cmd.read_only {
        options.push(MountOption::RO);
    }
    let bridge = Bridge::new(root, Duration::from_secs(cmd.cache));
    let mut session = fuser::Session::new(bridge, &mountpoint, &options)?;

    // Unmount on SIGINT/SIGTERM. A signal handler cannot unmount (the
    // unmounter takes locks), so the signals are blocked process-wide and a
    // helper thread waits for one synchronously; its unmount ends the
    // session loop below. An external `fusermount -u` ends it the same way.
    let mut unmounter = session.unmount_callable();
    let set = block_signals()?;
    std::thread::spawn(move || {
        let mut sig: libc::c_int = 0;
        if unsafe { libc::sigwait(&set, &mut sig) } == 0 {
            let _ = unmounter.unmount();
        }
    });

    eprintln!(
        "chimera: mounted {} at {}; unmount with `fusermount -u {}` or Ctrl-C",
        fsys.id,
        mountpoint.display(),
        mountpoint.display(),
    );
    let result = session.run();
    drop(session);
    fsys.finish(false);
    result
}

/// Block SIGINT and SIGTERM for this thread and every thread it spawns,
/// returning the set for `sigwait`.
fn block_signals() -> io::Result<libc::sigset_t> {
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::sigaddset(&mut set, libc::SIGTERM);
        if libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut()) != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(set)
    }
}

/// The reserved xattr namespace of the delta format. Bookkeeping is delta
/// state, never part of the merged view: reads answer as if the names did
/// not exist and writes are refused, the same rules the Personality gives a
/// guest.
const XATTR_NAMESPACE: &[u8] = b"user.chimera.";

/// One open FUSE handle.
enum Handle {
    File(Box<dyn chimera::File>),
    /// A directory snapshot taken at `opendir`, so `readdir` offsets stay
    /// stable across calls while entries come and go beneath the mount.
    Dir(Vec<(u64, FileType, OsString)>),
}

struct Bridge {
    root: Arc<dyn Vfs>,
    /// How long the kernel may cache an entry or attribute (`--cache`).
    /// Zero, the default, forbids caching outright: the lower layer is the
    /// live host, and a cached resolution could outlive the host state it
    /// described. A nonzero value accepts staleness bounded by the TTL in
    /// exchange for skipped round trips; changes made through the mount
    /// itself stay coherent either way, since the kernel tracks its own
    /// writes.
    ttl: Duration,
    /// The ino⇄path table. Entries live for the session: the kernel may
    /// hold an inode number as long as the mount exists, and the table is
    /// the only thing that can turn it back into a path.
    paths: HashMap<u64, PathBuf>,
    inos: HashMap<PathBuf, u64>,
    next_ino: u64,
    handles: HashMap<u64, Handle>,
    next_fh: u64,
}

impl Bridge {
    fn new(root: Arc<dyn Vfs>, ttl: Duration) -> Self {
        let mut paths = HashMap::new();
        let mut inos = HashMap::new();
        paths.insert(fuser::FUSE_ROOT_ID, PathBuf::from("/"));
        inos.insert(PathBuf::from("/"), fuser::FUSE_ROOT_ID);
        Bridge {
            root,
            ttl,
            paths,
            inos,
            next_ino: fuser::FUSE_ROOT_ID + 1,
            handles: HashMap::new(),
            next_fh: 1,
        }
    }

    /// The inode number for a path, minted on first sight.
    fn ino_for(&mut self, path: &Path) -> u64 {
        if let Some(&ino) = self.inos.get(path) {
            return ino;
        }
        let ino = self.next_ino;
        self.next_ino += 1;
        self.paths.insert(ino, path.to_path_buf());
        self.inos.insert(path.to_path_buf(), ino);
        ino
    }

    /// The path behind an inode the kernel presented. An unknown number is
    /// the kernel's handle to something this session never advertised.
    fn path_of(&self, ino: u64) -> Result<PathBuf, libc::c_int> {
        self.paths.get(&ino).cloned().ok_or(libc::ESTALE)
    }

    fn child(&self, parent: u64, name: &OsStr) -> Result<PathBuf, libc::c_int> {
        Ok(self.path_of(parent)?.join(name))
    }

    /// Drop a path's table entry after its object is gone: a later creation
    /// at the same name must mint a fresh inode, not resurrect this one.
    fn evict(&mut self, path: &Path) {
        if let Some(ino) = self.inos.remove(path) {
            self.paths.remove(&ino);
        }
    }

    /// Re-key everything under `from` to live under `to`, dropping whatever
    /// mappings the rename clobbered at the destination first.
    fn remap(&mut self, from: &Path, to: &Path) {
        let dead: Vec<u64> = self
            .paths
            .iter()
            .filter(|(_, p)| p.starts_with(to))
            .map(|(&ino, _)| ino)
            .collect();
        for ino in dead {
            if let Some(p) = self.paths.remove(&ino) {
                self.inos.remove(&p);
            }
        }
        let moved: Vec<(u64, PathBuf)> = self
            .paths
            .iter()
            .filter(|(_, p)| p.starts_with(from))
            .map(|(&ino, p)| (ino, p.clone()))
            .collect();
        for (ino, old) in moved {
            let new = match old.strip_prefix(from) {
                Ok(rest) if rest.as_os_str().is_empty() => to.to_path_buf(),
                Ok(rest) => to.join(rest),
                Err(_) => continue,
            };
            self.inos.remove(&old);
            self.paths.insert(ino, new.clone());
            self.inos.insert(new, ino);
        }
    }

    fn insert_handle(&mut self, handle: Handle) -> u64 {
        let fh = self.next_fh;
        self.next_fh += 1;
        self.handles.insert(fh, handle);
        fh
    }

    fn file(&self, fh: u64) -> Result<&dyn chimera::File, libc::c_int> {
        match self.handles.get(&fh) {
            Some(Handle::File(f)) => Ok(f.as_ref()),
            _ => Err(libc::EBADF),
        }
    }

    /// Stat + table entry for a path that just appeared (lookup or any
    /// creating operation), as the `ReplyEntry` pair.
    fn entry(&mut self, path: &Path) -> Result<(u64, FileAttr), libc::c_int> {
        let stat = self.root.stat(path, false).map_err(Errno::raw)?;
        let ino = self.ino_for(path);
        Ok((ino, attr(&stat, ino)))
    }
}

fn kind(t: VfsFileType) -> FileType {
    match t {
        VfsFileType::Regular => FileType::RegularFile,
        VfsFileType::Directory => FileType::Directory,
        VfsFileType::Symlink => FileType::Symlink,
        VfsFileType::CharDevice => FileType::CharDevice,
        VfsFileType::BlockDevice => FileType::BlockDevice,
        VfsFileType::Fifo => FileType::NamedPipe,
        VfsFileType::Socket => FileType::Socket,
    }
}

fn system_time(ts: Timespec) -> SystemTime {
    if ts.sec >= 0 {
        UNIX_EPOCH + Duration::new(ts.sec as u64, ts.nsec as u32)
    } else {
        // A pre-epoch timespec still counts nanoseconds forward: the moment
        // is `sec` (negative) plus `nsec` in [0, 1e9).
        UNIX_EPOCH - Duration::from_secs(ts.sec.unsigned_abs()) + Duration::new(0, ts.nsec as u32)
    }
}

fn timespec(t: SystemTime) -> Timespec {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => Timespec {
            sec: d.as_secs() as i64,
            nsec: d.subsec_nanos() as i64,
        },
        Err(e) => {
            let d = e.duration();
            let mut sec = -(d.as_secs() as i64);
            let mut nsec = d.subsec_nanos() as i64;
            if nsec > 0 {
                sec -= 1;
                nsec = 1_000_000_000 - nsec;
            }
            Timespec { sec, nsec }
        }
    }
}

/// One side of a `setattr` time pair as [`Vfs::utimens`] takes it, the
/// kernel's specials preserved.
fn utimens_arg(t: Option<TimeOrNow>) -> Timespec {
    match t {
        None => Timespec {
            sec: 0,
            nsec: libc::UTIME_OMIT,
        },
        Some(TimeOrNow::Now) => Timespec {
            sec: 0,
            nsec: libc::UTIME_NOW,
        },
        Some(TimeOrNow::SpecificTime(t)) => timespec(t),
    }
}

fn attr(stat: &Stat, ino: u64) -> FileAttr {
    FileAttr {
        ino,
        size: stat.size,
        blocks: stat.blocks,
        atime: system_time(stat.atime),
        mtime: system_time(stat.mtime),
        ctime: system_time(stat.ctime),
        crtime: UNIX_EPOCH,
        kind: kind(stat.file_type),
        perm: (stat.mode.0 & 0o7777) as u16,
        nlink: stat.nlink as u32,
        uid: stat.uid,
        gid: stat.gid,
        rdev: stat.rdev as u32,
        blksize: stat.blksize as u32,
        flags: 0,
    }
}

impl fuser::Filesystem for Bridge {
    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let path = match self.child(parent, name) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        match self.entry(&path) {
            Ok((_, attr)) => reply.entry(&self.ttl, &attr, 0),
            Err(e) => reply.error(e),
        }
    }

    fn getattr(&mut self, _req: &Request<'_>, ino: u64, fh: Option<u64>, reply: ReplyAttr) {
        // Prefer the open handle: it answers for an unlinked-but-open file
        // whose path is gone.
        if let Some(fh) = fh
            && let Ok(file) = self.file(fh)
        {
            return match file.fstat() {
                Ok(stat) => reply.attr(&self.ttl, &attr(&stat, ino)),
                Err(e) => reply.error(e.raw()),
            };
        }
        let path = match self.path_of(ino) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        match self.root.stat(&path, false) {
            Ok(stat) => reply.attr(&self.ttl, &attr(&stat, ino)),
            Err(e) => reply.error(e.raw()),
        }
    }

    fn setattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        mode: Option<u32>,
        uid: Option<u32>,
        gid: Option<u32>,
        size: Option<u64>,
        atime: Option<TimeOrNow>,
        mtime: Option<TimeOrNow>,
        _ctime: Option<SystemTime>,
        fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        let path = match self.path_of(ino) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        if let Some(mode) = mode
            && let Err(e) = self.root.chmod(&path, false, Mode(mode & 0o7777))
        {
            return reply.error(e.raw());
        }
        if (uid.is_some() || gid.is_some())
            && let Err(e) = self.root.chown(
                &path,
                false,
                uid.unwrap_or(u32::MAX),
                gid.unwrap_or(u32::MAX),
            )
        {
            return reply.error(e.raw());
        }
        if let Some(size) = size {
            // Truncate through the caller's own handle when it sent one; a
            // path truncate opens for write, which copies the file up first.
            let result = match fh.and_then(|fh| self.file(fh).ok()) {
                Some(file) => file.ftruncate(size),
                None => self
                    .root
                    .open(&path, OpenFlags(libc::O_WRONLY), Mode(0))
                    .and_then(|f| f.ftruncate(size)),
            };
            if let Err(e) = result {
                return reply.error(e.raw());
            }
        }
        if (atime.is_some() || mtime.is_some())
            && let Err(e) =
                self.root
                    .utimens(&path, false, Some([utimens_arg(atime), utimens_arg(mtime)]))
        {
            return reply.error(e.raw());
        }
        match self.root.stat(&path, false) {
            Ok(stat) => reply.attr(&self.ttl, &attr(&stat, ino)),
            Err(e) => reply.error(e.raw()),
        }
    }

    fn readlink(&mut self, _req: &Request<'_>, ino: u64, reply: ReplyData) {
        let path = match self.path_of(ino) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        match self.root.readlink(&path) {
            Ok(target) => reply.data(target.as_os_str().as_bytes()),
            Err(e) => reply.error(e.raw()),
        }
    }

    fn mknod(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        rdev: u32,
        reply: ReplyEntry,
    ) {
        let path = match self.child(parent, name) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        if let Err(e) = self.root.mknod(&path, Mode(mode), u64::from(rdev)) {
            return reply.error(e.raw());
        }
        match self.entry(&path) {
            Ok((_, attr)) => reply.entry(&self.ttl, &attr, 0),
            Err(e) => reply.error(e),
        }
    }

    fn mkdir(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let path = match self.child(parent, name) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        if let Err(e) = self.root.mkdir(&path, Mode(mode & 0o7777)) {
            return reply.error(e.raw());
        }
        match self.entry(&path) {
            Ok((_, attr)) => reply.entry(&self.ttl, &attr, 0),
            Err(e) => reply.error(e),
        }
    }

    fn unlink(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let path = match self.child(parent, name) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        match self.root.unlink(&path) {
            Ok(()) => {
                self.evict(&path);
                reply.ok()
            }
            Err(e) => reply.error(e.raw()),
        }
    }

    fn rmdir(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let path = match self.child(parent, name) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        match self.root.rmdir(&path) {
            Ok(()) => {
                self.evict(&path);
                reply.ok()
            }
            Err(e) => reply.error(e.raw()),
        }
    }

    fn symlink(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        link_name: &OsStr,
        target: &Path,
        reply: ReplyEntry,
    ) {
        let path = match self.child(parent, link_name) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        if let Err(e) = self.root.symlink(target, &path) {
            return reply.error(e.raw());
        }
        match self.entry(&path) {
            Ok((_, attr)) => reply.entry(&self.ttl, &attr, 0),
            Err(e) => reply.error(e),
        }
    }

    fn rename(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        flags: u32,
        reply: ReplyEmpty,
    ) {
        let (from, to) = match (self.child(parent, name), self.child(newparent, newname)) {
            (Ok(from), Ok(to)) => (from, to),
            (Err(e), _) | (_, Err(e)) => return reply.error(e),
        };
        let flags = RenameFlags(flags);
        match self.root.rename(&from, &to, flags) {
            Ok(()) => {
                if flags.exchange() {
                    // Both survive with swapped names; park one subtree at a
                    // name no path can collide with while the other moves.
                    let parked = PathBuf::from("\0exchange");
                    self.remap(&from, &parked);
                    self.remap(&to, &from);
                    self.remap(&parked, &to);
                } else {
                    self.remap(&from, &to);
                }
                reply.ok()
            }
            Err(e) => reply.error(e.raw()),
        }
    }

    fn link(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        newparent: u64,
        newname: &OsStr,
        reply: ReplyEntry,
    ) {
        let (old, new) = match (self.path_of(ino), self.child(newparent, newname)) {
            (Ok(old), Ok(new)) => (old, new),
            (Err(e), _) | (_, Err(e)) => return reply.error(e),
        };
        if let Err(e) = self.root.link(&old, &new) {
            return reply.error(e.raw());
        }
        match self.entry(&new) {
            Ok((_, attr)) => reply.entry(&self.ttl, &attr, 0),
            Err(e) => reply.error(e),
        }
    }

    fn open(&mut self, _req: &Request<'_>, ino: u64, flags: i32, reply: ReplyOpen) {
        let path = match self.path_of(ino) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        match self.root.open(&path, open_flags(flags), Mode(0)) {
            Ok(file) => {
                let fh = self.insert_handle(Handle::File(file));
                reply.opened(fh, 0);
            }
            Err(e) => reply.error(e.raw()),
        }
    }

    fn read(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        let file = match self.file(fh) {
            Ok(file) => file,
            Err(e) => return reply.error(e),
        };
        // The kernel takes a short count as end-of-file, so fill the buffer
        // until the file itself says otherwise.
        let mut buf = vec![0u8; size as usize];
        let mut filled = 0;
        while filled < buf.len() {
            match file.pread(&mut buf[filled..], offset as u64 + filled as u64) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(Errno(libc::EINTR)) => continue,
                Err(e) if filled == 0 => return reply.error(e.raw()),
                Err(_) => break,
            }
        }
        reply.data(&buf[..filled]);
    }

    fn write(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        fh: u64,
        offset: i64,
        data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyWrite,
    ) {
        let file = match self.file(fh) {
            Ok(file) => file,
            Err(e) => return reply.error(e),
        };
        // A short FUSE write is an error to the caller, not a retry point,
        // so push until everything the kernel sent has landed.
        let mut written = 0;
        while written < data.len() {
            match file.pwrite(&data[written..], offset as u64 + written as u64) {
                Ok(0) => return reply.error(libc::EIO),
                Ok(n) => written += n,
                Err(Errno(libc::EINTR)) => continue,
                Err(e) => return reply.error(e.raw()),
            }
        }
        reply.written(written as u32);
    }

    fn flush(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _lock_owner: u64,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    fn release(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        fh: u64,
        _flags: i32,
        _lock_owner: Option<u64>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        self.handles.remove(&fh);
        reply.ok();
    }

    fn fsync(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        fh: u64,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        match self.file(fh) {
            Ok(file) => match file.fsync() {
                Ok(()) => reply.ok(),
                Err(e) => reply.error(e.raw()),
            },
            Err(e) => reply.error(e),
        }
    }

    fn opendir(&mut self, _req: &Request<'_>, ino: u64, _flags: i32, reply: ReplyOpen) {
        let path = match self.path_of(ino) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        let entries = self
            .root
            .open(
                &path,
                OpenFlags(libc::O_RDONLY | libc::O_DIRECTORY),
                Mode(0),
            )
            .and_then(|dir| dir.getdents());
        match entries {
            Ok(entries) => {
                // The Vfs reports only real entries; `.` and `..` are the
                // caller's to synthesize, here as everywhere.
                let parent = path.parent().unwrap_or(&path).to_path_buf();
                let mut list = Vec::with_capacity(entries.len() + 2);
                list.push((ino, FileType::Directory, OsString::from(".")));
                list.push((
                    self.ino_for(&parent),
                    FileType::Directory,
                    OsString::from(".."),
                ));
                for entry in entries {
                    let child = path.join(&entry.name);
                    list.push((self.ino_for(&child), kind(entry.file_type), entry.name));
                }
                let fh = self.insert_handle(Handle::Dir(list));
                reply.opened(fh, 0);
            }
            Err(e) => reply.error(e.raw()),
        }
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        let Some(Handle::Dir(entries)) = self.handles.get(&fh) else {
            return reply.error(libc::EBADF);
        };
        for (i, (ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
            if reply.add(*ino, (i + 1) as i64, *kind, name) {
                break;
            }
        }
        reply.ok();
    }

    fn releasedir(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        fh: u64,
        _flags: i32,
        reply: ReplyEmpty,
    ) {
        self.handles.remove(&fh);
        reply.ok();
    }

    fn fsyncdir(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _datasync: bool,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    fn statfs(&mut self, _req: &Request<'_>, ino: u64, reply: ReplyStatfs) {
        let path = match self.path_of(ino) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        match self.root.statfs(&path) {
            Ok(st) => reply.statfs(
                st.blocks,
                st.blocks_free,
                st.blocks_available,
                st.files,
                st.files_free,
                st.block_size as u32,
                st.name_max as u32,
                st.fragment_size as u32,
            ),
            Err(e) => reply.error(e.raw()),
        }
    }

    fn setxattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        name: &OsStr,
        value: &[u8],
        flags: i32,
        position: u32,
        reply: ReplyEmpty,
    ) {
        if position != 0 {
            return reply.error(libc::EINVAL);
        }
        if name.as_bytes().starts_with(XATTR_NAMESPACE) {
            return reply.error(libc::EPERM);
        }
        let path = match self.path_of(ino) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        match self.root.setxattr(&path, false, name, value, flags) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(e.raw()),
        }
    }

    // The Vfs has no xattr readers — a run's Personality serves them from
    // the backing host path — so the bridge does the same through
    // `host_path`, with the bookkeeping names hidden.

    fn getxattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        name: &OsStr,
        size: u32,
        reply: ReplyXattr,
    ) {
        if name.as_bytes().starts_with(XATTR_NAMESPACE) {
            return reply.error(libc::ENODATA);
        }
        let host = match self.path_of(ino).map(|p| self.root.host_path(&p)) {
            Ok(Some(host)) => host,
            Ok(None) => return reply.error(libc::ENOTSUP),
            Err(e) => return reply.error(e),
        };
        let (chost, cname) = match (cbytes(host.as_os_str().as_bytes()), cbytes(name.as_bytes())) {
            (Ok(h), Ok(n)) => (h, n),
            _ => return reply.error(libc::EINVAL),
        };
        let mut buf = vec![0u8; size as usize];
        let n = unsafe {
            libc::lgetxattr(
                chost.as_ptr(),
                cname.as_ptr(),
                buf.as_mut_ptr() as *mut libc::c_void,
                buf.len(),
            )
        };
        if n < 0 {
            return reply.error(
                io::Error::last_os_error()
                    .raw_os_error()
                    .unwrap_or(libc::EIO),
            );
        }
        if size == 0 {
            reply.size(n as u32);
        } else {
            reply.data(&buf[..n as usize]);
        }
    }

    fn listxattr(&mut self, _req: &Request<'_>, ino: u64, size: u32, reply: ReplyXattr) {
        let host = match self.path_of(ino).map(|p| self.root.host_path(&p)) {
            Ok(Some(host)) => host,
            Ok(None) => return reply.error(libc::ENOTSUP),
            Err(e) => return reply.error(e),
        };
        let Ok(chost) = cbytes(host.as_os_str().as_bytes()) else {
            return reply.error(libc::EINVAL);
        };
        // Fetch the full list regardless of the caller's size: the hidden
        // names must come out before the size protocol is answered, or a
        // probe would over-report and a fetch could overflow-by-hidden.
        let mut names = vec![0u8; 1024];
        let len = loop {
            let n = unsafe {
                libc::llistxattr(
                    chost.as_ptr(),
                    names.as_mut_ptr() as *mut libc::c_char,
                    names.len(),
                )
            };
            if n >= 0 {
                break n as usize;
            }
            match io::Error::last_os_error().raw_os_error() {
                Some(libc::ERANGE) => names.resize(names.len() * 2, 0),
                Some(libc::ENOTSUP) => break 0,
                Some(e) => return reply.error(e),
                None => return reply.error(libc::EIO),
            }
        };
        let mut kept = Vec::with_capacity(len);
        for name in names[..len].split(|&b| b == 0).filter(|n| !n.is_empty()) {
            if name.starts_with(XATTR_NAMESPACE) {
                continue;
            }
            kept.extend_from_slice(name);
            kept.push(0);
        }
        if size == 0 {
            reply.size(kept.len() as u32);
        } else if kept.len() <= size as usize {
            reply.data(&kept);
        } else {
            reply.error(libc::ERANGE);
        }
    }

    fn removexattr(&mut self, _req: &Request<'_>, ino: u64, name: &OsStr, reply: ReplyEmpty) {
        if name.as_bytes().starts_with(XATTR_NAMESPACE) {
            return reply.error(libc::EPERM);
        }
        let path = match self.path_of(ino) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        match self.root.removexattr(&path, false, name) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(e.raw()),
        }
    }

    fn create(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        flags: i32,
        reply: ReplyCreate,
    ) {
        let path = match self.child(parent, name) {
            Ok(path) => path,
            Err(e) => return reply.error(e),
        };
        let flags = OpenFlags(open_flags(flags).raw() | libc::O_CREAT);
        match self.root.open(&path, flags, Mode(mode & 0o7777)) {
            Ok(file) => {
                let (ino, attr) = match self.entry(&path) {
                    Ok(entry) => entry,
                    Err(e) => return reply.error(e),
                };
                let _ = ino;
                let fh = self.insert_handle(Handle::File(file));
                reply.created(&self.ttl, &attr, 0, fh, 0);
            }
            Err(e) => reply.error(e.raw()),
        }
    }

    fn fallocate(
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        fh: u64,
        offset: i64,
        length: i64,
        mode: i32,
        reply: ReplyEmpty,
    ) {
        match self.file(fh) {
            Ok(file) => match file.fallocate(mode, offset as u64, length as u64) {
                Ok(()) => reply.ok(),
                Err(e) => reply.error(e.raw()),
            },
            Err(e) => reply.error(e),
        }
    }
}

/// `CString` from raw bytes; an embedded NUL is invalid input.
fn cbytes(bytes: &[u8]) -> Result<CString, ()> {
    CString::new(bytes).map_err(|_| ())
}

/// The kernel's `__FMODE_EXEC`: FUSE open requests carry the opening file's
/// `f_flags` verbatim, and an open that backs an `execve` includes this
/// kernel-internal bit, which is not an `O_*` flag at all.
const FMODE_EXEC: i32 = 0x20;

/// An incoming FUSE open-flag word, reduced to what the [`Vfs`] may see.
/// The kernel-internal fmode bits must go because `HostFs` opens through
/// `openat2`, which rejects unknown flag bits with `EINVAL`. `O_APPEND`
/// goes because the kernel computes the append offset itself and sends
/// positional writes — and a Linux `pwrite` on an `O_APPEND` descriptor
/// ignores its offset.
fn open_flags(flags: i32) -> OpenFlags {
    OpenFlags(flags & !(FMODE_EXEC | libc::O_APPEND))
}
