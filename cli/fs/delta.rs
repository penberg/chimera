//! Reading and replaying a filesystem's delta: the half of the `fs` tooling
//! that needs the whiteout and origin xattrs the runtime's delta format
//! defines, which exist on Linux alone. See [`super`] for the portable half.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use chimera::delta::{
    Origin, is_applied, is_opaque, is_whiteout, mark_applied, origin, record_origin,
};

use super::{cpath, replay_metadata, replayable_xattr, resolve};

/// One entry of a filesystem's change-set, keyed by the guest-visible path.
struct Change {
    kind: Kind,
    /// The absolute path the guest saw (and the host path the change is
    /// about — the mount covers `/`).
    path: PathBuf,
    /// The upper file backing the change; empty for a deletion.
    upper: PathBuf,
}

#[derive(PartialEq, Copy, Clone)]
enum Kind {
    Added,
    Modified,
    Deleted,
}

impl Kind {
    fn letter(self) -> char {
        match self {
            Kind::Added => 'A',
            Kind::Modified => 'M',
            Kind::Deleted => 'D',
        }
    }
}

/// Walk a delta's `data/` tree into the change list, parents before
/// children. Directories themselves are listed only when opaque (they
/// replace the lower directory wholesale); an ordinary upper directory is
/// just the scaffolding under its children.
fn changes(root: &Path) -> io::Result<Vec<Change>> {
    let data = root.join("data");
    let mut out = Vec::new();
    walk(&data, Path::new("/"), &mut out)?;
    Ok(out)
}

fn walk(upper_dir: &Path, guest_dir: &Path, out: &mut Vec<Change>) -> io::Result<()> {
    let mut entries: Vec<_> = fs::read_dir(upper_dir)?.filter_map(Result::ok).collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let upper = entry.path();
        let guest = guest_dir.join(entry.file_name());
        let md = fs::symlink_metadata(&upper)?;
        if is_whiteout(&upper).map_err(errno_to_io)? {
            out.push(Change {
                kind: Kind::Deleted,
                path: guest,
                upper,
            });
            continue;
        }
        if md.is_dir() {
            if is_opaque(&upper).map_err(errno_to_io)? {
                out.push(Change {
                    kind: kind_against_host(&guest),
                    path: guest.clone(),
                    upper: upper.clone(),
                });
            }
            walk(&upper, &guest, out)?;
            continue;
        }
        out.push(Change {
            kind: kind_against_host(&guest),
            path: guest,
            upper,
        });
    }
    Ok(())
}

/// Added or Modified, judged against the live host.
fn kind_against_host(guest: &Path) -> Kind {
    if fs::symlink_metadata(guest).is_ok() {
        Kind::Modified
    } else {
        Kind::Added
    }
}

fn errno_to_io(e: chimera::Errno) -> io::Error {
    io::Error::from_raw_os_error(e.raw())
}

pub fn diff(selector: &str) -> io::Result<()> {
    let root = resolve(selector)?;
    for change in changes(&root)? {
        println!("{} {}", change.kind.letter(), change.path.display());
    }
    Ok(())
}

/// Copy the filesystem's changes onto the host. A modified file whose host
/// copy no longer matches the origin recorded at copy-up is refused rather
/// than clobbered; everything else applies, and any conflict makes the whole
/// command report failure.
pub fn apply(selector: &str) -> io::Result<()> {
    let root = resolve(selector)?;
    let mut conflicts = 0u32;
    for change in changes(&root)? {
        match apply_change(&change) {
            Ok(()) => println!("{} {}", change.kind.letter(), change.path.display()),
            Err(ApplyError::Conflict) => {
                conflicts += 1;
                eprintln!(
                    "chimera: conflict: {} changed on the host since the filesystem change (skipped)",
                    change.path.display(),
                );
            }
            Err(ApplyError::Io(e)) => {
                return Err(io::Error::new(
                    e.kind(),
                    format!("applying {}: {e}", change.path.display()),
                ));
            }
        }
    }
    if conflicts > 0 {
        return Err(io::Error::other(format!(
            "{conflicts} conflict(s); resolve on the host and re-run apply"
        )));
    }
    Ok(())
}

enum ApplyError {
    Conflict,
    Io(io::Error),
}

impl From<io::Error> for ApplyError {
    fn from(e: io::Error) -> Self {
        ApplyError::Io(e)
    }
}

/// Remove whatever the host has at `path` so a replacement (or deletion) can
/// land. The removal call is selected from the existing target, never from
/// the incoming type — and never through a symlink: a symlink is unlinked
/// itself, only a real directory is removed recursively. An absent target is
/// already removed. A target that changes type between inspection and
/// removal surfaces as the resulting I/O error, not a mis-typed deletion.
fn remove_host(path: &std::path::Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(md) if md.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

fn apply_change(change: &Change) -> Result<(), ApplyError> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};

    let host = &change.path;
    if change.kind == Kind::Deleted {
        // Once this whiteout's removal has been applied, the name belongs
        // to the host again: an entry there now is the host's own and a
        // rerun must refuse it, never delete it twice.
        if is_applied(&change.upper).map_err(errno_to_io)? {
            return match fs::symlink_metadata(host) {
                Ok(_) => Err(ApplyError::Conflict),
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(e.into()),
            };
        }
        // The applied mark lands before the removal: a failure in between
        // surfaces as a conflict on the rerun — the safe direction, since
        // the mark is what guarantees no later host entry gets clobbered.
        mark_applied(&change.upper).map_err(errno_to_io)?;
        remove_host(host)?;
        return Ok(());
    }

    let md = fs::symlink_metadata(&change.upper)?;
    if md.is_dir() {
        // An opaque directory replaces the lower entry wholesale, whatever
        // its type; its children follow in the walk.
        remove_host(host)?;
        fs::create_dir_all(host)?;
        replay_metadata(&md, &change.upper, host, replayable_xattr)?;
        return Ok(());
    }

    if md.is_file() {
        // The origin check, both ways around. A copied-up file knows which
        // lower it shadowed: that lower must still exist and match, or the
        // host has moved on — deletion is as much a host-side change as an
        // edit, and recreating the file would overrule it. A guest-created
        // file has no origin and lands only where the host still has
        // nothing; a host entry that appeared since is equally a conflict.
        // Lookup failures other than NotFound are I/O errors, never
        // permission to apply.
        let host_md = match fs::symlink_metadata(host) {
            Ok(md) => Some(md),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        match (origin(&change.upper).map_err(errno_to_io)?, host_md) {
            (Some(o), Some(host_md)) => {
                let current = Origin {
                    dev: host_md.dev(),
                    ino: host_md.ino(),
                    size: host_md.size(),
                    mtime_sec: host_md.mtime(),
                    mtime_nsec: host_md.mtime_nsec(),
                };
                if current != o {
                    return Err(ApplyError::Conflict);
                }
            }
            (Some(_), None) | (None, Some(_)) => return Err(ApplyError::Conflict),
            (None, None) => {}
        }
        if let Some(parent) = host.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&change.upper, host)?;
        replay_metadata(&md, &change.upper, host, replayable_xattr)?;
        // Advance the origin to the exact host identity this apply produced
        // (after the metadata replay, whose timestamps are part of it): the
        // rerun then recognizes its own work as applied, and anything the
        // host does to the file afterward as a conflict. A failure between
        // the copy and here leaves the stale origin, which the rerun
        // reports as a conflict — never a silent overwrite.
        let applied = fs::symlink_metadata(host)?;
        record_origin(
            &change.upper,
            &Origin {
                dev: applied.dev(),
                ino: applied.ino(),
                size: applied.size(),
                mtime_sec: applied.mtime(),
                mtime_nsec: applied.mtime_nsec(),
            },
        )
        .map_err(errno_to_io)?;
        return Ok(());
    }

    if md.file_type().is_symlink() {
        let target = fs::read_link(&change.upper)?;
        remove_host(host)?;
        std::os::unix::fs::symlink(target, host)?;
        replay_metadata(&md, &change.upper, host, replayable_xattr)?;
        return Ok(());
    }

    if md.file_type().is_fifo() {
        let chost = cpath(host)?;
        remove_host(host)?;
        if unsafe { libc::mkfifo(chost.as_ptr(), (md.mode() & 0o7777) as libc::mode_t) } != 0 {
            return Err(io::Error::last_os_error().into());
        }
        replay_metadata(&md, &change.upper, host, replayable_xattr)?;
        return Ok(());
    }

    // Sockets and device nodes have no meaningful "apply".
    eprintln!("chimera: skipping {} (special file)", change.path.display());
    Ok(())
}

pub(super) fn copy_xattrs(
    upper: &std::ffi::CStr,
    host: &std::ffi::CStr,
    keep: fn(&[u8]) -> bool,
) -> io::Result<()> {
    let mut names = vec![0u8; 1024];
    let len = loop {
        let n = unsafe {
            libc::llistxattr(
                upper.as_ptr(),
                names.as_mut_ptr() as *mut libc::c_char,
                names.len(),
            )
        };
        if n >= 0 {
            break n as usize;
        }
        match io::Error::last_os_error() {
            e if e.raw_os_error() == Some(libc::ERANGE) => names.resize(names.len() * 2, 0),
            // A delta filesystem without xattrs holds none to replay.
            e if e.raw_os_error() == Some(libc::ENOTSUP) => return Ok(()),
            e => return Err(e),
        }
    };
    for name in names[..len].split(|&b| b == 0).filter(|n| !n.is_empty()) {
        if !keep(name) {
            continue;
        }
        let cname =
            std::ffi::CString::new(name).map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;
        let mut value = vec![0u8; 256];
        let vlen = loop {
            let n = unsafe {
                libc::lgetxattr(
                    upper.as_ptr(),
                    cname.as_ptr(),
                    value.as_mut_ptr() as *mut libc::c_void,
                    value.len(),
                )
            };
            if n >= 0 {
                break n as usize;
            }
            match io::Error::last_os_error() {
                e if e.raw_os_error() == Some(libc::ERANGE) => value.resize(value.len() * 2, 0),
                e => return Err(e),
            }
        };
        if unsafe {
            libc::lsetxattr(
                host.as_ptr(),
                cname.as_ptr(),
                value.as_ptr() as *const libc::c_void,
                vlen,
                0,
            )
        } != 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}
