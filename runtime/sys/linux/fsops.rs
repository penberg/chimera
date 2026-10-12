//! The Linux spelling of the host primitives the delta format is written
//! against: extended attributes, the atomic rename variants, and bulk data
//! transfer. See `crate::sys::fs::delta` for what each is used for.

use std::ffi::CStr;

use crate::sys::fs::vfs::Errno;

/// The errno a missing extended attribute reports.
pub const ENOATTR: i32 = libc::ENODATA;

/// A stat's device number as the delta format records it; `dev_t` is a
/// different width on each host.
pub fn dev_of(st: &libc::stat) -> u64 {
    st.st_dev
}

/// Read the xattr `name` of `path` without following a final symlink. An
/// empty `buf` queries the size.
pub fn lgetxattr(path: &CStr, name: &CStr, buf: &mut [u8]) -> Result<usize, Errno> {
    let n = unsafe {
        libc::lgetxattr(
            path.as_ptr(),
            name.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            buf.len(),
        )
    };
    size(n)
}

/// Set the xattr `name` of `path` without following a final symlink.
pub fn lsetxattr(path: &CStr, name: &CStr, value: &[u8]) -> Result<(), Errno> {
    check(unsafe {
        libc::lsetxattr(
            path.as_ptr(),
            name.as_ptr(),
            value.as_ptr() as *const libc::c_void,
            value.len(),
            0,
        )
    })
}

/// List the xattr names of `path` without following a final symlink, as
/// NUL-terminated names; an empty `buf` queries the size.
pub fn llistxattr(path: &CStr, buf: &mut [u8]) -> Result<usize, Errno> {
    let n = unsafe {
        libc::llistxattr(
            path.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_char,
            buf.len(),
        )
    };
    size(n)
}

pub fn fgetxattr(fd: libc::c_int, name: &CStr, buf: &mut [u8]) -> Result<usize, Errno> {
    let n = unsafe {
        libc::fgetxattr(
            fd,
            name.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            buf.len(),
        )
    };
    size(n)
}

pub fn fsetxattr(fd: libc::c_int, name: &CStr, value: &[u8]) -> Result<(), Errno> {
    check(unsafe {
        libc::fsetxattr(
            fd,
            name.as_ptr(),
            value.as_ptr() as *const libc::c_void,
            value.len(),
            0,
        )
    })
}

pub fn flistxattr(fd: libc::c_int, buf: &mut [u8]) -> Result<usize, Errno> {
    let n = unsafe { libc::flistxattr(fd, buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    size(n)
}

/// `rename(2)` that refuses to replace an existing destination — the
/// compare-and-swap a first-wins publication needs.
pub fn rename_noreplace(from: &CStr, to: &CStr) -> Result<(), Errno> {
    check(unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            from.as_ptr(),
            libc::AT_FDCWD,
            to.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    })
}

/// `rename(2)` that atomically swaps two existing entries — how a staged
/// object of any type takes a whiteout marker's place in one step.
pub fn rename_exchange(a: &CStr, b: &CStr) -> Result<(), Errno> {
    check(unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            a.as_ptr(),
            libc::AT_FDCWD,
            b.as_ptr(),
            libc::RENAME_EXCHANGE,
        )
    })
}

/// The lower attributes a copy-up carries and an apply replays: the
/// guest-visible user namespace and the POSIX ACLs. The reserved
/// `user.chimera.*` names never come along — a lower file carrying one,
/// forged or stray, must not become delta bookkeeping — and kernel-managed
/// namespaces (`security.*`) belong to the upper file's own filesystem.
pub fn carried_xattr(name: &[u8]) -> bool {
    (name.starts_with(b"user.") && !name.starts_with(crate::sys::fs::delta::XATTR_NAMESPACE))
        || name == b"system.posix_acl_access"
        || name == b"system.posix_acl_default"
}

/// Copy `src`'s contents to `dst` (both at offset 0, `dst` freshly created):
/// `FICLONE` shares extents in one ioctl where the filesystem supports
/// reflinks; `copy_file_range` copies in-kernel otherwise; a read/write loop
/// is the portable floor. The fallbacks trigger only when no bytes have
/// moved, so each strategy starts from a clean destination.
pub fn transfer(src: libc::c_int, dst: libc::c_int) -> Result<(), Errno> {
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
    copy_loop(src, dst)
}

/// The portable transfer: read and write until EOF.
pub fn copy_loop(src: libc::c_int, dst: libc::c_int) -> Result<(), Errno> {
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

fn size(n: isize) -> Result<usize, Errno> {
    if n < 0 {
        Err(last_errno())
    } else {
        Ok(n as usize)
    }
}

fn check(ret: libc::c_int) -> Result<(), Errno> {
    if ret < 0 { Err(last_errno()) } else { Ok(()) }
}

fn last_errno() -> Errno {
    Errno::from_io(&std::io::Error::last_os_error())
}
