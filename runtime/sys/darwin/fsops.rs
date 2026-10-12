//! The Darwin spelling of the host primitives the delta format is written
//! against: extended attributes, the atomic rename variants, and bulk data
//! transfer. See `crate::sys::fs::delta` for what each is used for.
//!
//! Darwin's xattr calls take a `position` (meaningful only for the resource
//! fork) and an options word, where `XATTR_NOFOLLOW` replaces Linux's `l*`
//! variants; `renamex_np` carries the exclusive and swap flags `renameat2`
//! carries on Linux; and `fcopyfile` is the kernel-assisted copy, cloning on
//! APFS where it can.

use std::ffi::CStr;

use crate::sys::fs::{delta::XATTR_NAMESPACE, vfs::Errno};

/// The errno a missing extended attribute reports.
pub const ENOATTR: i32 = libc::ENOATTR;

/// A stat's device number as the delta format records it; `dev_t` is a
/// different width on each host.
pub fn dev_of(st: &libc::stat) -> u64 {
    st.st_dev as u64
}

/// A size query is a NULL buffer here, where Linux takes any pointer with
/// a zero length; a non-NULL zero-length buffer is `ERANGE`.
fn query_or(buf: &mut [u8]) -> *mut libc::c_void {
    if buf.is_empty() {
        std::ptr::null_mut()
    } else {
        buf.as_mut_ptr() as *mut libc::c_void
    }
}

/// Read the xattr `name` of `path` without following a final symlink. An
/// empty `buf` queries the size.
pub fn lgetxattr(path: &CStr, name: &CStr, buf: &mut [u8]) -> Result<usize, Errno> {
    let n = unsafe {
        libc::getxattr(
            path.as_ptr(),
            name.as_ptr(),
            query_or(buf),
            buf.len(),
            0,
            libc::XATTR_NOFOLLOW,
        )
    };
    size(n)
}

/// Set the xattr `name` of `path` without following a final symlink.
pub fn lsetxattr(path: &CStr, name: &CStr, value: &[u8]) -> Result<(), Errno> {
    check(unsafe {
        libc::setxattr(
            path.as_ptr(),
            name.as_ptr(),
            value.as_ptr() as *const libc::c_void,
            value.len(),
            0,
            libc::XATTR_NOFOLLOW,
        )
    })
}

/// List the xattr names of `path` without following a final symlink, as
/// NUL-terminated names; an empty `buf` queries the size.
pub fn llistxattr(path: &CStr, buf: &mut [u8]) -> Result<usize, Errno> {
    let n = unsafe {
        libc::listxattr(
            path.as_ptr(),
            query_or(buf) as *mut libc::c_char,
            buf.len(),
            libc::XATTR_NOFOLLOW,
        )
    };
    size(n)
}

pub fn fgetxattr(fd: libc::c_int, name: &CStr, buf: &mut [u8]) -> Result<usize, Errno> {
    let n = unsafe { libc::fgetxattr(fd, name.as_ptr(), query_or(buf), buf.len(), 0, 0) };
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
            0,
        )
    })
}

pub fn flistxattr(fd: libc::c_int, buf: &mut [u8]) -> Result<usize, Errno> {
    let n = unsafe { libc::flistxattr(fd, query_or(buf) as *mut libc::c_char, buf.len(), 0) };
    size(n)
}

/// `rename(2)` that refuses to replace an existing destination — the
/// compare-and-swap a first-wins publication needs.
pub fn rename_noreplace(from: &CStr, to: &CStr) -> Result<(), Errno> {
    check(unsafe { libc::renamex_np(from.as_ptr(), to.as_ptr(), libc::RENAME_EXCL) })
}

/// `rename(2)` that atomically swaps two existing entries — how a staged
/// object of any type takes a whiteout marker's place in one step.
pub fn rename_exchange(a: &CStr, b: &CStr) -> Result<(), Errno> {
    check(unsafe { libc::renamex_np(a.as_ptr(), b.as_ptr(), libc::RENAME_SWAP) })
}

/// The lower attributes a copy-up carries and an apply replays. Darwin has
/// one flat namespace, so everything comes along except the reserved
/// `user.chimera.*` names — a lower file carrying one, forged or stray, must
/// not become delta bookkeeping — and the kernel-owned provenance record,
/// which the system stamps itself and refuses to have written.
pub fn carried_xattr(name: &[u8]) -> bool {
    !name.starts_with(XATTR_NAMESPACE) && name != b"com.apple.provenance"
}

/// Copy `src`'s contents to `dst` (both at offset 0, `dst` freshly created):
/// `fcopyfile` lets the kernel clone or copy the data; a read/write loop is
/// the portable floor, taken only when no bytes have moved.
pub fn transfer(src: libc::c_int, dst: libc::c_int) -> Result<(), Errno> {
    if unsafe { libc::fcopyfile(src, dst, std::ptr::null_mut(), libc::COPYFILE_DATA) } == 0 {
        return Ok(());
    }
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstat(dst, &mut st) } == 0 && st.st_size != 0 {
        return Err(last_errno());
    }
    if unsafe { libc::lseek(src, 0, libc::SEEK_SET) } < 0 {
        return Err(last_errno());
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
