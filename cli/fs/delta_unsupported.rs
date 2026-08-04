//! Stand-in for the delta half of the `fs` tooling on hosts that do not have
//! it. Reading and replaying a delta means reading and restoring the whiteout
//! and origin markers, which are xattrs; `list`, `rm` and `prune` are
//! directory work and are served by the real module either way.

use std::io;

fn unsupported() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "reading and applying a delta needs the xattrs only a Linux host has",
    )
}

pub fn diff(_selector: &str) -> io::Result<()> {
    Err(unsupported())
}

pub fn apply(_selector: &str) -> io::Result<()> {
    Err(unsupported())
}

/// A host without the delta format has no delta markers to replay, and the
/// portable copy path (`fs branch`) still needs to call this. Copying a
/// file's own extended attributes here would need the BSD spelling of the
/// xattr calls; nothing depends on it yet, so it is honestly a no-op rather
/// than a half-done imitation.
pub(super) fn copy_xattrs(
    _upper: &std::ffi::CStr,
    _host: &std::ffi::CStr,
    _keep: fn(&[u8]) -> bool,
) -> io::Result<()> {
    Ok(())
}
