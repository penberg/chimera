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
