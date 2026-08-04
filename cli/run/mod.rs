//! Running a guest. Which backend serves `run` is a property of the host,
//! selected once here so nothing above this module needs to care.

#[cfg(target_os = "linux")]
mod overlay;
#[cfg(not(target_os = "linux"))]
#[path = "plain.rs"]
mod overlay;

pub use overlay::run;

use std::{
    ffi::OsString,
    io,
    path::{Path, PathBuf},
};

pub(super) fn read_shebang(path: &Path) -> Result<Option<(PathBuf, Vec<OsString>)>, io::Error> {
    let bytes = std::fs::read(path)?;
    if !bytes.starts_with(b"#!") {
        return Ok(None);
    }

    let line_end = bytes
        .iter()
        .position(|&b| b == b'\n')
        .unwrap_or(bytes.len());
    let line = bytes[2..line_end]
        .strip_suffix(b"\r")
        .unwrap_or(&bytes[2..line_end]);
    let line = String::from_utf8_lossy(line);
    let words = shlex::split(&line).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid shebang in {}", path.display()),
        )
    })?;
    let Some((interpreter, args)) = words.split_first() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("empty shebang in {}", path.display()),
        ));
    };

    Ok(Some((
        PathBuf::from(interpreter),
        args.iter().map(OsString::from).collect(),
    )))
}
