//! The `#!` line an exec target may carry. Both hosts read it the same way —
//! a bounded probe of the image's leading bytes, the window the kernel
//! inspects to pick a loader — and differ only in how the text after the
//! interpreter becomes argv, which is left to the caller.

use std::{fs, io::Read, path::Path};

use crate::Error;

/// A parsed shebang line: the interpreter path, then the rest of the line
/// after it. Splitting that remainder into argv entries is the host's rule.
pub type ShebangLine = (Vec<u8>, Vec<u8>);

/// Parse `path`'s shebang line, probing at most `probe` leading bytes.
/// `Ok(None)` when the image is not a `#!` script; otherwise the interpreter
/// path and the rest of the line after it, trimmed of surrounding blanks.
/// Kernel semantics throughout: the line ends at the first newline and must
/// fit the probe, the interpreter ends at the first blank, and a shebang
/// naming no interpreter is a malformed image, reported as `ENOEXEC`.
pub fn shebang_line(path: &Path, probe: usize) -> Result<Option<ShebangLine>, Error> {
    let mut file = fs::File::open(path).map_err(|e| Error::io("execve", e))?;
    let mut buf = vec![0u8; probe];
    let mut len = 0;
    while len < buf.len() {
        match file
            .read(&mut buf[len..])
            .map_err(|e| Error::io("execve", e))?
        {
            0 => break,
            n => len += n,
        }
    }
    let buf = &buf[..len];
    if len < 2 || &buf[..2] != b"#!" {
        return Ok(None);
    }
    let line = match buf.iter().position(|&b| b == b'\n') {
        Some(nl) => &buf[2..nl],
        None if len < probe => &buf[2..],
        None => {
            return Err(Error::BadBinary(
                "shebang line does not fit the interpreter probe buffer".into(),
            ));
        }
    };
    let line = &line[..line.len() - line.iter().rev().take_while(|&&b| blank(b)).count()];
    let rest = &line[line.iter().take_while(|&&b| blank(b)).count()..];
    if rest.is_empty() {
        return Err(Error::BadBinary("shebang names no interpreter".into()));
    }
    let end = rest.iter().position(|&b| blank(b)).unwrap_or(rest.len());
    let interp = rest[..end].to_vec();
    let after = &rest[end..];
    let args = after[after.iter().take_while(|&&b| blank(b)).count()..].to_vec();
    Ok(Some((interp, args)))
}

/// The blanks a shebang line separates its words with; the kernel recognizes
/// no others.
pub fn blank(b: u8) -> bool {
    b == b' ' || b == b'\t'
}
