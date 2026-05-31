//! Host-OS-specific support code. Each submodule is gated on the matching
//! `target_os`; an unsupported host yields a compile error at the use site.

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_arch = "x86_64")]
pub mod mmap;

#[cfg(target_os = "linux")]
pub use linux::exec;

#[cfg(not(any(target_os = "linux")))]
pub mod exec {
    use std::{ffi::OsString, path::Path};

    use crate::{Error, SystemCalls};

    pub fn execv(
        _program: &Path,
        _args: &[OsString],
        _envs: Option<&[(OsString, OsString)]>,
        _handler: Box<dyn SystemCalls>,
    ) -> Result<i32, Error> {
        Err(Error::UnsupportedHost {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
        })
    }
}
