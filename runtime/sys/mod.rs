//! Host-OS-specific support code. Each submodule is gated on the matching
//! `target_os`; an unsupported host yields a compile error at the use site.

#[cfg(target_os = "linux")]
pub mod linux;

pub mod fs;
#[cfg(target_arch = "x86_64")]
pub mod mmap;
pub mod shebang;

/// The host's passthrough filesystem and the host primitives the delta
/// format is written against, re-exported host-neutrally so the shared
/// filesystem layer names `sys::hostfs` and `sys::fsops`.
#[cfg(target_os = "linux")]
pub use linux::{fsops, hostfs};

#[cfg(target_os = "linux")]
pub use linux::exec;
