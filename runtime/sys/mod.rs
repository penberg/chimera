//! Host-OS-specific support code. Each submodule is gated on the matching
//! `target_os`; an unsupported host yields a compile error at the use site.

#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_arch = "x86_64")]
pub mod mmap;

#[cfg(target_os = "linux")]
pub use linux::exec;
