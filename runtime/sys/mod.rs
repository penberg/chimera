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

/// The host syscall policy — the driver the arch dispatcher calls per guest
/// syscall — re-exported host-neutrally so the crate-root `syscall` module
/// names `sys::policy::syscall` rather than a per-OS path.
#[cfg(target_os = "linux")]
pub use linux::policy;

/// The host raw-syscall bridge and the syscall-result writeback, re-exported
/// host-neutrally so the crate-root syscall driver, the public API, and the
/// dispatcher name `sys::host_syscall` / `sys::write_syscall_result` rather
/// than a per-OS path.
#[cfg(target_os = "linux")]
pub use linux::syscall::{host_syscall, write_syscall_result};
