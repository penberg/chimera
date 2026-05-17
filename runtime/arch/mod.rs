//! Host-architecture-specific support code. Each submodule is gated on the
//! matching `target_arch`; an unsupported host yields a compile error at the
//! use site. `dispatch` is re-exported here so callers can write
//! `crate::arch::dispatch` regardless of which arch is active. `trampoline`
//! and `translate` are internal details of each arch backend and stay
//! unexposed.

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "x86_64")]
pub use x86::dispatch;
