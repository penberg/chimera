//! Host-architecture-specific support code. Each submodule is gated on the
//! matching `target_arch`; an unsupported host yields a compile error at the
//! use site. `dispatch` and `cache` are re-exported here so callers can write
//! `crate::arch::dispatch` / `crate::arch::cache` regardless of which arch is
//! active. `trampoline` and `translate` are internal details of each arch
//! backend and stay unexposed.
//!
//! ## Backend contract
//!
//! A backend must expose the surface the host-neutral runtime reaches through
//! these re-exports. The pieces:
//!
//! - `cache::BlockCache` — the translated-block cache the guest address space
//!   ([`crate::sys::mmap::AddressSpace`]) owns. Must provide `new`, `resolve`
//!   (returning the block's host PC and the guest span to arm for SMC),
//!   `invalidate_page`, `invalidate_range`, and `reset`.
//! - `dispatch::ThreadState` — the guest register file. Its byte layout is
//!   consumed by the backend's own trampolines and exit stubs, but neutral code
//!   also reaches two atomics through it cross-thread — `tid` and the
//!   `exit_requested` safepoint slot (see [`crate::process::Process`]) — and the
//!   syscall writeback stores the return register via `regs`.
//! - `dispatch::Thread` — the per-thread driver the runtime loops on: `new`,
//!   `run() -> Result<ExitReason>`, `enter`, `addr_space()`, `process()`, and
//!   the `state`/`exit_code`/`running` fields. Its paired OS policy calls the
//!   rest of its surface (clone/spawn/signal helpers), which may differ per OS.
//! - `dispatch::ExitReason` — why a run returned: `Exited(code)` or `Execve`.
//! - `dispatch::{RAX, RSP, …}` register-index constants, `read_clone3_args`,
//!   and `CLONE_ARGS_SIZE_MAX`, named by the syscall policy.
//! - `init` — one-time host-capability checks run before the first `Sandbox`.

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "x86_64")]
pub use x86::cache;

#[cfg(target_arch = "x86_64")]
pub use x86::dispatch;

#[cfg(target_arch = "x86_64")]
pub use x86::init;

#[cfg(target_arch = "x86_64")]
pub use x86::mpk_enabled;

#[cfg(target_arch = "aarch64")]
pub mod arm64;

#[cfg(target_arch = "aarch64")]
pub use arm64::cache;

#[cfg(target_arch = "aarch64")]
pub use arm64::dispatch;

#[cfg(target_arch = "aarch64")]
pub use arm64::init;
