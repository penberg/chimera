//! Assembly trampolines that cross between Chimera and translated guest code.
//!
//! The three entry points are defined in `trampoline.S` and pulled in via
//! `global_asm!`. `dispatch` is called from Rust with `(ctx, host_pc)`; it
//! saves Chimera's callee-saved registers, switches `sp` to the guest stack,
//! loads the guest GPRs/NZCV/FPSIMD from `ctx`, and branches to `host_pc`.
//! `exit_block` is the common tail of every per-block exit stub; it saves
//! the guest state and returns to the caller of `dispatch` with
//! `exit_kind = BLOCK`. `exit_syscall` does the same but sets
//! `exit_kind = SYSCALL`, so the run loop knows to invoke the embedder's
//! `SystemCalls` handler before re-entering the cache.
//!
//! Field offsets and constants flow from Rust to assembly through
//! `global_asm!`'s `const` operands: every `{TS_*}` placeholder in
//! `trampoline.S` resolves to an `offset_of!` against `ThreadState`. Update
//! the struct and the asm picks up the new offsets at compile time — no
//! magic numbers to keep in sync.

use std::{arch::global_asm, mem::offset_of};

use super::dispatch::{EXIT_KIND_SYSCALL, ThreadState};

/// Byte offset of `ThreadState::regs[idx]`.
const fn reg_off(idx: usize) -> usize {
    offset_of!(ThreadState, regs) + idx * 8
}

global_asm!(
    include_str!("trampoline.S"),
    TS_REGS        = const reg_off(0),
    TS_FPSTATE     = const offset_of!(ThreadState, fpstate),
    TS_SP          = const offset_of!(ThreadState, sp),
    TS_NZCV        = const offset_of!(ThreadState, nzcv),
    TS_CHIMERA_SP  = const offset_of!(ThreadState, chimera_sp),
    TS_HOST_PC     = const offset_of!(ThreadState, host_pc_target),
    TS_EXIT_KIND   = const offset_of!(ThreadState, exit_kind),
    EXIT_KIND_SYSCALL = const EXIT_KIND_SYSCALL,
);

unsafe extern "C" {
    pub fn dispatch(ctx: *mut ThreadState, host_pc: u64);
    pub fn exit_block();
    pub fn exit_syscall_no_stack();
    /// Re-entry after a synchronous fault redirected into a guest handler: the
    /// fault handler saved the guest state into `ctx` and points the interrupted
    /// context here (with x17 = ctx) to unwind back to the run loop.
    pub fn fault_resume();
}
