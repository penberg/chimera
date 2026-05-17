//! Assembly trampolines that cross between Chimera and translated guest code.
//!
//! The three entry points are defined in `trampoline.S` and pulled in via
//! `global_asm!`. `dispatch` is called from Rust with `(ctx, host_pc)`; it
//! saves Chimera's callee-saved registers, switches `rsp` to the guest stack,
//! loads the guest GPRs and rflags from `ctx`, and jumps to `host_pc`.
//! `exit_block` is the common tail of every per-block exit stub; it saves the
//! guest state and returns to the caller of `dispatch` with
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
    TS_RAX         = const reg_off(0),
    TS_RBX         = const reg_off(1),
    TS_RCX         = const reg_off(2),
    TS_RDX         = const reg_off(3),
    TS_RSI         = const reg_off(4),
    TS_RDI         = const reg_off(5),
    TS_RBP         = const reg_off(6),
    TS_RSP         = const reg_off(7),
    TS_R8          = const reg_off(8),
    TS_R9          = const reg_off(9),
    TS_R10         = const reg_off(10),
    TS_R11         = const reg_off(11),
    TS_R12         = const reg_off(12),
    TS_R13         = const reg_off(13),
    TS_R14         = const reg_off(14),
    TS_R15         = const reg_off(15),
    TS_RFLAGS      = const offset_of!(ThreadState, rflags),
    TS_CHIMERA_RSP = const offset_of!(ThreadState, chimera_rsp),
    TS_HOST_PC     = const offset_of!(ThreadState, host_pc_target),
    TS_EXIT_KIND   = const offset_of!(ThreadState, exit_kind),
    TS_GUEST_FS    = const offset_of!(ThreadState, guest_fs_base),
    TS_FPSTATE     = const offset_of!(ThreadState, fpstate),
    TS_CHIMERA_FS  = const offset_of!(ThreadState, chimera_fs_base),
    EXIT_KIND_SYSCALL = const EXIT_KIND_SYSCALL,
);

unsafe extern "C" {
    pub fn dispatch(ctx: *mut ThreadState, host_pc: u64);
    pub fn exit_block();
    pub fn exit_syscall();
}
