// Trampolines that cross between Chimera and translated guest code.
//
// Three entry points: `dispatch` is called from Rust with
// `(ctx, host_pc)` and jumps into the code cache; `exit_block` is the
// tail of every per-block exit stub and returns to the Rust caller of
// `dispatch` with `exit_kind = BLOCK`; `exit_syscall` does the same, but
// sets `exit_kind = SYSCALL` so the dispatcher knows to invoke the
// embedder's `SystemCalls` handler before re-entering the cache.
//
// No GPR is reserved across translated code. Translated blocks find their
// ThreadState through a per-thread pthread TSD slot (read via TPIDRRO_EL0 —
// see translate::emit_load_ctx), so nothing needs a reserved register. Apple's
// arm64 platform reserves x18 for the kernel, which on macOS may be silently
// zeroed on signal delivery, which is one reason the ctx lives in the TSD, not
// a register.
//
// All TS_* placeholders are field offsets into ThreadState, supplied by
// global_asm! via offset_of!. Update the struct and the asm picks up the
// new offsets at compile time — no magic numbers to keep in sync.

.text
.p2align 2

.globl _dispatch
.globl _exit_block
.globl _exit_syscall
.globl _exit_syscall_no_stack

// fn dispatch(ctx: *mut ThreadState /* x0 */, host_pc: u64 /* x1 */)
//
// Saves Chimera's callee-saves on Chimera's stack, restores the guest's NZCV /
// FPSIMD / GPRs, switches `sp` to the guest stack, and branches to `host_pc`.
_dispatch:
    // Save Chimera callee-saves on the Chimera stack.
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!
    stp     x29, x30, [sp, #-16]!

    // Save Chimera sp and stash host_pc target in ctx.
    mov     x9, sp
    str     x9, [x0, #{TS_CHIMERA_SP}]
    str     x1, [x0, #{TS_HOST_PC}]

    // Restore guest FPSIMD (32 Q-regs + FPSR + FPCR), using x0 as base.
    ldp     q0,  q1,  [x0, #({TS_FPSTATE} + 0)]
    ldp     q2,  q3,  [x0, #({TS_FPSTATE} + 32)]
    ldp     q4,  q5,  [x0, #({TS_FPSTATE} + 64)]
    ldp     q6,  q7,  [x0, #({TS_FPSTATE} + 96)]
    ldp     q8,  q9,  [x0, #({TS_FPSTATE} + 128)]
    ldp     q10, q11, [x0, #({TS_FPSTATE} + 160)]
    ldp     q12, q13, [x0, #({TS_FPSTATE} + 192)]
    ldp     q14, q15, [x0, #({TS_FPSTATE} + 224)]
    ldp     q16, q17, [x0, #({TS_FPSTATE} + 256)]
    ldp     q18, q19, [x0, #({TS_FPSTATE} + 288)]
    ldp     q20, q21, [x0, #({TS_FPSTATE} + 320)]
    ldp     q22, q23, [x0, #({TS_FPSTATE} + 352)]
    ldp     q24, q25, [x0, #({TS_FPSTATE} + 384)]
    ldp     q26, q27, [x0, #({TS_FPSTATE} + 416)]
    ldp     q28, q29, [x0, #({TS_FPSTATE} + 448)]
    ldp     q30, q31, [x0, #({TS_FPSTATE} + 480)]
    ldr     w9, [x0, #({TS_FPSTATE} + 512)]
    msr     fpsr, x9
    ldr     w9, [x0, #({TS_FPSTATE} + 516)]
    msr     fpcr, x9

    // Restore guest NZCV.
    ldr     x9, [x0, #{TS_NZCV}]
    msr     nzcv, x9

    // Switch to the guest stack.
    ldr     x9, [x0, #{TS_SP}]
    mov     sp, x9

    // Load guest GPRs from regs[]. x0 stays as the ctx base until the end;
    // we load x16 last because it carries host_pc into the branch.
    ldr     x1,  [x0, #({TS_REGS} + 8)]
    ldp     x2,  x3,  [x0, #({TS_REGS} + 16)]
    ldp     x4,  x5,  [x0, #({TS_REGS} + 32)]
    ldp     x6,  x7,  [x0, #({TS_REGS} + 48)]
    ldp     x8,  x9,  [x0, #({TS_REGS} + 64)]
    ldp     x10, x11, [x0, #({TS_REGS} + 80)]
    ldp     x12, x13, [x0, #({TS_REGS} + 96)]
    ldp     x14, x15, [x0, #({TS_REGS} + 112)]
    ldr     x17, [x0, #({TS_REGS} + 136)]
    ldp     x18, x19, [x0, #({TS_REGS} + 144)]
    ldp     x20, x21, [x0, #({TS_REGS} + 160)]
    ldp     x22, x23, [x0, #({TS_REGS} + 176)]
    ldp     x24, x25, [x0, #({TS_REGS} + 192)]
    ldp     x26, x27, [x0, #({TS_REGS} + 208)]
    ldp     x28, x29, [x0, #({TS_REGS} + 224)]
    ldr     x30, [x0, #({TS_REGS} + 240)]
    // x16 = host_pc; x0 = guest x0; branch.
    ldr     x16, [x0, #{TS_HOST_PC}]
    ldr     x0,  [x0, #({TS_REGS} + 0)]
    br      x16
    // x16 has been clobbered; every translated block's prologue reloads it.

// Common exit-trampoline tail. Reached after the per-block exit stub has:
//   - pushed guest x16/x17 to the guest stack (the very first thing it does)
//   - written next_pc to ctx.pc
//   - loaded the ctx pointer into x17
//   - branched here via x16 (so x16 is also free)
//
// Pre-conditions on entry:
//   - x0..x15, x18..x30 hold guest values
//   - sp = guest_sp - 16, with guest x16 at [sp, #0] and guest x17 at [sp, #8]
//   - x16 = scratch (was the trampoline target), x17 = ctx pointer
.macro EXIT_TAIL_COMMON
    // Save x0..x15 directly via the ctx base in x17.
    stp     x0,  x1,  [x17, #({TS_REGS} + 0)]
    stp     x2,  x3,  [x17, #({TS_REGS} + 16)]
    stp     x4,  x5,  [x17, #({TS_REGS} + 32)]
    stp     x6,  x7,  [x17, #({TS_REGS} + 48)]
    stp     x8,  x9,  [x17, #({TS_REGS} + 64)]
    stp     x10, x11, [x17, #({TS_REGS} + 80)]
    stp     x12, x13, [x17, #({TS_REGS} + 96)]
    stp     x14, x15, [x17, #({TS_REGS} + 112)]
    // Save x18..x30.
    stp     x18, x19, [x17, #({TS_REGS} + 144)]
    stp     x20, x21, [x17, #({TS_REGS} + 160)]
    stp     x22, x23, [x17, #({TS_REGS} + 176)]
    stp     x24, x25, [x17, #({TS_REGS} + 192)]
    stp     x26, x27, [x17, #({TS_REGS} + 208)]
    stp     x28, x29, [x17, #({TS_REGS} + 224)]
    str     x30, [x17, #({TS_REGS} + 240)]
    // Pop the pushed guest x16/x17 off the guest stack and store them in
    // their proper slots.
    ldp     x0,  x1,  [sp], #16
    stp     x0,  x1,  [x17, #({TS_REGS} + 128)]
    // Save guest sp (now back to its pre-push value).
    mov     x0, sp
    str     x0, [x17, #{TS_SP}]
    // Save NZCV.
    mrs     x0, nzcv
    str     x0, [x17, #{TS_NZCV}]
    // Save FPSIMD.
    stp     q0,  q1,  [x17, #({TS_FPSTATE} + 0)]
    stp     q2,  q3,  [x17, #({TS_FPSTATE} + 32)]
    stp     q4,  q5,  [x17, #({TS_FPSTATE} + 64)]
    stp     q6,  q7,  [x17, #({TS_FPSTATE} + 96)]
    stp     q8,  q9,  [x17, #({TS_FPSTATE} + 128)]
    stp     q10, q11, [x17, #({TS_FPSTATE} + 160)]
    stp     q12, q13, [x17, #({TS_FPSTATE} + 192)]
    stp     q14, q15, [x17, #({TS_FPSTATE} + 224)]
    stp     q16, q17, [x17, #({TS_FPSTATE} + 256)]
    stp     q18, q19, [x17, #({TS_FPSTATE} + 288)]
    stp     q20, q21, [x17, #({TS_FPSTATE} + 320)]
    stp     q22, q23, [x17, #({TS_FPSTATE} + 352)]
    stp     q24, q25, [x17, #({TS_FPSTATE} + 384)]
    stp     q26, q27, [x17, #({TS_FPSTATE} + 416)]
    stp     q28, q29, [x17, #({TS_FPSTATE} + 448)]
    stp     q30, q31, [x17, #({TS_FPSTATE} + 480)]
    mrs     x0, fpsr
    str     w0, [x17, #({TS_FPSTATE} + 512)]
    mrs     x0, fpcr
    str     w0, [x17, #({TS_FPSTATE} + 516)]
    // Restore Chimera sp + callee-saves and return to the Rust caller of
    // `dispatch`.
    ldr     x0, [x17, #{TS_CHIMERA_SP}]
    mov     sp, x0
    ldp     x29, x30, [sp], #16
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret
.endm

_exit_block:
    EXIT_TAIL_COMMON

// Re-entry point for a synchronous guest fault delivered to a guest handler.
// The Darwin fault handler (`sys::darwin::fault`) took a fault mid-block, saved
// the interrupted guest register file into `ctx` itself (from the signal's
// mcontext), and rewrote the interrupted context to resume here with x17 = ctx.
// Unlike `_exit_block` there is no guest state to save — it is already in
// `ctx` — so this is only the Chimera-side unwind: restore Chimera's stack and
// callee-saves (still parked by `dispatch`) and return to the caller of
// `dispatch`, exactly as an ordinary block exit would. The run loop then sees
// exit_kind = BLOCK (never changed since entry) and delivers the pending fault
// signal at the next boundary.
.globl _fault_resume
_fault_resume:
    ldr     x0, [x17, #{TS_CHIMERA_SP}]
    mov     sp, x0
    ldp     x29, x30, [sp], #16
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret

_exit_syscall:
    // Same flow as _exit_block, but record exit_kind = SYSCALL before
    // restoring chimera state. We must use x16 — already clobbered by the
    // exit stub — as the scratch; touching x0 here would overwrite the
    // guest's x0 *before* EXIT_TAIL_COMMON has a chance to save it,
    // causing every syscall to see x0 = 1 (the EXIT_KIND_SYSCALL value).
    mov     x16, #{EXIT_KIND_SYSCALL}
    str     x16, [x17, #{TS_EXIT_KIND}]
    EXIT_TAIL_COMMON

// Variant of _exit_syscall for SVC sites: assumes the per-block stub
// already saved guest x16 directly into `ctx.regs[16]` and did NOT push
// anything onto the guest stack. The kernel's SVC contract is "user
// stack untouched", and our translator must honour that — any guest that
// parks `sp` over a guard page (as the syscall-no-stack-touch
// conformance test does) would otherwise fault when our stub pushed.
//
// Pre-conditions on entry:
//   - x0..x15 hold guest values (x16 already saved by the stub).
//   - x17 = ctx pointer (set by the stub).
//   - x18..x30 hold guest values.
//   - sp = guest_sp (untouched).
//
// x17's guest value is discarded: x17 is an IP scratch register per the
// AArch64 PCS, and Darwin syscalls are free to clobber it.
_exit_syscall_no_stack:
    stp     x0,  x1,  [x17, #({TS_REGS} + 0)]
    stp     x2,  x3,  [x17, #({TS_REGS} + 16)]
    stp     x4,  x5,  [x17, #({TS_REGS} + 32)]
    stp     x6,  x7,  [x17, #({TS_REGS} + 48)]
    stp     x8,  x9,  [x17, #({TS_REGS} + 64)]
    stp     x10, x11, [x17, #({TS_REGS} + 80)]
    stp     x12, x13, [x17, #({TS_REGS} + 96)]
    stp     x14, x15, [x17, #({TS_REGS} + 112)]
    // x16: already saved by the stub. x17: not preserved (scratch).
    stp     x18, x19, [x17, #({TS_REGS} + 144)]
    stp     x20, x21, [x17, #({TS_REGS} + 160)]
    stp     x22, x23, [x17, #({TS_REGS} + 176)]
    stp     x24, x25, [x17, #({TS_REGS} + 192)]
    stp     x26, x27, [x17, #({TS_REGS} + 208)]
    stp     x28, x29, [x17, #({TS_REGS} + 224)]
    str     x30, [x17, #({TS_REGS} + 240)]
    mov     x0, sp
    str     x0, [x17, #{TS_SP}]
    mrs     x0, nzcv
    str     x0, [x17, #{TS_NZCV}]
    stp     q0,  q1,  [x17, #({TS_FPSTATE} + 0)]
    stp     q2,  q3,  [x17, #({TS_FPSTATE} + 32)]
    stp     q4,  q5,  [x17, #({TS_FPSTATE} + 64)]
    stp     q6,  q7,  [x17, #({TS_FPSTATE} + 96)]
    stp     q8,  q9,  [x17, #({TS_FPSTATE} + 128)]
    stp     q10, q11, [x17, #({TS_FPSTATE} + 160)]
    stp     q12, q13, [x17, #({TS_FPSTATE} + 192)]
    stp     q14, q15, [x17, #({TS_FPSTATE} + 224)]
    stp     q16, q17, [x17, #({TS_FPSTATE} + 256)]
    stp     q18, q19, [x17, #({TS_FPSTATE} + 288)]
    stp     q20, q21, [x17, #({TS_FPSTATE} + 320)]
    stp     q22, q23, [x17, #({TS_FPSTATE} + 352)]
    stp     q24, q25, [x17, #({TS_FPSTATE} + 384)]
    stp     q26, q27, [x17, #({TS_FPSTATE} + 416)]
    stp     q28, q29, [x17, #({TS_FPSTATE} + 448)]
    stp     q30, q31, [x17, #({TS_FPSTATE} + 480)]
    mrs     x0, fpsr
    str     w0, [x17, #({TS_FPSTATE} + 512)]
    mrs     x0, fpcr
    str     w0, [x17, #({TS_FPSTATE} + 516)]
    mov     x0, #{EXIT_KIND_SYSCALL}
    str     x0, [x17, #{TS_EXIT_KIND}]
    ldr     x0, [x17, #{TS_CHIMERA_SP}]
    mov     sp, x0
    ldp     x29, x30, [sp], #16
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16
    ret
