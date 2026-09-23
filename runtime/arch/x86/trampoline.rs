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
//!
//! `trampoline.S` also defines `chimera_fetch_copy`, the fault-guarded copy
//! behind [`fetch_copy`]: the translator's per-block decode-window fetch,
//! which must tolerate an unreadable guest address (a wild jump) without
//! faulting the runtime.

use std::{arch::global_asm, mem::offset_of};

use super::dispatch::{EXIT_KIND_SYSCALL, EXIT_KIND_TRAP, ThreadState};

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
    TS_FPSTATE     = const offset_of!(ThreadState, fpstate),
    TS_FP_IN_REGS  = const offset_of!(ThreadState, fp_in_regs),
    TS_FS_IS_GUEST = const offset_of!(ThreadState, fs_is_guest),
    TS_CHIMERA_FS  = const offset_of!(ThreadState, chimera_fs_base),
    TS_EXIT_REQUESTED = const offset_of!(ThreadState, exit_requested),
    EXIT_KIND_SYSCALL = const EXIT_KIND_SYSCALL,
    EXIT_KIND_TRAP = const EXIT_KIND_TRAP,
);

unsafe extern "C" {
    pub fn dispatch(ctx: *mut ThreadState, host_pc: u64);
    fn dispatch_host_pc_stashed();
    fn dispatch_end();
    pub fn exit_block();
    fn exit_now();
    pub fn exit_syscall();
    pub fn exit_trap();
    fn chimera_fetch_copy(dst: *mut u8, src: *const u8, len: usize) -> usize;
    fn chimera_fetch_copy_start();
    fn chimera_fetch_fixup();
    pub fn guarded_copy(dst: *mut u8, src: *const u8, len: usize) -> i64;
    fn guarded_copy_fault();
    fn guarded_copy_end();
}

/// The host-rip layout of `dispatch` for the preemption code: the routine's
/// bounds `[lo, hi)` and the label past the `TS_HOST_PC` stash. A signal
/// caught at a rip in `[lo, stashed)` still has the host PC in rsi; one caught
/// in `[stashed, hi)` has it in `TS_HOST_PC`.
pub struct DispatchSpan {
    pub lo: usize,
    pub stashed: usize,
    pub hi: usize,
}

pub fn dispatch_span() -> DispatchSpan {
    DispatchSpan {
        lo: dispatch as *const () as usize,
        stashed: dispatch_host_pc_stashed as *const () as usize,
        hi: dispatch_end as *const () as usize,
    }
}

/// Host address of `exit_now`, the block exit an interrupted thread is aimed
/// at: it stashes rax and runs `exit_block` with the live guest registers.
pub fn exit_now_addr() -> usize {
    exit_now as *const () as usize
}

/// Copy up to `buf.len()` bytes of guest memory at `src` into `buf` without
/// trusting the address. A direct `rep movsb` whose fault the handler fixes
/// up ([`fetch_copy_span`]), so the common case — every byte readable — costs
/// a memcpy, not a syscall. Returns the byte-exact readable prefix length:
/// `buf.len()` on a full copy, 0 when `src` itself is unreadable. Callers run
/// only after [`crate::sys::linux::fault::install`]; without the handler an
/// unreadable byte is a fatal fault.
pub fn fetch_copy(src: u64, buf: &mut [u8]) -> usize {
    unsafe { chimera_fetch_copy(buf.as_mut_ptr(), src as *const u8, buf.len()) }
}

/// Host-rip bounds `[start, fixup)` of `chimera_fetch_copy`'s faultable span.
/// A `SIGSEGV`/`SIGBUS` whose rip lies inside resumes at `fixup`, which
/// returns the partial count.
pub fn fetch_copy_span() -> (usize, usize) {
    (
        chimera_fetch_copy_start as *const () as usize,
        chimera_fetch_fixup as *const () as usize,
    )
}

/// Whether `rip` lies inside [`guarded_copy`], so a fault taken there is a
/// failed read of guest memory rather than a crash in Chimera.
pub fn in_guarded_copy(rip: usize) -> bool {
    let lo = guarded_copy as *const () as usize;
    let hi = guarded_copy_end as *const () as usize;
    (lo..hi).contains(&rip)
}

/// Where the fault handler resumes a faulted [`guarded_copy`]: the routine's
/// failure tail, which returns 0 to the caller.
pub fn guarded_copy_fixup() -> usize {
    guarded_copy_fault as *const () as usize
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    const PAGE: usize = 4096;

    #[test]
    fn fetch_copy_returns_readable_prefix() {
        crate::sys::linux::fault::install();

        // A readable page followed by a PROT_NONE page, so a copy can start
        // readable and fault partway through.
        let map = unsafe {
            libc::mmap(
                ptr::null_mut(),
                PAGE * 2,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(map, libc::MAP_FAILED);
        let base = map as usize;
        unsafe {
            ptr::write_bytes(map as *mut u8, 0xab, PAGE);
            let ret = libc::mprotect((base + PAGE) as *mut libc::c_void, PAGE, libc::PROT_NONE);
            assert_eq!(ret, 0);
        }

        let mut buf = [0u8; 64];
        assert_eq!(fetch_copy(base as u64, &mut buf), buf.len());
        assert!(buf.iter().all(|&b| b == 0xab));

        let mut buf = [0u8; 64];
        assert_eq!(fetch_copy((base + PAGE - 24) as u64, &mut buf), 24);
        assert!(buf[..24].iter().all(|&b| b == 0xab));
        assert!(buf[24..].iter().all(|&b| b == 0));

        let mut buf = [0u8; 64];
        assert_eq!(fetch_copy((base + PAGE) as u64, &mut buf), 0);

        unsafe { libc::munmap(map, PAGE * 2) };
    }
}
