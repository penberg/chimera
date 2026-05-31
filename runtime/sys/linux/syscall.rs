//! The Linux x86-64 raw-syscall bridge. This is the one place Chimera
//! actually reaches the host kernel; the dispatch logic around it lives in
//! [`crate::syscall`].

use std::arch::asm;

use crate::{SyscallResult, SystemCall};

/// Issue the host kernel's `syscall` instruction with `call`'s number in `rax`
/// and the six argument registers in Linux x86-64 syscall ABI order. Decodes
/// the kernel's negative-errno convention into a [`SyscallResult`] so callers
/// don't have to inspect the sign themselves.
pub fn host_syscall(call: &SystemCall) -> SyscallResult {
    let ret: i64;
    unsafe {
        asm!(
            "syscall",
            in("rax") call.number,
            in("rdi") call.args[0],
            in("rsi") call.args[1],
            in("rdx") call.args[2],
            in("r10") call.args[3],
            in("r8")  call.args[4],
            in("r9")  call.args[5],
            lateout("rax") ret,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack, preserves_flags),
        );
    }
    // Linux signals errors in the closed range `[-4095, -1]`; anything else
    // is a successful result (including "negative-looking" high addresses
    // `mmap` can hand back).
    if (-4095..0).contains(&ret) {
        SyscallResult::Error(-ret as i32)
    } else {
        SyscallResult::Ok(ret)
    }
}
