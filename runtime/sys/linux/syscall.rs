//! The Linux x86-64 raw-syscall bridge. This is the one place Chimera
//! actually reaches the host kernel; the intercept logic around it (the
//! syscalls Chimera handles itself instead of forwarding) lives in
//! [`crate::syscall`].

use std::arch::asm;

use crate::SystemCall;

/// Issue the host kernel's `syscall` instruction with `call`'s number in `rax`
/// and the six argument registers in Linux x86-64 syscall ABI order, and
/// return whatever the kernel leaves in `rax`. Linux signals errors as a
/// negative return value (`-errno`), not out-of-band, so the caller can use
/// the result directly.
pub fn host_syscall(call: &SystemCall) -> i64 {
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
    ret
}
