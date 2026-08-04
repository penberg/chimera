//! The Darwin arm64 raw-syscall bridge. This is the one place Chimera actually
//! reaches the host kernel on Darwin; the interception policy around it lives in
//! [`crate::sys::darwin::policy`].

use std::arch::asm;

use crate::{
    SyscallResult, SystemCall,
    arch::dispatch::{ThreadState, X0},
};

/// Bit position of the carry flag inside NZCV (bit 29).
const NZCV_CARRY: u64 = 1 << 29;

/// Issue the host kernel's `svc #0x80` with `call`'s number in `x16` and its
/// arguments in `x0..x5`. Darwin reports failure in the carry flag rather than
/// with a negative return value, so the carry is captured with `cset` and
/// decoded into a [`SyscallResult`] whose `Error` carries the positive errno
/// the kernel left in `x0`.
pub fn host_syscall(call: &SystemCall) -> SyscallResult {
    host_syscall2(call).0
}

/// Like [`host_syscall`], but also return the kernel's second return register.
/// Darwin writes `retval[1]` into `x1` on every syscall return — usually zero,
/// but a pair-returning syscall like `pipe(2)` puts its second value there
/// (which is also why `x1` must be an output of the `asm!` block: the kernel
/// clobbers it unconditionally).
pub fn host_syscall2(call: &SystemCall) -> (SyscallResult, u64) {
    let ret: u64;
    let ret1: u64;
    let carry: u64;
    unsafe {
        asm!(
            "svc #0x80",
            "cset {carry}, cs",
            in("x16") call.number,
            inout("x0") call.args[0] => ret,
            inout("x1") call.args[1] => ret1,
            in("x2") call.args[2],
            in("x3") call.args[3],
            in("x4") call.args[4],
            in("x5") call.args[5],
            in("x6") call.args[6],
            in("x7") call.args[7],
            carry = lateout(reg) carry,
            options(nostack),
        );
    }
    if carry != 0 {
        (SyscallResult::Error(ret as i32), ret1)
    } else {
        (SyscallResult::Ok(ret as i64), ret1)
    }
}

/// Commit a serviced syscall's result to the guest register file in the Darwin
/// arm64 ABI: the value goes in `x0` and failure is signalled by setting the
/// NZCV carry bit (with the positive errno in `x0`), the way the guest's libc
/// reads it on resume. The dispatcher calls this once per serviced syscall.
pub fn write_syscall_result(state: &mut ThreadState, call: &SystemCall) {
    match call.result() {
        Some(SyscallResult::Ok(value)) => {
            state.regs[X0] = value as u64;
            state.nzcv &= !NZCV_CARRY;
        }
        Some(SyscallResult::Error(errno)) => {
            state.regs[X0] = errno as u64;
            state.nzcv |= NZCV_CARRY;
        }
        // No result to commit (e.g. an intercept that set nothing); leave x0.
        None => {}
    }
}
