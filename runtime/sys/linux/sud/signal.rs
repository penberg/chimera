//! Guest signals, forwarded with two substitutions.
//!
//! The guest's handlers are installed on the host as the guest asked, since
//! native delivery into guest code is exactly this backend's model. Two
//! things cannot be forwarded as they are. The restorer a guest supplies sits
//! below the exempt floor, where its `rt_sigreturn` would trap into the
//! dispatch handler with no way to complete it, so Chimera substitutes a
//! trampoline in its own text. And `SIGSYS` must never be blocked — it *is*
//! the dispatch trap, so a guest that blocks it around a critical section
//! would have its next syscall kill the process with the signal's default
//! action instead of trapping — so every mask the guest supplies is filtered
//! before it reaches the kernel.
//!
//! The filtering is visible to the guest in one direction: a mask it reads
//! back reports `SIGSYS` unblocked, because it is. Mirroring the intended
//! mask, and delivering signals at a safepoint of Chimera's rather than
//! wherever the kernel lands them, is what the translating backend does and
//! what a complete implementation needs here.

use std::{mem, ptr};

use crate::{
    SyscallResult, SystemCall,
    sys::mmap::{copy_from_guest, copy_to_guest},
};

use super::super::{signal::KernelSigaction, syscall::host_syscall};

/// The bit `SIGSYS` occupies in a kernel `sigset_t` (bit `signo - 1`).
const SIGSYS_BIT: u64 = 1 << (libc::SIGSYS as u64 - 1);

/// `sa_flags` bit: the caller supplied a restorer.
const SA_RESTORER: u64 = 0x0400_0000;

// The signal-return trampoline handed to the kernel for every guest
// `rt_sigaction`: two instructions in Chimera's text, and therefore inside
// the exempt range.
std::arch::global_asm!(
    ".globl chimera_sud_restorer",
    "chimera_sud_restorer:",
    "mov eax, 15", // SYS_rt_sigreturn
    "syscall",
    "ud2",
);
unsafe extern "C" {
    fn chimera_sud_restorer();
}

fn read_guest_sigset(ptr: u64) -> Option<u64> {
    let mut raw = [0u8; 8];
    copy_from_guest(ptr, &mut raw).then(|| u64::from_ne_bytes(raw))
}

/// Forward a syscall whose argument at `arg` is a `sigset_t` pointer, with
/// `SIGSYS` cleared from the set. A null pointer, and a set that does not
/// block `SIGSYS`, forward untouched.
fn forward_with_filtered_mask(call: &mut SystemCall, arg: usize) {
    let ptr = call.args[arg];
    if ptr == 0 {
        call.set_result(host_syscall(call));
        return;
    }
    let Some(set) = read_guest_sigset(ptr) else {
        call.set_result(SyscallResult::Error(libc::EFAULT));
        return;
    };
    if set & SIGSYS_BIT == 0 {
        call.set_result(host_syscall(call));
        return;
    }
    let filtered = set & !SIGSYS_BIT;
    let mut args = call.args;
    args[arg] = &filtered as *const u64 as u64;
    call.set_result(host_syscall(&SystemCall::new(call.number, args)));
}

pub fn do_sigprocmask(call: &mut SystemCall) {
    // SIG_UNBLOCK and a query (null set) can only ever clear bits.
    if call.args[0] as i32 == libc::SIG_UNBLOCK {
        call.set_result(host_syscall(call));
        return;
    }
    forward_with_filtered_mask(call, 1);
}

pub fn do_sigsuspend(call: &mut SystemCall) {
    forward_with_filtered_mask(call, 0);
}

/// Forward a guest `rt_sigaction`, swapping the restorer for
/// [`chimera_sud_restorer`] and clearing `SIGSYS` from the handler's mask.
/// The runtime owns `SIGSYS` itself: a guest installing a handler for it is
/// told it succeeded, so a harness-style catch-all keeps running.
pub fn do_sigaction(call: &mut SystemCall) {
    let sig = call.args[0] as i32;
    if sig == libc::SIGSYS {
        call.set_result(SyscallResult::Ok(0));
        return;
    }
    let act_ptr = call.args[1];
    if act_ptr == 0 {
        call.set_result(host_syscall(call));
        return;
    }
    let mut raw = [0u8; mem::size_of::<KernelSigaction>()];
    if !copy_from_guest(act_ptr, &mut raw) {
        call.set_result(SyscallResult::Error(libc::EFAULT));
        return;
    }
    let mut act: KernelSigaction = unsafe { mem::transmute(raw) };
    if act.handler != libc::SIG_DFL as u64 && act.handler != libc::SIG_IGN as u64 {
        act.flags |= SA_RESTORER;
        act.restorer = chimera_sud_restorer as *const () as u64;
    }
    act.mask &= !SIGSYS_BIT;
    let patched = SystemCall::new(
        call.number,
        [
            call.args[0],
            &act as *const KernelSigaction as u64,
            call.args[2],
            call.args[3],
            0,
            0,
        ],
    );
    let result = host_syscall(&patched);
    // What the guest reads back is the restorer it gave, not the
    // substitute: the kernel reports the substitute, so the old action is
    // rewritten on the way out.
    if let (SyscallResult::Ok(_), true) = (result, call.args[2] != 0) {
        let mut old = [0u8; mem::size_of::<KernelSigaction>()];
        if copy_from_guest(call.args[2], &mut old) {
            let mut old: KernelSigaction = unsafe { mem::transmute(old) };
            if old.restorer == chimera_sud_restorer as *const () as u64 {
                old.restorer = 0;
                old.flags &= !SA_RESTORER;
                let raw: [u8; mem::size_of::<KernelSigaction>()] = unsafe { mem::transmute(old) };
                copy_to_guest(call.args[2], &raw);
            }
        }
    }
    call.set_result(result);
}

/// POSIX `execve` resets caught signals to their default disposition and
/// leaves ignored ones ignored; `clone3`'s `CLONE_CLEAR_SIGHAND` asks for the
/// same in the child. The signals the runtime owns are skipped: `SIGSYS`,
/// the dispatch trap, and `SIGSEGV`/`SIGBUS`, the guarded-copy fixup.
pub fn reset_guest_signals() {
    for sig in 1..=libc::SIGRTMAX() {
        if matches!(
            sig,
            libc::SIGKILL | libc::SIGSTOP | libc::SIGSYS | libc::SIGSEGV | libc::SIGBUS
        ) {
            continue;
        }
        unsafe {
            let mut old: libc::sigaction = mem::zeroed();
            if libc::sigaction(sig, ptr::null(), &mut old) == 0
                && old.sa_sigaction != libc::SIG_DFL
                && old.sa_sigaction != libc::SIG_IGN
            {
                let mut dfl: libc::sigaction = mem::zeroed();
                dfl.sa_sigaction = libc::SIG_DFL;
                libc::sigemptyset(&mut dfl.sa_mask);
                libc::sigaction(sig, &dfl, ptr::null_mut());
            }
        }
    }
}
