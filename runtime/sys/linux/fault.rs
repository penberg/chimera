//! Synchronous fault handler for self-modifying code.
//!
//! JIT engines — JavaScriptCore in Bun — rewrite their own generated machine
//! code in place. Chimera translates guest code ahead of execution and caches
//! the result, so a guest store that overwrites already-translated code would
//! leave a stale translation running. Once Chimera has translated code from a
//! writable+executable (JIT) guest page it maps that page read-only on the host
//! ([`crate::sys::mmap::AddressSpace::arm_span`]); a guest store then traps here.
//! This handler drops the page's stale translations, restores write permission,
//! and returns so the faulting store re-executes and lands — and the next
//! execution of that page re-translates the new code.
//!
//! It owns the host `SIGSEGV`/`SIGBUS` disposition. The guest's own handlers for
//! those are recorded in the disposition table but never installed on the host
//! slot ([`super::signal`]), so this handler always runs first and can classify
//! the fault. A fault that is not an SMC write — a genuine guest fault, or one
//! taken in Chimera's own code — is left to terminate the process with the
//! faithful signal; precise delivery into a guest fault handler is not modeled.
//!
//! The handler is async-signal-safe: it touches only atomics, a `Mutex` and
//! `HashMap` whose operations never allocate, and `mprotect`. It must not call
//! the allocator, use TLS, or panic — it runs on whatever context the store
//! interrupted, with the guest's `FS` base still loaded.

use std::{
    mem, ptr,
    sync::{
        Once,
        atomic::{AtomicPtr, Ordering},
    },
};

use crate::{arch::x86::translate::code_cache_contains, process::Process};

/// The running guest's shared process state, published by [`set_process`] so
/// the fault handler can reach the shared address space. A raw pointer because
/// the handler cannot hold an `Arc`; the process outlives every fault.
static PROCESS: AtomicPtr<Process> = AtomicPtr::new(ptr::null_mut());

/// Publish the guest process for the fault handler, before any guest code runs.
pub fn set_process(process: &Process) {
    PROCESS.store(process as *const Process as *mut Process, Ordering::Release);
}

/// Install the synchronous `SIGSEGV`/`SIGBUS` handler, once per process.
pub fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = chimera_fault as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = libc::SA_SIGINFO;
        libc::sigaction(libc::SIGSEGV, &sa, ptr::null_mut());
        libc::sigaction(libc::SIGBUS, &sa, ptr::null_mut());
    });
}

extern "C" fn chimera_fault(
    signo: libc::c_int,
    info: *const libc::siginfo_t,
    ucontext: *mut libc::c_void,
) {
    let fault_addr = unsafe { (*info).si_addr() } as usize;
    let rip = fault_rip(ucontext);

    // Only a fault taken while executing translated guest code can be an SMC
    // write, and only then is the address-space lock guaranteed free on this
    // thread — it is never held across dispatch — so the handler can take it
    // without deadlocking against itself.
    if code_cache_contains(rip) {
        let process = PROCESS.load(Ordering::Acquire);
        if !process.is_null() {
            let process = unsafe { &*process };
            if process.addr_space.lock().unwrap().on_smc_write(fault_addr) {
                return; // write permission restored; re-execute the store
            }
        }
    }

    // Not an SMC write: a genuine guest fault, or a fault in Chimera itself.
    // Restore the default disposition and return; the faulting instruction
    // re-executes and the kernel terminates the process with the real signal.
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0;
        libc::sigaction(signo, &sa, ptr::null_mut());
    }
}

/// Read the faulting instruction pointer from the signal's `ucontext`.
fn fault_rip(ucontext: *mut libc::c_void) -> usize {
    if ucontext.is_null() {
        return 0;
    }
    let uc = ucontext as *const libc::ucontext_t;
    unsafe { (*uc).uc_mcontext.gregs[libc::REG_RIP as usize] as usize }
}
