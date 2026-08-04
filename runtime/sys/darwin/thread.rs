//! The Darwin primitive for forcing a guest thread to its run-loop boundary: a
//! reserved signal delivered with `__pthread_kill`. A do-nothing handler is
//! installed for it (without `SA_RESTART`) so the kernel interrupts a thread
//! parked in a blocking host syscall — it returns `EINTR` and re-checks the
//! process exit flags at its next block boundary — without the guest observing
//! the signal.
//!
//! Chimera reaches this behind the neutral `sys::thread` name; the Linux host
//! supplies its own `interrupt`/`reserved_signal` (`tgkill` on `SIGRTMAX`).

use std::sync::Once;

/// Signal Chimera reserves to interrupt a guest thread parked in a forwarded
/// syscall. Darwin has no real-time signals, so there is no collision-free
/// choice; `SIGURG` is the conventional runtime-preemption signal (the Go
/// runtime uses it for exactly this on macOS) — its default action is Ignore
/// and compute guests rarely touch it. A guest that installs its own `SIGURG`
/// handler never sees it: [`super::signal`] keeps this slot for Chimera, the
/// way it keeps `SIGSEGV`/`SIGBUS` for the fault handler.
pub fn reserved_signal() -> libc::c_int {
    libc::SIGURG
}

/// Darwin `SYS___pthread_kill`: deliver a signal to a thread by its Mach port.
const SYS_PTHREAD_KILL: libc::c_int = 328;

extern "C" fn interrupt_noop(_sig: libc::c_int) {}

static INSTALL_INTERRUPT: Once = Once::new();

/// Install the no-op handler for [`reserved_signal`] once per process. No
/// `SA_RESTART`, so a parked syscall surfaces `EINTR` rather than restarting.
pub fn install_interrupt_handler() {
    INSTALL_INTERRUPT.call_once(|| unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = interrupt_noop as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0;
        libc::sigaction(reserved_signal(), &sa, std::ptr::null_mut());
    });
}

/// Force the guest thread whose Mach port is `tid` to its run-loop boundary by
/// delivering [`reserved_signal`] to it. A thread parked in a blocking host
/// syscall returns `EINTR`; one executing translated code is already pulled out
/// by its armed safepoint slot, which the caller sets before calling here.
/// (`tid` holds the Mach thread port — `pthread_mach_thread_np` — that
/// `Thread::run` published in `ThreadState::tid`.)
pub fn interrupt(tid: i32) {
    if tid != 0 {
        unsafe {
            libc::syscall(SYS_PTHREAD_KILL, tid as u32, reserved_signal());
        }
    }
}
