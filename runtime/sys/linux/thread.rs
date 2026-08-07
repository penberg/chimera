//! The Linux primitive for forcing a guest thread to its run-loop boundary: a
//! reserved real-time signal delivered with `tgkill`. A do-nothing handler is
//! installed for it (without `SA_RESTART`) so the kernel interrupts a thread
//! parked in a blocking host syscall — it returns `EINTR` and re-checks the
//! process exit/exec flags — without the guest ever observing the signal.
//!
//! Chimera reaches this behind the neutral `sys::thread` name; a second host
//! supplies its own `interrupt`/`reserved_signal` (Darwin: `pthread_kill`).

use std::sync::Once;

/// Host signal Chimera reserves to interrupt a guest thread parked in a
/// forwarded syscall. The highest real-time signal is the one least likely to
/// collide with a signal the guest itself installs.
pub fn reserved_signal() -> libc::c_int {
    libc::SIGRTMAX()
}

extern "C" fn interrupt_noop(_sig: libc::c_int) {}

static INSTALL_INTERRUPT: Once = Once::new();

/// Install the no-op handler for [`reserved_signal`] once per process. The
/// handler carries no `SA_RESTART`, so a parked syscall surfaces `EINTR`
/// rather than silently restarting.
pub fn install_interrupt_handler() {
    INSTALL_INTERRUPT.call_once(|| unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = interrupt_noop as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0; // no SA_RESTART: a parked syscall must surface EINTR
        libc::sigaction(reserved_signal(), &sa, std::ptr::null_mut());
    });
}

/// Force the guest thread with kernel id `tid` to its run-loop boundary by
/// delivering [`reserved_signal`] to it. A thread parked in a blocking host
/// syscall returns `EINTR`; one executing translated code is already pulled
/// out by its armed safepoint slot, which the caller sets before calling here.
pub fn interrupt(tid: i32) {
    unsafe {
        let pid = libc::getpid();
        libc::syscall(libc::SYS_tgkill, pid, tid, reserved_signal());
    }
}
