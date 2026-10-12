//! A one-shot value handoff that survives fork.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::SystemCall;

const SYS_ULOCK_WAIT: u64 = 515;
const SYS_ULOCK_WAKE: u64 = 516;
const UL_COMPARE_AND_WAIT: u64 = 1;
const ULF_NO_ERRNO: u64 = 0x0100_0000;

/// A one-shot value handoff between two threads. Blocking on a Rust channel
/// parks through std's Darwin thread parker — a per-thread libdispatch
/// semaphore, created once for the thread's lifetime — and Mach port names do
/// not survive fork, so a thread promoted into a fork child dies inside
/// libdispatch on the parent's stale port the first time it parks or is
/// signaled. This handoff is a ulock on ordinary memory, which fork
/// preserves.
pub struct Handoff {
    full: AtomicU32,
    value: AtomicU64,
}

impl Handoff {
    pub fn new() -> Self {
        Self {
            full: AtomicU32::new(0),
            value: AtomicU64::new(0),
        }
    }

    /// Publish `value` and wake the receiver. Call at most once.
    pub fn send(&self, value: u64) {
        self.value.store(value, Ordering::Release);
        self.full.store(1, Ordering::Release);
        let op = UL_COMPARE_AND_WAIT | ULF_NO_ERRNO;
        let call = SystemCall::new(
            SYS_ULOCK_WAKE,
            [op, self.full.as_ptr() as u64, 0, 0, 0, 0, 0, 0],
        );
        super::syscall::host_syscall(&call);
    }

    /// Block until [`Self::send`] has published, and return its value.
    pub fn recv(&self) -> u64 {
        let op = UL_COMPARE_AND_WAIT | ULF_NO_ERRNO;
        while self.full.load(Ordering::Acquire) == 0 {
            let call = SystemCall::new(
                SYS_ULOCK_WAIT,
                [op, self.full.as_ptr() as u64, 0, 0, 0, 0, 0, 0],
            );
            super::syscall::host_syscall(&call);
        }
        self.value.load(Ordering::Acquire)
    }
}
