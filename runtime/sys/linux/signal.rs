//! Guest signal virtualization.
//!
//! Chimera never lets the host kernel deliver a signal straight to a guest
//! handler: that would jump the host thread natively into untranslated guest
//! code (outside the sandbox, and a hard fault under W^X). Instead Chimera owns
//! the guest's signal disposition. `rt_sigaction`/`rt_sigprocmask`/`sigaltstack`
//! are intercepted into [`Signals`]; for any caught signal a single host-side
//! catcher ([`chimera_sigcatch`]) is installed that records the signal as
//! pending and preempts the interrupted thread out of translated code
//! ([`crate::arch::preempt`]), so it reaches its run loop at once with the
//! precise register state of the interrupted instruction. The run loop drains
//! the pending set there, builds a kernel-ABI `rt_sigframe` on the guest stack
//! with [`Signals::deliver`], and re-enters the translator at the handler — so
//! the handler runs translated, in-sandbox. The handler returns through its
//! restorer (`sa_restorer`, or a built-in `rt_sigreturn` stub), whose
//! `rt_sigreturn` is intercepted into [`Signals::restore`].
//!
//! A blocking host syscall forwarded on the guest's behalf is interrupted
//! (the catcher is installed without `SA_RESTART`), so the loop regains control
//! and delivers at its next iteration.
//!
//! The guest's blocked mask is emulated in [`Signals`] and mirrored onto the
//! host thread's mask ([`mirror_host_blocked`]) for the guest's whole lifetime,
//! intercepted-`execve` windows included; [`HostMaskGuard`] restores the
//! caller's own mask once the guest is done.
//!
//! The mirror claims no isolation between the guest's signals and the
//! embedder's: signal-pending state is process-scoped and the guest runs inside
//! the embedder's process, so a blocked process-directed signal is visible to
//! the guest's `rt_sigpending` and can be dequeued by its `rt_sigtimedwait`
//! whoever it was meant for. An embedder that concurrently relies on its own
//! process-directed signal handling (a dedicated `sigwait` thread, say) is not
//! supported while a guest is running.

use std::{
    cell::UnsafeCell,
    mem, ptr,
    sync::{
        Arc, Mutex, Once,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
};

use crate::{SyscallResult, arch::dispatch::ThreadState};

/// Number of signals (`SIGRTMAX`); signals are numbered `1..=NSIG`.
const NSIG: usize = 64;

/// Guest handler sentinels, matching the kernel's `SIG_DFL`/`SIG_IGN`.
const SIG_DFL: u64 = 0;
const SIG_IGN: u64 = 1;

/// `sa_flags` bit (not always exposed by `libc`): the guest supplied a restorer.
const SA_RESTORER: u64 = 0x0400_0000;

/// `ss_flags` bit: the alternate stack is disabled.
const SS_DISABLE: i32 = 2;

/// Signals that can never be caught, blocked, or ignored.
const SIGKILL: u64 = 9;
const SIGSTOP: u64 = 19;

/// The synchronous fault signals, whose host disposition Chimera keeps for its
/// own self-modifying-code trap handler rather than mirroring the guest's.
const SIGBUS: u64 = 7;
const SIGSEGV: u64 = 11;

// Guest register-file indices (see `crate::arch::dispatch`). r8..r15 are 8..15.
const RAX: usize = 0;
const RBX: usize = 1;
const RCX: usize = 2;
const RDX: usize = 3;
const RSI: usize = 4;
const RDI: usize = 5;
const RBP: usize = 6;
const RSP: usize = 7;

/// One guest signal disposition, in the raw `rt_sigaction` ABI shape. Public
/// only because it appears in [`SharedSigTable`], which the engine stores; the
/// fields stay an implementation detail of this module.
#[derive(Clone, Copy, Default)]
pub struct GuestSigaction {
    handler: u64,
    flags: u64,
    restorer: u64,
    mask: u64,
}

/// The kernel `rt_sigaction` struct (x86-64), as the raw syscall sees it.
#[repr(C)]
#[derive(Clone, Copy)]
struct KernelSigaction {
    handler: u64,
    flags: u64,
    restorer: u64,
    mask: u64,
}

/// `stack_t` as the kernel lays it out on x86-64.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct StackT {
    ss_sp: u64,
    ss_flags: i32,
    _pad: i32,
    ss_size: u64,
}

/// `struct sigcontext` (x86-64) — the machine state inside `ucontext`.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct SigContext {
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    rdi: u64,
    rsi: u64,
    rbp: u64,
    rbx: u64,
    rdx: u64,
    rax: u64,
    rcx: u64,
    rsp: u64,
    rip: u64,
    eflags: u64,
    cs: u16,
    gs: u16,
    fs: u16,
    ss: u16,
    err: u64,
    trapno: u64,
    oldmask: u64,
    cr2: u64,
    /// Pointer to the saved extended state (XSAVE area).
    fpstate: u64,
    reserved: [u64; 8],
}

/// `struct ucontext` (x86-64).
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct UContext {
    uc_flags: u64,
    uc_link: u64,
    uc_stack: StackT,
    uc_mcontext: SigContext,
    uc_sigmask: [u64; 16],
}

/// `siginfo_t` (128 bytes). Captured wholesale from the kernel by the catcher, so
/// the named fields plus the trailing union (si_pid/si_uid/si_value/…) carried in
/// `_pad` reach the guest handler intact.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct SigInfo {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _pad: [i32; 29],
}

/// The kernel `rt_sigframe` (x86-64): restorer return address, then `ucontext`
/// and `siginfo`.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct RtSigframe {
    pretcode: u64,
    uc: UContext,
    info: SigInfo,
}

/// The lowest real-time signal number at the kernel ABI level. Signals
/// `1..SIGRTMIN_KERNEL` are standard (coalescing); `SIGRTMIN_KERNEL..=NSIG` are
/// real-time and queue. glibc reserves the first few for NPTL but they are still
/// real-time at the ABI Chimera operates on.
const SIGRTMIN_KERNEL: u32 = 32;

/// Size of the kernel `siginfo_t` on x86-64.
const SIGINFO_SIZE: usize = 128;

const _: () = assert!(mem::size_of::<SigInfo>() == SIGINFO_SIZE);

/// A captured `siginfo_t`, used as a per-signal slot. For a standard signal the
/// catcher writes it directly (coalesced, latest wins); for a real-time signal
/// it is the staging slot that [`PendingSet::stage_rt`] pops the next queued
/// payload into just before delivery. Written by the catcher (a signal handler)
/// or the drainer and read by the dispatch loop, so access is unsynchronized;
/// the `memcpy` is async-signal-safe and a slot is only read after the write
/// that filled it has completed.
struct SigInfoSlot(UnsafeCell<[u8; SIGINFO_SIZE]>);

// SAFETY: writes happen only in the async-signal-safe catcher; reads only at a
// dispatch-loop safe point after the corresponding catcher has completed.
unsafe impl Sync for SigInfoSlot {}

/// Capacity of each real-time signal's `siginfo` ring. Beyond this many queued
/// instances the oldest `siginfo` payloads are overwritten (the host's
/// `RLIMIT_SIGPENDING` bounds real queue depth well below this in practice).
const RT_RING_CAP: usize = 32;

/// Per-thread pending-signal state: the signals the host catcher has recorded
/// on this thread but the dispatch loop has not yet delivered.
///
/// Pending state must be per-thread, not process-wide: a thread-directed
/// signal (`tgkill`, and with the guest's blocked masks mirrored onto the host,
/// any signal the kernel routed) belongs to the thread that caught it. With one
/// shared set, whichever thread reached a delivery point first consumed
/// everyone's signals — JavaScriptCore's SIGPWR thread-suspend/resume protocol
/// deadlocked exactly that way: a sibling swallowed the resume while the
/// suspended thread waited in `sigsuspend` holding the heap lock.
///
/// Each thread's set is owned (via `Arc`) by its [`Signals`]; a raw pointer to
/// it is published in the gs-addressed `ThreadState` so the host catcher can
/// reach the right set with no TLS, allocation, or locking.
pub struct PendingSet {
    /// One bit per signal: "at least one instance pending". For real-time
    /// signals the depth is tracked in `counts`; standard signals coalesce.
    bits: AtomicU64,
    /// Per-signal queue depth for real-time signals (indexed by `signo - 1`).
    counts: [AtomicU32; NSIG],
    /// Per-signal captured/staged `siginfo_t`.
    siginfo: [SigInfoSlot; NSIG],
    /// Per-real-time-signal FIFO ring of captured `siginfo_t`, so each queued
    /// instance keeps its own payload (e.g. a `sigqueue` value). `rt_head` is
    /// the push index (advanced by the catcher), `rt_tail` the pop index
    /// (advanced as each instance is delivered); their difference mirrors
    /// `counts`.
    rt_ring: [[SigInfoSlot; RT_RING_CAP]; NSIG],
    rt_head: [AtomicU32; NSIG],
    rt_tail: [AtomicU32; NSIG],
}

impl PendingSet {
    fn new() -> Self {
        Self {
            bits: AtomicU64::new(0),
            counts: [const { AtomicU32::new(0) }; NSIG],
            siginfo: [const { SigInfoSlot(UnsafeCell::new([0u8; SIGINFO_SIZE])) }; NSIG],
            rt_ring: [const {
                [const { SigInfoSlot(UnsafeCell::new([0u8; SIGINFO_SIZE])) }; RT_RING_CAP]
            }; NSIG],
            rt_head: [const { AtomicU32::new(0) }; NSIG],
            rt_tail: [const { AtomicU32::new(0) }; NSIG],
        }
    }

    /// Record one instance of `signo` (with its captured `siginfo_t`): the
    /// shared body of the host catcher and of the [`wait_for_set`] drain, which
    /// dequeues signals the host kernel held pending while the guest had them
    /// blocked. Async-signal-safe: only plain atomics and a fixed-size `memcpy`
    /// — no TLS, no allocation, no locks, no panics. For a real-time signal the
    /// queue depth is bumped before the bit is marked, so a drainer that
    /// observes the bit set always sees a positive count.
    fn record(&self, signo: u32, info: *const libc::siginfo_t) {
        let idx = signo as usize - 1;
        if signo >= SIGRTMIN_KERNEL {
            // Real-time: enqueue this instance's siginfo and bump the depth.
            let slot = self.rt_head[idx].fetch_add(1, Ordering::AcqRel) as usize % RT_RING_CAP;
            if !info.is_null() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        info as *const u8,
                        self.rt_ring[idx][slot].0.get() as *mut u8,
                        SIGINFO_SIZE,
                    );
                }
            }
            self.counts[idx].fetch_add(1, Ordering::AcqRel);
        } else if !info.is_null() {
            // Standard: coalesced, a single slot holds the latest siginfo.
            unsafe {
                ptr::copy_nonoverlapping(
                    info as *const u8,
                    self.siginfo[idx].0.get() as *mut u8,
                    SIGINFO_SIZE,
                );
            }
        }
        self.bits
            .fetch_or(1u64 << (signo as u64 - 1), Ordering::Release);
    }

    /// Pop the oldest queued `siginfo` for a real-time signal into its delivery
    /// staging slot, so [`PendingSet::captured`] reads the right per-instance
    /// payload. Balanced with the `counts` decrement in [`PendingSet::take`],
    /// one pop per taken instance.
    fn stage_rt(&self, idx: usize) {
        let slot = self.rt_tail[idx].fetch_add(1, Ordering::AcqRel) as usize % RT_RING_CAP;
        unsafe {
            ptr::copy_nonoverlapping(
                self.rt_ring[idx][slot].0.get() as *const u8,
                self.siginfo[idx].0.get() as *mut u8,
                SIGINFO_SIZE,
            );
        }
    }

    /// Copy the captured `siginfo_t` for `signo` out of its slot.
    fn captured(&self, signo: u32) -> SigInfo {
        let mut si: SigInfo = unsafe { mem::zeroed() };
        unsafe {
            ptr::copy_nonoverlapping(
                self.siginfo[signo as usize - 1].0.get() as *const u8,
                &mut si as *mut SigInfo as *mut u8,
                SIGINFO_SIZE,
            );
        }
        si
    }

    /// Snapshot the recorded-but-undelivered set as a sigset bitmask.
    fn snapshot(&self) -> u64 {
        self.bits.load(Ordering::Acquire)
    }

    /// Clear all pending state (a freshly forked child starts empty).
    fn reset(&self) {
        self.bits.store(0, Ordering::Release);
        for i in 0..NSIG {
            self.counts[i].store(0, Ordering::Release);
            self.rt_head[i].store(0, Ordering::Release);
            self.rt_tail[i].store(0, Ordering::Release);
        }
    }

    /// Atomically remove and return the lowest-numbered pending signal within
    /// the `allowed` set, or `None`. A standard signal clears its bit
    /// (coalesced); a real-time signal consumes one queued instance and keeps
    /// its bit set while more remain.
    fn take(&self, allowed: u64) -> Option<u32> {
        loop {
            let cur = self.bits.load(Ordering::Acquire);
            let avail = cur & allowed;
            if avail == 0 {
                return None;
            }
            let bit = avail & avail.wrapping_neg();
            let signo = bit.trailing_zeros() + 1;

            if signo >= SIGRTMIN_KERNEL {
                let idx = signo as usize - 1;
                let depth = self.counts[idx].load(Ordering::Acquire);
                if depth == 0 {
                    // Bit set with an empty queue (already drained elsewhere):
                    // clear it and re-pick.
                    self.bits.fetch_and(!bit, Ordering::AcqRel);
                    continue;
                }
                if self.counts[idx]
                    .compare_exchange_weak(depth, depth - 1, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    continue;
                }
                // We own one instance; stage its queued siginfo for delivery.
                self.stage_rt(idx);
                if depth == 1 {
                    // Took the last queued instance; clear the bit, then re-arm
                    // it if the catcher enqueued another in the meantime.
                    self.bits.fetch_and(!bit, Ordering::AcqRel);
                    if self.counts[idx].load(Ordering::Acquire) > 0 {
                        self.bits.fetch_or(bit, Ordering::Release);
                    }
                }
                return Some(signo);
            }

            // Standard signal: coalesced, clear the single bit.
            let new = cur & !bit;
            if self
                .bits
                .compare_exchange_weak(cur, new, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(signo);
            }
        }
    }
}

/// Host signal catcher. Runs asynchronously on whatever context the host thread
/// happened to be in (translated guest code or Chimera Rust), so it must be
/// async-signal-safe: [`PendingSet::record`], the preemption code, and
/// `gs`-relative accesses. Installed with `SA_SIGINFO`, so it captures the
/// kernel's `siginfo_t` for the guest handler.
///
/// The signal is recorded on the *catching* thread's own [`PendingSet`],
/// reached through the gs-addressed `ThreadState`. The kernel already routed
/// the signal correctly — a `tgkill` to its target thread, a process-directed
/// signal to a thread with it unblocked (the guest's masks are mirrored onto
/// the host) — so catching thread == owning thread, and only that thread's
/// dispatch loop delivers it.
extern "C" fn chimera_sigcatch(
    signo: libc::c_int,
    info: *const libc::siginfo_t,
    uc: *mut libc::c_void,
) {
    if signo >= 1 && signo as usize <= NSIG {
        // Temporary diagnostic: raw, alloc-free trace of every SIGPWR catch
        // (Bun/JSC's thread-suspend/resume signal) with the catching tid.
        let pending: *const PendingSet;
        unsafe {
            core::arch::asm!(
                "mov {out}, qword ptr gs:[{off}]",
                out = out(reg) pending,
                off = const core::mem::offset_of!(ThreadState, pending_set),
                options(nostack, preserves_flags, readonly),
            );
        }
        if !pending.is_null() {
            unsafe { (*pending).record(signo as u32, info) };
        }

        // Drag the interrupted guest thread back to its run loop. Translated code
        // keeps the guest registers live across linked block chains and only
        // returns to the dispatcher at a block exit, so a tight syscall-free loop
        // would never observe the bit set above. Preempt it: the interrupted
        // context is rewritten so the thread leaves the cache at this very
        // instruction boundary. A thread caught in Chimera's own Rust instead
        // has its next cache entry declined, so the run loop comes back to its
        // delivery point before any translated code runs. GS is this thread's
        // `ThreadState` throughout guest execution (bound once via
        // `ARCH_SET_GS`, never changed — only FS is swapped), so a single
        // `gs:[]` store reaches the right context with no TLS, allocation, or
        // locking, keeping the catcher async-signal-safe.
        if !crate::arch::preempt(uc) {
            unsafe {
                core::arch::asm!(
                    "mov dword ptr gs:[{off}], 1",
                    off = const core::mem::offset_of!(ThreadState, exit_requested),
                    options(nostack, preserves_flags),
                );
            }
        }
    }
}

/// Build a host `sigset_t` holding the signals set in the `mask` bitmask.
fn host_sigset(mask: u64) -> libc::sigset_t {
    unsafe {
        let mut s: libc::sigset_t = mem::zeroed();
        libc::sigemptyset(&mut s);
        for i in 0..NSIG {
            if mask & (1u64 << i) != 0 {
                libc::sigaddset(&mut s, i as i32 + 1);
            }
        }
        s
    }
}

/// Mirror the guest's blocked mask onto the host thread's signal mask. The
/// kernel consults the caller's real mask whenever it generates a signal
/// itself — `tcsetpgrp` from a background process group stops the caller with
/// SIGTTOU unless SIGTTOU is blocked or ignored, and a blocked fatal signal
/// must be held pending rather than acted on — so a mask emulated only in
/// software gets those cases wrong.
fn mirror_host_blocked(mask: u64) {
    let set = host_sigset(mask);
    unsafe {
        libc::pthread_sigmask(libc::SIG_SETMASK, &set, ptr::null_mut());
    }
}

/// The bitmask of the signals set in a host `sigset_t`.
fn sigset_mask(s: &libc::sigset_t) -> u64 {
    let mut mask = 0u64;
    for i in 0..NSIG {
        if unsafe { libc::sigismember(s, i as i32 + 1) } == 1 {
            mask |= 1u64 << i;
        }
    }
    mask
}

/// Saves the host thread's signal mask on construction and restores it on
/// drop, so the mask the guest leaves behind does not leak onto the embedder's
/// thread. Held around the guest's whole lifetime — the run/execve loop, not a
/// single `run()` call — because POSIX preserves the mask across `execve`,
/// loader window included.
pub struct HostMaskGuard {
    saved: libc::sigset_t,
}

impl HostMaskGuard {
    pub fn save() -> Self {
        unsafe {
            let mut saved: libc::sigset_t = mem::zeroed();
            libc::pthread_sigmask(libc::SIG_SETMASK, ptr::null(), &mut saved);
            Self { saved }
        }
    }
}

impl Drop for HostMaskGuard {
    fn drop(&mut self) {
        unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, &self.saved, ptr::null_mut());
        }
    }
}

/// Snapshot the host kernel's pending set for this thread as a sigset bitmask.
/// With the guest's blocked mask mirrored onto the host, a signal that arrives
/// while blocked is held pending by the kernel and never reaches the catcher,
/// so the guest-visible pending set is the union of this and [`PENDING`].
fn host_pending_snapshot() -> u64 {
    unsafe {
        let mut s: libc::sigset_t = mem::zeroed();
        libc::sigemptyset(&mut s);
        libc::sigpending(&mut s);
        sigset_mask(&s)
    }
}

/// Block the calling host thread until a signal deliverable under `blocked`
/// (pending and not blocked) has been recorded by the host catcher. Race-free:
/// all host signals are blocked first, the already-pending set is checked, then
/// `sigsuspend` atomically unblocks the deliverable ones and waits, so a signal
/// arriving in the window is not lost. Returns once such a signal is pending,
/// leaving it in `PENDING` for the dispatch loop to deliver.
fn wait_for_signal(pending: &PendingSet, blocked: u64) {
    unsafe {
        let mut all: libc::sigset_t = mem::zeroed();
        libc::sigfillset(&mut all);
        let mut prev: libc::sigset_t = mem::zeroed();
        libc::pthread_sigmask(libc::SIG_BLOCK, &all, &mut prev);
        // During the wait, keep the guest's blocked signals masked so only a
        // deliverable signal wakes us.
        let wait = host_sigset(blocked);
        while pending.snapshot() & !blocked == 0 {
            libc::sigsuspend(&wait);
        }
        libc::pthread_sigmask(libc::SIG_SETMASK, &prev, ptr::null_mut());
    }
}

/// The current `CLOCK_MONOTONIC` reading in nanoseconds.
fn monotonic_ns() -> u128 {
    unsafe {
        let mut ts: libc::timespec = mem::zeroed();
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
        ts.tv_sec as u128 * 1_000_000_000 + ts.tv_nsec as u128
    }
}

/// Outcome of [`wait_for_set`].
enum WaitResult {
    /// A signal from the wait set was accepted (and removed from pending).
    Got(u32),
    /// The deadline elapsed first.
    Timeout,
    /// A deliverable signal outside the wait set arrived and must be handled.
    Interrupted,
}

/// Synchronously accept a signal in `set`: block the host thread until one is
/// pending (removing it), the optional `deadline` (in `CLOCK_MONOTONIC` ns)
/// elapses, or a deliverable signal outside `set` arrives. All host signals
/// are blocked around the wait so the pending checks cannot race a late
/// arrival; the wait itself is the host's own `sigtimedwait`, which dequeues a
/// signal without running its disposition — exactly the accept semantics — and
/// sees both a signal the catcher has recorded on this thread and one the host
/// kernel held pending while the guest had it blocked. A deliverable signal
/// outside `set` is dequeued too and recorded for dispatch-loop delivery —
/// except the `ignored` ones, which the kernel would discard at generation and
/// so must not surface as an `EINTR`, and the default-`stops`, which are left
/// unblocked for the host kernel to act on: the process stops, and once
/// continued the wait reports `EINTR`. Inside `set` both are still accepted;
/// Linux holds a blocked ignored signal pending precisely so a waiter can
/// dequeue it. A zero `remaining` makes `sigtimedwait` a poll, so an
/// already-pending signal is still accepted after the deadline.
fn wait_for_set(
    pending: &PendingSet,
    set: u64,
    blocked: u64,
    ignored: u64,
    stops: u64,
    deadline: Option<u128>,
) -> WaitResult {
    unsafe {
        let mut all: libc::sigset_t = mem::zeroed();
        libc::sigfillset(&mut all);
        let mut prev: libc::sigset_t = mem::zeroed();
        libc::pthread_sigmask(libc::SIG_BLOCK, &all, &mut prev);
        let want = host_sigset(set | (!blocked & !ignored & !stops));
        let unblock_stops = stops & !blocked & !set;
        let unblock = host_sigset(unblock_stops);

        let result = loop {
            if let Some(signo) = pending.take(set) {
                break WaitResult::Got(signo);
            }
            if pending.snapshot() & !blocked & !set & !ignored != 0 {
                break WaitResult::Interrupted;
            }
            let remaining = deadline.map(|d| {
                let rem = d.saturating_sub(monotonic_ns());
                libc::timespec {
                    tv_sec: (rem / 1_000_000_000) as libc::time_t,
                    tv_nsec: (rem % 1_000_000_000) as i64,
                }
            });
            let ts_ptr = remaining
                .as_ref()
                .map_or(ptr::null(), |ts| ts as *const libc::timespec);
            let mut si: libc::siginfo_t = mem::zeroed();
            if unblock_stops != 0 {
                libc::pthread_sigmask(libc::SIG_UNBLOCK, &unblock, ptr::null_mut());
            }
            let r = libc::sigtimedwait(&want, &mut si, ts_ptr);
            let interrupted =
                r < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR);
            if unblock_stops != 0 {
                libc::pthread_sigmask(libc::SIG_BLOCK, &unblock, ptr::null_mut());
            }
            if r > 0 {
                pending.record(r as u32, &si);
            } else if interrupted {
                // Everything but the unblocked stop signals is blocked, so an
                // EINTR means the process was stopped and continued.
                if unblock_stops != 0 {
                    break WaitResult::Interrupted;
                }
            } else {
                break WaitResult::Timeout; // EAGAIN
            }
        };
        libc::pthread_sigmask(libc::SIG_SETMASK, &prev, ptr::null_mut());
        result
    }
}

/// Install the host disposition for `signo`: the real handler for `SIG_DFL`/
/// `SIG_IGN`, or [`chimera_sigcatch`] for any custom guest handler. The host
/// handler is deliberately installed without `SA_RESTART` so a forwarded
/// blocking syscall is interrupted and the dispatch loop regains control.
fn install_host(signo: usize, handler: u64) {
    // SIGSEGV and SIGBUS belong to the synchronous fault handler (see
    // [`super::fault`]), which must run first to catch self-modifying-code write
    // traps. The guest's disposition for them is still recorded in the table
    // above (so it is reported back and can be honored once guest-fault delivery
    // exists), but it is never installed on the host slot.
    if signo == SIGSEGV as usize || signo == SIGBUS as usize {
        return;
    }
    let (host, flags) = match handler {
        SIG_DFL => (libc::SIG_DFL, 0),
        SIG_IGN => (libc::SIG_IGN, 0),
        // SA_SIGINFO so the kernel hands the catcher the real siginfo_t. Still no
        // SA_RESTART: a forwarded blocking syscall must return EINTR so the
        // dispatch loop regains control and can deliver.
        _ => (chimera_sigcatch as *const () as usize, libc::SA_SIGINFO),
    };
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = host;
        sa.sa_flags = flags;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(signo as i32, &sa, ptr::null_mut());
    }
}

/// Whether `signo`'s default action does not terminate the process: Ign
/// (SIGCHLD/SIGURG/SIGWINCH), Cont (SIGCONT), or Stop (SIGSTOP/SIGTSTP/
/// SIGTTIN/SIGTTOU). Chimera does not yet model job control, so delivering
/// such a signal with a SIG_DFL disposition is a no-op.
fn default_action_discards(signo: u32) -> bool {
    matches!(signo, 17 | 18 | 19 | 20 | 21 | 22 | 23 | 28)
}

/// Whether the kernel discards `signo` at generation when its disposition is
/// `SIG_DFL`: the kernel's ignore set — SIGCHLD, SIGCONT (to a process that is
/// not stopped), SIGURG, SIGWINCH. Narrower than [`default_action_discards`]:
/// generating a default-Stop signal is not a no-op, it wakes the task to stop
/// it.
fn default_action_ignores(signo: u32) -> bool {
    matches!(signo, 17 | 18 | 23 | 28)
}

/// Carry out a signal's default action when it reaches delivery with a SIG_DFL
/// disposition (the disposition reverted to default while the signal was already
/// pending). The fatal signals — those whose default action is Term or Core —
/// terminate the guest by re-raising on the host with the default disposition
/// restored and the signal unblocked, so the host kernel produces the correct
/// termination status (and core dump). The signals whose default action is Ign,
/// Cont, or Stop are dropped, since Chimera does not yet model job control.
fn default_action(signo: u32) {
    if default_action_discards(signo) {
        return;
    }
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = libc::SIG_DFL;
        sa.sa_flags = 0;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(signo as i32, &sa, ptr::null_mut());
        let mut set: libc::sigset_t = mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, signo as i32);
        libc::sigprocmask(libc::SIG_UNBLOCK, &set, ptr::null_mut());
        libc::raise(signo as i32); // fatal default action: does not return
    }
}

/// The built-in `rt_sigreturn` restorer, used for handlers registered without
/// `SA_RESTORER`. A one-page guest mapping holding `mov eax, 15; syscall`,
/// readable (so the translator can read it) but not executable (W^X).
static RESTORER_INIT: Once = Once::new();
static RESTORER_ADDR: AtomicU64 = AtomicU64::new(0);

fn builtin_restorer() -> u64 {
    RESTORER_INIT.call_once(|| {
        // mov eax, 15 (rt_sigreturn) ; syscall
        let code: [u8; 7] = [0xB8, 0x0F, 0x00, 0x00, 0x00, 0x0F, 0x05];
        unsafe {
            let p = libc::mmap(
                ptr::null_mut(),
                4096,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            assert!(p != libc::MAP_FAILED, "signal restorer mmap failed");
            ptr::copy_nonoverlapping(code.as_ptr(), p as *mut u8, code.len());
            libc::mprotect(p, 4096, libc::PROT_READ);
            RESTORER_ADDR.store(p as u64, Ordering::Release);
        }
    });
    RESTORER_ADDR.load(Ordering::Acquire)
}

/// The guest's signal-disposition table, shared by every thread of the process.
/// POSIX (and a `clone(CLONE_SIGHAND)` thread) keeps signal dispositions
/// process-wide: `rt_sigaction` on one thread is visible to all, and a
/// thread-directed signal (e.g. JSC's `SIGPWR` stop-the-world) is delivered
/// against the disposition the installing thread set, not a per-thread default.
/// The host disposition is already process-wide (one `chimera_sigcatch`), so the
/// guest table must match; a per-thread table would have a freshly cloned thread
/// run the fatal default action for a signal the process actually handles.
pub type SharedSigTable = Arc<Mutex<[GuestSigaction; NSIG]>>;

/// Create a process's shared disposition table, all signals at their default.
pub fn new_shared_table() -> SharedSigTable {
    Arc::new(Mutex::new([GuestSigaction::default(); NSIG]))
}

/// Per-thread guest signal state. The disposition `table` is shared across the
/// thread group ([`SharedSigTable`]); the blocked mask, the alternate signal
/// stack, and the suspend-saved mask are per-thread, as POSIX requires.
pub struct Signals {
    table: SharedSigTable,
    /// This thread's pending-signal state, written by the host catcher (via
    /// the raw pointer published in `ThreadState`) and drained by this
    /// thread's dispatch loop. The `Arc` keeps the set alive for as long as
    /// the thread can receive signals.
    pending: Arc<PendingSet>,
    /// Currently blocked signal mask (emulated in software).
    pub blocked: u64,
    /// Alternate signal stack as `(ss_sp, ss_size, ss_flags)`, if set.
    altstack: Option<(u64, u64, i32)>,
    /// Mask to restore once the next handler returns, set by `sigsuspend`. While
    /// a `sigsuspend` is in flight the live `blocked` is its temporary mask; the
    /// signal frame must save this pre-suspend mask so the handler's
    /// `rt_sigreturn` restores it rather than the temporary one.
    saved_mask: Option<u64>,
}

impl Signals {
    /// Build a thread's signal state over the process's shared disposition table.
    /// A `clone(CLONE_VM)` child passes the same [`SharedSigTable`] so the two
    /// threads see one disposition table; the per-thread fields start cleared.
    pub fn new(table: SharedSigTable) -> Self {
        Self {
            table,
            pending: Arc::new(PendingSet::new()),
            blocked: 0,
            altstack: None,
            saved_mask: None,
        }
    }

    /// The address of this thread's [`PendingSet`], for publication in the
    /// gs-addressed `ThreadState` so the host catcher can record signals on
    /// the thread that caught them.
    pub fn pending_set_ptr(&self) -> *const PendingSet {
        Arc::as_ptr(&self.pending)
    }

    /// Atomically remove and return the lowest-numbered deliverable (pending
    /// and not blocked) signal on this thread, or `None`.
    pub fn pending_take_one(&self) -> Option<u32> {
        self.pending.take(!self.blocked)
    }

    /// Clear the pending-signal state in a freshly forked child. POSIX requires
    /// a child to start with an empty pending set, but the host `fork` copied
    /// the parent's pending bitmask, real-time queue depths, and ring indices.
    /// The disposition table and blocked mask are correctly inherited.
    pub fn reset_pending_after_fork(&self) {
        self.pending.reset();
    }

    /// Set the guest's blocked mask, stripping the unblockable signals. Every
    /// write to `blocked` funnels through here so the host mask never drifts
    /// from the emulated one.
    fn set_blocked(&mut self, mask: u64) {
        // SIGKILL and SIGSTOP can never be blocked.
        self.blocked = mask & !((1u64 << (SIGKILL - 1)) | (1u64 << (SIGSTOP - 1)));
        mirror_host_blocked(self.blocked);
    }

    /// The signals whose generation right now would be discarded outright:
    /// disposition `SIG_IGN`, or `SIG_DFL` where the kernel ignores the signal
    /// at generation (see [`default_action_ignores`]). The arrival of such a
    /// signal never interrupts a wait.
    fn discarded_mask(&self) -> u64 {
        let table = self.table.lock().unwrap();
        let mut mask = 0u64;
        for i in 0..NSIG {
            let h = table[i].handler;
            if h == SIG_IGN || (h == SIG_DFL && default_action_ignores(i as u32 + 1)) {
                mask |= 1u64 << i;
            }
        }
        mask
    }

    /// The stop signals whose disposition is currently the default, so their
    /// generation stops the process: SIGSTOP (whose disposition can never
    /// change), plus SIGTSTP/SIGTTIN/SIGTTOU while at `SIG_DFL`.
    fn dfl_stop_mask(&self) -> u64 {
        let table = self.table.lock().unwrap();
        let mut mask = 1u64 << (SIGSTOP - 1);
        for signo in [20u32, 21, 22] {
            // SIGTSTP, SIGTTIN, SIGTTOU
            if table[signo as usize - 1].handler == SIG_DFL {
                mask |= 1u64 << (signo - 1);
            }
        }
        mask
    }

    /// Push the guest's blocked mask onto the host thread, for run-loop entry.
    pub fn mirror_host_mask(&self) {
        mirror_host_blocked(self.blocked);
    }

    /// Service guest `rt_sigaction`. Records the new disposition, returns the old
    /// one through `oldact`, and mirrors the host disposition.
    pub fn sigaction(&mut self, signo: u64, act: u64, oldact: u64) -> SyscallResult {
        if signo == 0 || signo as usize > NSIG || signo == SIGKILL || signo == SIGSTOP {
            return SyscallResult::Error(libc::EINVAL);
        }
        let idx = signo as usize - 1;
        let mut table = self.table.lock().unwrap();
        let prev = table[idx];
        if oldact != 0 {
            let ka = KernelSigaction {
                handler: prev.handler,
                flags: prev.flags,
                restorer: prev.restorer,
                mask: prev.mask,
            };
            unsafe { (oldact as *mut KernelSigaction).write_unaligned(ka) };
        }
        if act != 0 {
            let a = unsafe { (act as *const KernelSigaction).read_unaligned() };
            table[idx] = GuestSigaction {
                handler: a.handler,
                flags: a.flags,
                restorer: a.restorer,
                mask: a.mask,
            };
            install_host(signo as usize, a.handler);
        }
        SyscallResult::Ok(0)
    }

    /// Service guest `rt_sigprocmask`, emulating the blocked mask in software.
    pub fn sigprocmask(&mut self, how: i32, set: u64, oldset: u64) -> SyscallResult {
        if oldset != 0 {
            unsafe { (oldset as *mut u64).write_unaligned(self.blocked) };
        }
        if set != 0 {
            let s = unsafe { (set as *const u64).read_unaligned() };
            let new = match how {
                libc::SIG_BLOCK => self.blocked | s,
                libc::SIG_UNBLOCK => self.blocked & !s,
                libc::SIG_SETMASK => s,
                _ => return SyscallResult::Error(libc::EINVAL),
            };
            self.set_blocked(new);
        }
        SyscallResult::Ok(0)
    }

    /// Service guest `rt_sigpending`: the union of Chimera's recorded pending
    /// set and the host's (see [`host_pending_snapshot`]).
    pub fn sigpending(&self, set: u64, sigsetsize: u64) -> SyscallResult {
        if sigsetsize as usize != mem::size_of::<u64>() {
            return SyscallResult::Error(libc::EINVAL);
        }
        if set != 0 {
            let pending = self.pending.snapshot() | host_pending_snapshot();
            unsafe { (set as *mut u64).write_unaligned(pending) };
        }
        SyscallResult::Ok(0)
    }

    /// Service guest `rt_sigsuspend`: atomically install the temporary mask,
    /// block the host thread until a signal deliverable under that mask is
    /// pending, and return `EINTR`. The pre-suspend mask is stashed in
    /// `saved_mask` so that when the dispatch loop delivers the waking signal,
    /// the handler's frame saves the original mask and `rt_sigreturn` restores
    /// it. `sigsuspend` never restarts.
    pub fn sigsuspend(&mut self, mask: u64, sigsetsize: u64) -> SyscallResult {
        if sigsetsize as usize != mem::size_of::<u64>() {
            return SyscallResult::Error(libc::EINVAL);
        }
        let temp = if mask != 0 {
            unsafe { (mask as *const u64).read_unaligned() }
        } else {
            0
        };
        self.saved_mask = Some(self.blocked);
        self.set_blocked(temp);
        wait_for_signal(&self.pending, self.blocked);
        SyscallResult::Error(libc::EINTR)
    }

    /// Service guest `rt_sigtimedwait`: synchronously accept one signal in `set`,
    /// blocking up to `timeout`, and report it through `info` without running its
    /// handler. Answered from the emulated pending set rather than the host,
    /// which never sees the guest's pending signals.
    pub fn sigtimedwait(
        &mut self,
        set: u64,
        info: u64,
        timeout: u64,
        sigsetsize: u64,
    ) -> SyscallResult {
        if sigsetsize as usize != mem::size_of::<u64>() {
            return SyscallResult::Error(libc::EINVAL);
        }
        let want = if set != 0 {
            unsafe { (set as *const u64).read_unaligned() }
        } else {
            0
        };
        // SIGKILL and SIGSTOP cannot be waited for.
        let want = want & !((1u64 << (SIGKILL - 1)) | (1u64 << (SIGSTOP - 1)));

        let deadline = if timeout != 0 {
            let ts = unsafe { (timeout as *const libc::timespec).read_unaligned() };
            if ts.tv_sec < 0 || ts.tv_nsec < 0 || ts.tv_nsec >= 1_000_000_000 {
                return SyscallResult::Error(libc::EINVAL);
            }
            Some(monotonic_ns() + ts.tv_sec as u128 * 1_000_000_000 + ts.tv_nsec as u128)
        } else {
            None
        };

        match wait_for_set(
            &self.pending,
            want,
            self.blocked,
            self.discarded_mask(),
            self.dfl_stop_mask(),
            deadline,
        ) {
            WaitResult::Got(signo) => {
                if info != 0 {
                    let mut si: SigInfo = unsafe { mem::zeroed() };
                    si.si_signo = signo as i32;
                    unsafe { (info as *mut SigInfo).write_unaligned(si) };
                }
                SyscallResult::Ok(signo as i64)
            }
            WaitResult::Timeout => SyscallResult::Error(libc::EAGAIN),
            WaitResult::Interrupted => SyscallResult::Error(libc::EINTR),
        }
    }

    /// Service guest `sigaltstack`.
    pub fn sigaltstack(&mut self, ss: u64, old_ss: u64) -> SyscallResult {
        if old_ss != 0 {
            let cur = match self.altstack {
                Some((sp, size, fl)) => StackT {
                    ss_sp: sp,
                    ss_flags: fl,
                    _pad: 0,
                    ss_size: size,
                },
                None => StackT {
                    ss_sp: 0,
                    ss_flags: SS_DISABLE,
                    _pad: 0,
                    ss_size: 0,
                },
            };
            unsafe { (old_ss as *mut StackT).write_unaligned(cur) };
        }
        if ss != 0 {
            let n = unsafe { (ss as *const StackT).read_unaligned() };
            if n.ss_flags & SS_DISABLE != 0 {
                self.altstack = None;
            } else {
                self.altstack = Some((n.ss_sp, n.ss_size, n.ss_flags));
            }
        }
        SyscallResult::Ok(0)
    }

    /// Reset signal state across `execve` (POSIX): caught handlers revert to
    /// their default, ignored ones stay ignored, the blocked mask is preserved,
    /// and the alternate stack is dropped.
    pub fn on_execve(&mut self) {
        self.clear_sighand();
        self.altstack = None;
    }

    /// Revert every caught disposition to its default, leaving `SIG_IGN` and
    /// the blocked mask alone — the kernel's `flush_signal_handlers`, the
    /// shared core of `execve`'s reset and `clone3`'s `CLONE_CLEAR_SIGHAND`.
    pub fn clear_sighand(&mut self) {
        let mut table = self.table.lock().unwrap();
        for i in 0..NSIG {
            let h = table[i].handler;
            if h != SIG_DFL && h != SIG_IGN {
                table[i] = GuestSigaction::default();
                install_host(i + 1, SIG_DFL);
            }
        }
    }

    /// Build a signal frame on the guest stack and redirect `state` to the
    /// handler for `signo`. Called only at a safe point (block boundary).
    ///
    /// `restart` carries `(resume rip after the interrupted syscall, original
    /// syscall number)` when the signal interrupted a restartable forwarded
    /// syscall that returned `EINTR`. If the handler has `SA_RESTART`, the saved
    /// context is rewound so the handler returns into a re-execution of the
    /// syscall rather than seeing `EINTR` — mirroring the kernel, which rewinds
    /// the saved `rip` by the 2-byte `syscall` and restores the original `rax`.
    pub fn deliver(&mut self, state: &mut ThreadState, signo: u32, restart: Option<(u64, u64)>) {
        // Copy the disposition out of the shared table and release the lock; the
        // rest of delivery touches only this thread's own state.
        let act = self.table.lock().unwrap()[signo as usize - 1];
        // The disposition may have changed since the host caught the signal.
        // SIG_IGN discards it; SIG_DFL means we must carry out the kernel's
        // default action rather than jump to 0/1.
        if act.handler == SIG_IGN {
            return;
        }
        if act.handler == SIG_DFL {
            default_action(signo);
            return;
        }

        let on_alt = act.flags & libc::SA_ONSTACK as u64 != 0 && self.altstack.is_some();
        let mut sp = if on_alt {
            let (base, size, _) = self.altstack.unwrap();
            base + size
        } else {
            // Skip the red zone the System V ABI reserves below rsp.
            state.regs[RSP] - 128
        };

        // Save the extended state below the frame, 64-byte aligned.
        let fpsize = state.fpstate.len();
        sp -= fpsize as u64;
        sp &= !63;
        let fpstate_ptr = sp;
        unsafe {
            ptr::copy_nonoverlapping(state.fpstate.as_ptr(), fpstate_ptr as *mut u8, fpsize);
        }

        // Place the frame so that rsp % 16 == 8 at handler entry (as if `call`ed).
        sp -= mem::size_of::<RtSigframe>() as u64;
        sp &= !15;
        sp -= 8;
        let frame = sp as *mut RtSigframe;

        let mut f: RtSigframe = unsafe { mem::zeroed() };
        f.pretcode = if act.flags & SA_RESTORER != 0 {
            act.restorer
        } else {
            builtin_restorer()
        };
        // After a `sigsuspend`, the frame must restore the pre-suspend mask, not
        // the temporary one that is live while suspended.
        f.uc.uc_sigmask[0] = self.saved_mask.take().unwrap_or(self.blocked);
        f.uc.uc_stack = match self.altstack {
            Some((base, size, fl)) => StackT {
                ss_sp: base,
                ss_flags: fl,
                _pad: 0,
                ss_size: size,
            },
            None => StackT {
                ss_sp: 0,
                ss_flags: SS_DISABLE,
                _pad: 0,
                ss_size: 0,
            },
        };
        // When the signal interrupted a restartable syscall and the handler asked
        // to restart, the saved context resumes by re-executing the `syscall`
        // (rip back by 2) with its original number in rax; otherwise it resumes
        // after the syscall with the EINTR result already in rax.
        let (resume_rip, resume_rax) = match restart {
            Some((next_ip, nr))
                if next_ip == state.rip && act.flags & libc::SA_RESTART as u64 != 0 =>
            {
                (next_ip - 2, nr)
            }
            _ => (state.rip, state.regs[RAX]),
        };

        {
            let mc = &mut f.uc.uc_mcontext;
            mc.r8 = state.regs[8];
            mc.r9 = state.regs[9];
            mc.r10 = state.regs[10];
            mc.r11 = state.regs[11];
            mc.r12 = state.regs[12];
            mc.r13 = state.regs[13];
            mc.r14 = state.regs[14];
            mc.r15 = state.regs[15];
            mc.rdi = state.regs[RDI];
            mc.rsi = state.regs[RSI];
            mc.rbp = state.regs[RBP];
            mc.rbx = state.regs[RBX];
            mc.rdx = state.regs[RDX];
            mc.rax = resume_rax;
            mc.rcx = state.regs[RCX];
            mc.rsp = state.regs[RSP];
            mc.rip = resume_rip;
            mc.eflags = state.rflags;
            mc.fpstate = fpstate_ptr;
        }
        f.info = self.pending.captured(signo);
        f.info.si_signo = signo as i32;
        unsafe { ptr::write(frame, f) };

        let siginfo_ptr = unsafe { ptr::addr_of!((*frame).info) } as u64;
        let uc_ptr = unsafe { ptr::addr_of!((*frame).uc) } as u64;

        // Block sa_mask (and this signal, unless SA_NODEFER) for the handler.
        let mut handler_mask = self.blocked | act.mask;
        if act.flags & libc::SA_NODEFER as u64 == 0 {
            handler_mask |= 1u64 << (signo - 1);
        }
        self.set_blocked(handler_mask);

        // Enter the handler: rdi=signo, rsi=&siginfo, rdx=&ucontext.
        state.regs[RDI] = signo as u64;
        state.regs[RSI] = siginfo_ptr;
        state.regs[RDX] = uc_ptr;
        state.regs[RAX] = 0;
        state.regs[RSP] = sp;
        state.rip = act.handler;
        state.rflags &= !((1u64 << 10) | (1u64 << 8)); // clear DF, TF

        if act.flags & libc::SA_RESETHAND as u64 != 0 {
            self.table.lock().unwrap()[signo as usize - 1].handler = SIG_DFL;
            install_host(signo as usize, SIG_DFL);
        }
    }

    /// Restore the pre-signal context from the frame on the guest stack, on
    /// guest `rt_sigreturn`.
    pub fn restore(&mut self, state: &mut ThreadState) {
        let frame = (state.regs[RSP] - 8) as *const RtSigframe;
        let f = unsafe { &*frame };
        let mc = &f.uc.uc_mcontext;
        state.regs[8] = mc.r8;
        state.regs[9] = mc.r9;
        state.regs[10] = mc.r10;
        state.regs[11] = mc.r11;
        state.regs[12] = mc.r12;
        state.regs[13] = mc.r13;
        state.regs[14] = mc.r14;
        state.regs[15] = mc.r15;
        state.regs[RDI] = mc.rdi;
        state.regs[RSI] = mc.rsi;
        state.regs[RBP] = mc.rbp;
        state.regs[RBX] = mc.rbx;
        state.regs[RDX] = mc.rdx;
        state.regs[RAX] = mc.rax;
        state.regs[RCX] = mc.rcx;
        state.regs[RSP] = mc.rsp;
        state.rip = mc.rip;
        state.rflags = mc.eflags;
        if mc.fpstate != 0 {
            let fpsize = state.fpstate.len();
            unsafe {
                ptr::copy_nonoverlapping(
                    mc.fpstate as *const u8,
                    state.fpstate.as_mut_ptr(),
                    fpsize,
                );
            }
        }
        self.set_blocked(f.uc.uc_sigmask[0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The current host thread mask as a sigset bitmask.
    fn current_host_mask() -> u64 {
        unsafe {
            let mut s: libc::sigset_t = mem::zeroed();
            libc::pthread_sigmask(libc::SIG_SETMASK, ptr::null(), &mut s);
            sigset_mask(&s)
        }
    }

    #[test]
    fn sigset_mask_roundtrips_host_sigset() {
        let mask = (1u64 << (libc::SIGUSR1 - 1)) | (1u64 << (42 - 1)) | (1u64 << (NSIG - 1));
        assert_eq!(sigset_mask(&host_sigset(mask)), mask);
    }

    #[test]
    fn host_mask_guard_restores_caller_mask() {
        let caller = 1u64 << (libc::SIGUSR2 - 1);
        mirror_host_blocked(caller);
        {
            let _guard = HostMaskGuard::save();
            mirror_host_blocked(1u64 << (libc::SIGHUP - 1));
        }
        assert_eq!(current_host_mask(), caller);
        mirror_host_blocked(0);
    }

    #[test]
    fn set_blocked_mirrors_guest_mask_onto_host() {
        let guest = 1u64 << (libc::SIGUSR1 - 1);
        let _guard = HostMaskGuard::save();
        let mut signals = Signals::new(new_shared_table());
        signals.set_blocked(guest | (1u64 << (SIGKILL - 1)));
        assert_eq!(current_host_mask(), guest);
        signals.set_blocked(0);
        assert_eq!(current_host_mask(), 0);
    }
}
