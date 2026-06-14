//! Guest signal virtualization.
//!
//! Chimera never lets the host kernel deliver a signal straight to a guest
//! handler: that would jump the host thread natively into untranslated guest
//! code (outside the sandbox, and a hard fault under W^X). Instead Chimera owns
//! the guest's signal disposition. `rt_sigaction`/`rt_sigprocmask`/`sigaltstack`
//! are intercepted into [`Signals`]; for any caught signal a single host-side
//! catcher ([`chimera_sigcatch`]) is installed that does nothing but record the
//! signal as pending. The dispatch loop drains the pending set at a safe point
//! (a block boundary), builds a kernel-ABI `rt_sigframe` on the guest stack with
//! [`Signals::deliver`], and re-enters the translator at the handler — so the
//! handler runs translated, in-sandbox. The handler returns through its restorer
//! (`sa_restorer`, or a built-in `rt_sigreturn` stub), whose `rt_sigreturn` is
//! intercepted into [`Signals::restore`].
//!
//! A blocking host syscall forwarded on the guest's behalf is interrupted
//! (the catcher is installed without `SA_RESTART`), so the loop regains control
//! and delivers at its next iteration.

use std::{
    cell::UnsafeCell,
    mem, ptr,
    sync::{
        Once,
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

// Guest register-file indices (see `crate::arch::dispatch`). r8..r15 are 8..15.
const RAX: usize = 0;
const RBX: usize = 1;
const RCX: usize = 2;
const RDX: usize = 3;
const RSI: usize = 4;
const RDI: usize = 5;
const RBP: usize = 6;
const RSP: usize = 7;

/// One guest signal disposition, in the raw `rt_sigaction` ABI shape.
#[derive(Clone, Copy, Default)]
struct GuestSigaction {
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

/// `siginfo_t` is 128 bytes; only `si_signo` is populated for now.
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

/// Process-wide set of signals the host catcher has seen but Chimera has not yet
/// delivered to the guest. A set bit means "at least one instance pending". For
/// real-time signals the depth is tracked in [`COUNTS`]; standard signals simply
/// coalesce. Written by [`chimera_sigcatch`] (async-signal-safely, via atomic
/// OR), drained by the dispatch loop.
static PENDING: AtomicU64 = AtomicU64::new(0);

/// Per-signal queue depth for real-time signals (indexed by `signo - 1`). Bumped
/// by the catcher and decremented as each queued instance is delivered, so the
/// `PENDING` bit for an RT signal stays set until its queue drains. Standard
/// signals do not use this.
static COUNTS: [AtomicU32; NSIG] = [const { AtomicU32::new(0) }; NSIG];

/// Size of the kernel `siginfo_t` on x86-64.
const SIGINFO_SIZE: usize = 128;

const _: () = assert!(mem::size_of::<SigInfo>() == SIGINFO_SIZE);

/// A captured `siginfo_t`, one slot per signal. Written by the catcher (a signal
/// handler) and read by the dispatch loop, so access is unsynchronized; the
/// `memcpy` is async-signal-safe and the dispatch loop only reads a slot after
/// the catcher that filled it has run. For a real-time signal with several
/// queued instances this keeps only the most recent `siginfo` (full per-instance
/// queueing is a follow-up).
struct SigInfoSlot(UnsafeCell<[u8; SIGINFO_SIZE]>);

// SAFETY: writes happen only in the async-signal-safe catcher; reads only at a
// dispatch-loop safe point after the corresponding catcher has completed.
unsafe impl Sync for SigInfoSlot {}

static SIGINFO: [SigInfoSlot; NSIG] =
    [const { SigInfoSlot(UnsafeCell::new([0u8; SIGINFO_SIZE])) }; NSIG];

/// Host signal catcher. Runs asynchronously on whatever context the host thread
/// happened to be in (translated guest code or Chimera Rust), so it must be
/// async-signal-safe: it only touches plain global atomics and does a fixed-size
/// `memcpy` — no TLS, no allocation, no locks, no panics. Installed with
/// `SA_SIGINFO`, so it captures the kernel's `siginfo_t` for the guest handler.
/// For a real-time signal it bumps the queue depth before marking the bit, so a
/// drainer that observes the bit set always sees a positive count.
extern "C" fn chimera_sigcatch(
    signo: libc::c_int,
    info: *const libc::siginfo_t,
    _uc: *mut libc::c_void,
) {
    if signo >= 1 && signo as usize <= NSIG {
        let idx = signo as usize - 1;
        if !info.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(
                    info as *const u8,
                    SIGINFO[idx].0.get() as *mut u8,
                    SIGINFO_SIZE,
                );
            }
        }
        if signo as u32 >= SIGRTMIN_KERNEL {
            COUNTS[idx].fetch_add(1, Ordering::AcqRel);
        }
        PENDING.fetch_or(1u64 << (signo as u64 - 1), Ordering::Release);
    }
}

/// Copy the captured `siginfo_t` for `signo` out of its slot.
fn captured_siginfo(signo: u32) -> SigInfo {
    let mut si: SigInfo = unsafe { mem::zeroed() };
    unsafe {
        ptr::copy_nonoverlapping(
            SIGINFO[signo as usize - 1].0.get() as *const u8,
            &mut si as *mut SigInfo as *mut u8,
            SIGINFO_SIZE,
        );
    }
    si
}

/// Snapshot the set of signals the host catcher has recorded as pending but the
/// dispatch loop has not yet delivered, as a sigset bitmask. Unblocked pending
/// signals are drained at each block boundary, so by the time the guest observes
/// this (at a syscall) the remaining bits are the blocked-and-pending ones.
pub fn pending_snapshot() -> u64 {
    PENDING.load(Ordering::Acquire)
}

/// Atomically remove and return the lowest-numbered pending signal within the
/// `allowed` set, or `None`. A standard signal clears its bit (coalesced); a
/// real-time signal consumes one queued instance and keeps its bit set while
/// more remain.
fn pending_take(allowed: u64) -> Option<u32> {
    loop {
        let cur = PENDING.load(Ordering::Acquire);
        let avail = cur & allowed;
        if avail == 0 {
            return None;
        }
        let bit = avail & avail.wrapping_neg();
        let signo = bit.trailing_zeros() + 1;

        if signo >= SIGRTMIN_KERNEL {
            let idx = signo as usize - 1;
            let depth = COUNTS[idx].load(Ordering::Acquire);
            if depth == 0 {
                // Bit set with an empty queue (already drained elsewhere): clear
                // it and re-pick.
                PENDING.fetch_and(!bit, Ordering::AcqRel);
                continue;
            }
            if COUNTS[idx]
                .compare_exchange_weak(depth, depth - 1, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            if depth == 1 {
                // Took the last queued instance; clear the bit, then re-arm it if
                // the catcher enqueued another in the meantime.
                PENDING.fetch_and(!bit, Ordering::AcqRel);
                if COUNTS[idx].load(Ordering::Acquire) > 0 {
                    PENDING.fetch_or(bit, Ordering::Release);
                }
            }
            return Some(signo);
        }

        // Standard signal: coalesced, clear the single bit.
        let new = cur & !bit;
        if PENDING
            .compare_exchange_weak(cur, new, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return Some(signo);
        }
    }
}

/// Atomically remove and return the lowest-numbered deliverable (pending and not
/// `blocked`) signal, or `None`. Blocked signals stay pending.
pub fn pending_take_one(blocked: u64) -> Option<u32> {
    pending_take(!blocked)
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

/// Block the calling host thread until a signal deliverable under `blocked`
/// (pending and not blocked) has been recorded by the host catcher. Race-free:
/// all host signals are blocked first, the already-pending set is checked, then
/// `sigsuspend` atomically unblocks the deliverable ones and waits, so a signal
/// arriving in the window is not lost. Returns once such a signal is pending,
/// leaving it in `PENDING` for the dispatch loop to deliver.
fn wait_for_signal(blocked: u64) {
    unsafe {
        let mut all: libc::sigset_t = mem::zeroed();
        libc::sigfillset(&mut all);
        let mut prev: libc::sigset_t = mem::zeroed();
        libc::pthread_sigmask(libc::SIG_BLOCK, &all, &mut prev);
        // During the wait, keep the guest's blocked signals masked so only a
        // deliverable signal wakes us.
        let wait = host_sigset(blocked);
        while pending_snapshot() & !blocked == 0 {
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
/// elapses, or a deliverable signal outside `set` arrives. Race-free in the same
/// way as [`wait_for_signal`]; `ppoll` provides the timed wait that a signal
/// interrupts.
fn wait_for_set(set: u64, blocked: u64, deadline: Option<u128>) -> WaitResult {
    unsafe {
        let mut all: libc::sigset_t = mem::zeroed();
        libc::sigfillset(&mut all);
        let mut prev: libc::sigset_t = mem::zeroed();
        libc::pthread_sigmask(libc::SIG_BLOCK, &all, &mut prev);
        let mut empty: libc::sigset_t = mem::zeroed();
        libc::sigemptyset(&mut empty);

        let result = loop {
            if let Some(signo) = pending_take(set) {
                break WaitResult::Got(signo);
            }
            if pending_snapshot() & !blocked & !set != 0 {
                break WaitResult::Interrupted;
            }
            let remaining = match deadline {
                Some(d) => {
                    let now = monotonic_ns();
                    if now >= d {
                        break WaitResult::Timeout;
                    }
                    let rem = d - now;
                    Some(libc::timespec {
                        tv_sec: (rem / 1_000_000_000) as libc::time_t,
                        tv_nsec: (rem % 1_000_000_000) as i64,
                    })
                }
                None => None,
            };
            let ts_ptr = remaining
                .as_ref()
                .map_or(ptr::null(), |ts| ts as *const libc::timespec);
            libc::ppoll(ptr::null_mut(), 0, ts_ptr, &empty);
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

/// Carry out a signal's default action when it reaches delivery with a SIG_DFL
/// disposition (the disposition reverted to default while the signal was already
/// pending). The fatal signals — those whose default action is Term or Core —
/// terminate the guest by re-raising on the host with the default disposition
/// restored and the signal unblocked, so the host kernel produces the correct
/// termination status (and core dump). The signals whose default action is Ign,
/// Cont, or Stop are dropped, since Chimera does not yet model job control.
fn default_action(signo: u32) {
    // SIGCHLD/SIGURG/SIGWINCH (Ign), SIGCONT (Cont), and SIGSTOP/SIGTSTP/
    // SIGTTIN/SIGTTOU (Stop) do not terminate; drop them.
    if matches!(signo, 17 | 18 | 19 | 20 | 21 | 22 | 23 | 28) {
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

/// Per-process guest signal state: the disposition table, the blocked mask, and
/// the alternate signal stack.
pub struct Signals {
    table: [GuestSigaction; NSIG],
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
    pub fn new() -> Self {
        Self {
            table: [GuestSigaction::default(); NSIG],
            blocked: 0,
            altstack: None,
            saved_mask: None,
        }
    }

    /// Service guest `rt_sigaction`. Records the new disposition, returns the old
    /// one through `oldact`, and mirrors the host disposition.
    pub fn sigaction(&mut self, signo: u64, act: u64, oldact: u64) -> SyscallResult {
        if signo == 0 || signo as usize > NSIG || signo == SIGKILL || signo == SIGSTOP {
            return SyscallResult::Error(libc::EINVAL);
        }
        let idx = signo as usize - 1;
        let prev = self.table[idx];
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
            self.table[idx] = GuestSigaction {
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
            match how {
                libc::SIG_BLOCK => self.blocked |= s,
                libc::SIG_UNBLOCK => self.blocked &= !s,
                libc::SIG_SETMASK => self.blocked = s,
                _ => return SyscallResult::Error(libc::EINVAL),
            }
            // SIGKILL and SIGSTOP can never be blocked.
            self.blocked &= !((1u64 << (SIGKILL - 1)) | (1u64 << (SIGSTOP - 1)));
        }
        SyscallResult::Ok(0)
    }

    /// Service guest `rt_sigpending`: report the emulated pending set. The host
    /// kernel never sees the guest's pending signals (the catcher clears them
    /// into `PENDING`), so this must be answered from Chimera's own state.
    pub fn sigpending(&self, set: u64, sigsetsize: u64) -> SyscallResult {
        if sigsetsize as usize != mem::size_of::<u64>() {
            return SyscallResult::Error(libc::EINVAL);
        }
        if set != 0 {
            unsafe { (set as *mut u64).write_unaligned(pending_snapshot()) };
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
        // SIGKILL and SIGSTOP can never be blocked.
        let temp = temp & !((1u64 << (SIGKILL - 1)) | (1u64 << (SIGSTOP - 1)));
        self.saved_mask = Some(self.blocked);
        self.blocked = temp;
        wait_for_signal(temp);
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

        match wait_for_set(want, self.blocked, deadline) {
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
        for i in 0..NSIG {
            let h = self.table[i].handler;
            if h != SIG_DFL && h != SIG_IGN {
                self.table[i] = GuestSigaction::default();
                install_host(i + 1, SIG_DFL);
            }
        }
        self.altstack = None;
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
        let act = self.table[signo as usize - 1];
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
        f.info = captured_siginfo(signo);
        f.info.si_signo = signo as i32;
        unsafe { ptr::write(frame, f) };

        let siginfo_ptr = unsafe { ptr::addr_of!((*frame).info) } as u64;
        let uc_ptr = unsafe { ptr::addr_of!((*frame).uc) } as u64;

        // Block sa_mask (and this signal, unless SA_NODEFER) for the handler.
        self.blocked |= act.mask;
        if act.flags & libc::SA_NODEFER as u64 == 0 {
            self.blocked |= 1u64 << (signo - 1);
        }

        // Enter the handler: rdi=signo, rsi=&siginfo, rdx=&ucontext.
        state.regs[RDI] = signo as u64;
        state.regs[RSI] = siginfo_ptr;
        state.regs[RDX] = uc_ptr;
        state.regs[RAX] = 0;
        state.regs[RSP] = sp;
        state.rip = act.handler;
        state.rflags &= !((1u64 << 10) | (1u64 << 8)); // clear DF, TF

        if act.flags & libc::SA_RESETHAND as u64 != 0 {
            self.table[signo as usize - 1].handler = SIG_DFL;
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
        self.blocked = f.uc.uc_sigmask[0];
    }
}
