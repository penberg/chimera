//! Darwin guest-signal virtualization: the disposition table, blocked masks,
//! pending set, and delivery into translated guest handlers.
//!
//! A guest `sigaction` is never allowed to install its handler on the host
//! slot — the kernel would then call untranslated guest code directly, outside
//! the sandbox. Instead the disposition is recorded here, and the host slot
//! gets one of three things: the real `SIG_DFL`/`SIG_IGN` (the kernel's default
//! terminate/stop/ignore actions are exactly right and involve no guest code),
//! or [`chimera_sigcatch`] for a custom guest handler. The catcher records the
//! signal in the receiving thread's pending set; the arm64 dispatch loop
//! delivers it at the next block boundary by synthesizing a Darwin signal frame
//! on the guest stack and pointing the guest at its handler with the link
//! register set to `SIGRETURN_SENTINEL` — the handler's return is recognized by
//! the run loop, which restores the interrupted context from the (possibly
//! handler-modified) frame, the way `sigreturn` would.
//!
//! The guest's blocked mask is mirrored onto the host thread so kernel-side
//! generation semantics (a blocked fatal signal held pending, `SIGTTOU` from
//! `tcsetpgrp`) stay right; `SIGSEGV`/`SIGBUS` stay unblocked and owned by
//! [`super::fault`]. The host catcher is installed without `SA_RESTART`, so a
//! forwarded blocking syscall returns `EINTR` and the dispatch loop regains
//! control; whether the guest then sees `EINTR` or a restarted syscall is
//! decided at delivery from the guest handler's own `SA_RESTART` (see
//! [`Signals::deliver`]).

use std::{
    cell::UnsafeCell,
    mem, ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
    },
};

use crate::{SyscallResult, arch::dispatch::ThreadState, process::Process};

/// Darwin has 31 signals (`1..=31`), all coalescing — no real-time queues.
const NSIG: usize = 31;

const SIG_DFL: u64 = 0;
const SIG_IGN: u64 = 1;

/// The synchronous fault signals, whose host disposition belongs to
/// [`super::fault`]; the guest's disposition for them is recorded but never
/// installed on the host slot.
const SIGBUS: usize = 10;
const SIGSEGV: usize = 11;

/// One guest signal disposition. The guest's libc `_sigtramp` (the `sa_tramp`
/// slot of the new-action struct) is deliberately not kept: Chimera plays the
/// kernel's role and calls the handler with the C ABI directly, so the guest
/// trampoline is never used.
#[derive(Clone, Copy, Default)]
pub struct GuestSigaction {
    handler: u64,
    mask: u32,
    flags: i32,
}

/// `struct __sigaction` — the new-action shape the raw syscall reads.
#[repr(C)]
#[derive(Clone, Copy)]
struct KernelSigaction {
    handler: u64,
    tramp: u64,
    mask: u32,
    flags: i32,
}

/// `struct sigaction` — the old-action shape the raw syscall writes back.
#[repr(C)]
#[derive(Clone, Copy)]
struct OldSigaction {
    handler: u64,
    mask: u32,
    flags: i32,
}

/// The process-wide disposition table shared by every thread of the guest.
pub type SharedSigTable = Arc<Mutex<[GuestSigaction; NSIG]>>;

pub fn new_shared_table() -> SharedSigTable {
    Arc::new(Mutex::new([GuestSigaction::default(); NSIG]))
}

/// A captured `siginfo_t` slot, written by the async-signal-safe catcher and
/// read by the dispatch loop only after the write completed (the pending bit
/// is set with release ordering after the slot write).
struct SigInfoSlot(UnsafeCell<libc::siginfo_t>);

// SAFETY: writes happen only in the catcher; reads only at a dispatch-loop
// safe point ordered after the bit that published them. `siginfo_t` carries a
// raw pointer (`si_addr`), which is neither `Send` nor `Sync` by default, but
// the slot is plain captured data moved between threads by value, so both are
// sound.
unsafe impl Sync for SigInfoSlot {}
unsafe impl Send for SigInfoSlot {}

/// Per-thread pending-signal state: signals the host catcher has recorded on
/// this thread but the dispatch loop has not yet delivered. Standard Darwin
/// signals coalesce, so one bit and one `siginfo` slot per signal suffice.
pub struct PendingSet {
    bits: AtomicU32,
    siginfo: [SigInfoSlot; NSIG],
}

impl PendingSet {
    fn new() -> Self {
        Self {
            bits: AtomicU32::new(0),
            siginfo: [const { SigInfoSlot(UnsafeCell::new(unsafe { mem::zeroed() })) }; NSIG],
        }
    }

    /// Record one caught signal: capture its `siginfo`, then publish the bit.
    /// Async-signal-safe.
    fn record(&self, signo: u32, info: *const libc::siginfo_t) {
        let idx = signo as usize - 1;
        if idx >= NSIG {
            return;
        }
        if !info.is_null() {
            unsafe { ptr::copy_nonoverlapping(info, self.siginfo[idx].0.get(), 1) };
        }
        self.bits.fetch_or(1 << idx, Ordering::Release);
    }

    fn snapshot(&self) -> u32 {
        self.bits.load(Ordering::Acquire)
    }

    /// Remove and return the lowest-numbered signal that is pending and
    /// allowed by `allowed`, with its captured `siginfo`.
    fn take(&self, allowed: u32) -> Option<(u32, libc::siginfo_t)> {
        loop {
            let bits = self.bits.load(Ordering::Acquire);
            let ready = bits & allowed;
            if ready == 0 {
                return None;
            }
            let idx = ready.trailing_zeros();
            let info = unsafe { *self.siginfo[idx as usize].0.get() };
            if self
                .bits
                .compare_exchange(
                    bits,
                    bits & !(1 << idx),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Some((idx + 1, info));
            }
        }
    }
}

/// Whether the guest has installed a real handler for `signo` (not `SIG_DFL`
/// or `SIG_IGN`). Used by the synchronous-fault path to decide whether to
/// deliver a fault into the guest or let it terminate the process. `try_lock`
/// because it runs in the async fault handler: the faulting thread could hold
/// the table lock (a fault while `sigaction` writes a bad guest `oldact`
/// pointer), and a blocking lock would deadlock — treat contention as "no
/// handler" and fall through to the crash report.
pub fn guest_handles(table: &SharedSigTable, signo: u32) -> bool {
    if signo == 0 || signo as usize > NSIG {
        return false;
    }
    match table.try_lock() {
        Ok(t) => {
            let h = t[signo as usize - 1].handler;
            h != SIG_DFL && h != SIG_IGN
        }
        Err(_) => false,
    }
}

/// Record a synchronous fault as a pending signal on `pending` (a raw
/// [`PendingSet`] pointer published in `ThreadState`), for the run loop to
/// deliver into the guest handler at the next boundary. Async-signal-safe.
///
/// # Safety
/// `pending` must be a valid `PendingSet` for the calling thread, or null.
pub unsafe fn record_fault(pending: *const PendingSet, signo: u32, info: *const libc::siginfo_t) {
    if !pending.is_null() {
        unsafe { (*pending).record(signo, info) };
    }
}

/// Pending set for a signal caught on a host thread that runs no guest code
/// (no `ThreadState` in its TSD slot) — a process-directed signal the kernel
/// routed to a runtime-internal thread. Drained by whichever guest thread
/// reaches a delivery point first, which matches process-directed semantics.
static STRAY_PENDING: PendingSet = PendingSet {
    bits: AtomicU32::new(0),
    siginfo: [const { SigInfoSlot(UnsafeCell::new(unsafe { mem::zeroed() })) }; NSIG],
};

/// One in-flight delivery: where its frame lives on the guest stack. The
/// register file is restored from the guest-visible frame itself (so a handler
/// that edits its `ucontext` takes effect); only the frame's location is kept
/// here, since the guest is not trusted to report it back.
struct SavedCtx {
    /// Guest address of the [`SigFrame`] built for this delivery.
    frame: u64,
}

/// Per-thread guest signal state over the process-shared disposition table.
pub struct Signals {
    table: SharedSigTable,
    /// This thread's pending set; the `Arc` keeps it alive as long as the
    /// catcher may write through the pointer published in `ThreadState`.
    pending: Arc<PendingSet>,
    /// Currently blocked guest mask, mirrored onto the host thread.
    blocked: u32,
    /// Alternate signal stack `(ss_sp, ss_size, ss_flags)`, if set.
    altstack: Option<(u64, u64, i32)>,
    /// Mask to restore when the next handler returns, set by `sigsuspend`.
    saved_mask: Option<u32>,
    /// In-flight deliveries, innermost last (nested signals push).
    saved_ctx: Vec<SavedCtx>,
}

impl Signals {
    pub fn new(table: SharedSigTable) -> Self {
        Self {
            table,
            pending: Arc::new(PendingSet::new()),
            blocked: 0,
            altstack: None,
            saved_mask: None,
            saved_ctx: Vec::new(),
        }
    }

    /// The address of this thread's [`PendingSet`], for publication in
    /// `ThreadState` so the catcher can record on the thread that caught.
    /// POSIX empties the pending set in the child of a fork; the dispositions,
    /// blocked mask, and alternate stack are inherited with the copy.
    pub fn reset_pending_after_fork(&self) {
        self.pending.bits.store(0, Ordering::Release);
    }

    /// posix_spawn's `SETSIGDEF`: every signal in `set` reverts to its
    /// default disposition in the spawned child, whatever it was.
    pub fn apply_spawn_sigdefault(&mut self, set: u32) {
        let mut table = self.table.lock().unwrap();
        for i in 0..NSIG {
            if set & (1 << i) != 0 {
                table[i] = GuestSigaction::default();
                install_host(i + 1, SIG_DFL);
            }
        }
    }

    /// posix_spawn's `SETSIGMASK`: the spawned child starts with exactly this
    /// blocked mask.
    pub fn apply_spawn_sigmask(&mut self, set: u32) {
        self.set_blocked(set);
    }

    /// Reset signal state across `execve` (POSIX): caught handlers revert to
    /// their default, ignored ones stay ignored, the blocked mask and the
    /// pending set are preserved, and the alternate stack is dropped.
    pub fn on_execve(&mut self) {
        let mut table = self.table.lock().unwrap();
        for i in 0..NSIG {
            let h = table[i].handler;
            if h != SIG_DFL && h != SIG_IGN {
                table[i] = GuestSigaction::default();
                install_host(i + 1, SIG_DFL);
            }
        }
        drop(table);
        self.altstack = None;
    }

    pub fn pending_set_ptr(&self) -> *const PendingSet {
        Arc::as_ptr(&self.pending)
    }

    /// Set the guest's blocked mask, stripping the unblockable signals, and
    /// mirror it onto the host thread.
    fn set_blocked(&mut self, mask: u32) {
        let kill_stop = (1u32 << (libc::SIGKILL - 1)) | (1u32 << (libc::SIGSTOP - 1));
        self.blocked = mask & !kill_stop;
        mirror_host_blocked(self.blocked);
    }

    /// Service guest `sigaction` (raw Darwin ABI: the new action is a
    /// `struct __sigaction` carrying `sa_tramp`, the old action is written
    /// back without it).
    pub fn sigaction(&mut self, signo: u64, act: u64, oldact: u64) -> SyscallResult {
        if signo == 0
            || signo as usize > NSIG
            || signo == libc::SIGKILL as u64
            || signo == libc::SIGSTOP as u64
        {
            return SyscallResult::Error(libc::EINVAL);
        }
        let idx = signo as usize - 1;
        let mut table = self.table.lock().unwrap();
        let prev = table[idx];
        if oldact != 0 {
            let old = OldSigaction {
                handler: prev.handler,
                mask: prev.mask,
                flags: prev.flags,
            };
            unsafe { (oldact as *mut OldSigaction).write_unaligned(old) };
        }
        if act != 0 {
            let a = unsafe { (act as *const KernelSigaction).read_unaligned() };
            table[idx] = GuestSigaction {
                handler: a.handler,
                mask: a.mask,
                flags: a.flags,
            };
            install_host(idx + 1, a.handler);
        }
        SyscallResult::Ok(0)
    }

    /// Service guest `sigprocmask` (pointer-shaped `set`/`oldset`, 32-bit sets).
    pub fn sigprocmask(&mut self, how: i32, set: u64, oldset: u64) -> SyscallResult {
        if oldset != 0 {
            unsafe { (oldset as *mut u32).write_unaligned(self.blocked) };
        }
        if set != 0 {
            let s = unsafe { (set as *const u32).read_unaligned() };
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

    /// Service guest `sigpending`: the union of the recorded pending sets and
    /// the host kernel's (a signal that arrived while blocked is held pending
    /// by the kernel, since the guest's mask is mirrored).
    pub fn sigpending(&self, set: u64) -> SyscallResult {
        if set != 0 {
            let pending =
                self.pending.snapshot() | STRAY_PENDING.snapshot() | host_pending_snapshot();
            unsafe { (set as *mut u32).write_unaligned(pending) };
        }
        SyscallResult::Ok(0)
    }

    /// Service guest `sigaltstack`.
    pub fn sigaltstack(&mut self, nss: u64, oss: u64) -> SyscallResult {
        if oss != 0 {
            let (sp, size, flags) = self.altstack.unwrap_or((0, 0, libc::SS_DISABLE));
            let out = StackT {
                ss_sp: sp,
                ss_size: size,
                ss_flags: flags,
            };
            unsafe { (oss as *mut StackT).write_unaligned(out) };
        }
        if nss != 0 {
            let ss = unsafe { (nss as *const StackT).read_unaligned() };
            if ss.ss_flags & libc::SS_DISABLE != 0 {
                self.altstack = None;
            } else {
                if (ss.ss_size as usize) < libc::MINSIGSTKSZ {
                    return SyscallResult::Error(libc::ENOMEM);
                }
                self.altstack = Some((ss.ss_sp, ss.ss_size, ss.ss_flags));
            }
        }
        SyscallResult::Ok(0)
    }

    /// Service guest `sigsuspend` (Darwin passes the mask by value): install
    /// the temporary mask, block until a deliverable signal is recorded, and
    /// return `EINTR`. The pre-suspend mask is stashed so the waking handler's
    /// frame carries it and its return restores it. `process` supplies the
    /// group-stop flags: a sibling's `exit_group` or committed `execve` must
    /// pull this thread out of the wait — its interrupt records no guest
    /// signal, so the wait would otherwise swallow the wake and re-park.
    pub fn sigsuspend(&mut self, mask: u64, process: &Process) -> SyscallResult {
        self.saved_mask = Some(self.blocked);
        self.set_blocked(mask as u32);
        wait_for_signal(&self.pending, self.blocked, process);
        SyscallResult::Error(libc::EINTR)
    }

    /// Whether any signal is pending and unblocked right now — the condition
    /// [`Signals::take_deliverable`] would satisfy, without consuming it. The
    /// run loop mirrors this into the safepoint flag so a fully linked guest
    /// loop is forced back to a delivery boundary.
    /// Whether a *synchronous fault* is pending and unblocked on this
    /// thread. A guest call that faults cannot make progress — the faulting
    /// instruction re-executes on every resume — so the call is abandoned
    /// and the fault delivered by the outer run loop, at a block boundary
    /// where the guest's stack is its own. Only this thread's own set is
    /// consulted: a stray process-directed signal is not this call's
    /// problem.
    pub fn has_pending_fault(&self) -> bool {
        const FAULTS: u32 =
            (1 << (SIGSEGV as u32 - 1)) | (1 << (SIGBUS as u32 - 1)) | (1 << (libc::SIGTRAP - 1));
        self.pending.snapshot() & FAULTS & !self.blocked != 0
    }

    pub fn has_deliverable(&self) -> bool {
        (self.pending.snapshot() | STRAY_PENDING.snapshot()) & !self.blocked != 0
    }

    /// Remove and return one deliverable (pending and unblocked) signal, from
    /// this thread's set first, then the stray set.
    /// Best-effort name for a guest pc: dladdr knows the shared cache and
    /// every natively loaded image; a bare address is guest-image code.
    fn name_of(pc: u64) -> String {
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        if unsafe { libc::dladdr(pc as *const libc::c_void, &mut info) } != 0
            && !info.dli_sname.is_null()
        {
            let name = unsafe { std::ffi::CStr::from_ptr(info.dli_sname) };
            return format!(
                "{}+{:#x}",
                name.to_string_lossy(),
                pc.wrapping_sub(info.dli_saddr as u64)
            );
        }
        "?".into()
    }

    pub fn take_deliverable(&self) -> Option<(u32, libc::siginfo_t)> {
        let allowed = !self.blocked;
        self.pending
            .take(allowed)
            .or_else(|| STRAY_PENDING.take(allowed))
    }

    /// Deliver `signo` into the guest: build a Darwin signal frame on the
    /// guest stack (or the registered alternate stack) from the interrupted
    /// `state`, then point the guest at its handler with the link register set
    /// to the sigreturn sentinel. Returns `false` if the disposition turned
    /// out not to be a live handler (discarded, or default-acted).
    ///
    /// `restart` carries `(resume pc after the interrupted svc, syscall
    /// number, original x0)` when the signal interrupted a restartable
    /// forwarded syscall that returned `EINTR`. If the handler has
    /// `SA_RESTART`, the saved context is rewound so the handler returns into
    /// a re-execution of the `svc` rather than seeing `EINTR` — the pc backs
    /// up 4 bytes, and `x0` (clobbered by the errno writeback; on arm64 it is
    /// both first argument and result register) and `x16` are restored.
    pub fn deliver(
        &mut self,
        state: &mut ThreadState,
        signo: u32,
        info: &libc::siginfo_t,
        restart: Option<(u64, u64, u64)>,
    ) -> bool {
        let act = self.table.lock().unwrap()[signo as usize - 1];
        // The disposition may have changed since the catch. SIG_IGN discards;
        // SIG_DFL means carrying out the kernel's default action.
        if act.handler == SIG_IGN {
            return false;
        }
        if act.handler == SIG_DFL {
            if !default_action_discards(signo) {
                // The guest dies by this signal with no handler of its own —
                // an abort(), typically. Say where it was: this runs at the
                // dispatch boundary, not in a signal handler, so allocation
                // and dladdr are fine, and a silent SIGABRT death is
                // otherwise undiagnosable (the fault reporter only sees
                // synchronous faults).
                eprintln!(
                    "chimera: guest dies by signal {signo} at pc {:#x} ({}) lr {:#x} ({})",
                    state.pc,
                    Self::name_of(state.pc),
                    state.regs[30],
                    Self::name_of(state.regs[30]),
                );
                eprintln!(
                    "  x0={:#x} x1={:#x} x2={:#x} x3={:#x} x8={:#x} sp={:#x}",
                    state.regs[0],
                    state.regs[1],
                    state.regs[2],
                    state.regs[3],
                    state.regs[8],
                    state.sp,
                );
                super::fault::die(signo as i32);
            }
            return false;
        }

        let on_alt = act.flags & libc::SA_ONSTACK != 0 && self.altstack.is_some();
        let top = if on_alt {
            let (base, size, _) = self.altstack.unwrap();
            base + size
        } else {
            state.sp
        };
        let sp = (top - mem::size_of::<SigFrame>() as u64) & !15;
        let frame = sp as *mut SigFrame;

        // The mask the frame carries — restored by the handler's return — is
        // the pre-signal blocked mask, or the pre-suspend mask if this wake
        // came out of a `sigsuspend`.
        let carried = self.saved_mask.take().unwrap_or(self.blocked);

        let mut x: [u64; 29] = state.regs[..29].try_into().unwrap();
        let resume_pc = match restart {
            Some((next_pc, nr, arg0))
                if next_pc == state.pc && act.flags & libc::SA_RESTART != 0 =>
            {
                x[0] = arg0;
                x[16] = nr;
                next_pc - 4
            }
            _ => state.pc,
        };

        unsafe {
            (*frame).info = *info;
            (*frame).mc = MContext64 {
                far: 0,
                esr: 0,
                exception: 0,
                x,
                fp: state.regs[29],
                lr: state.regs[30],
                sp: state.sp,
                pc: resume_pc,
                cpsr: state.nzcv as u32,
                pad: 0,
                ns: state.fpstate,
            };
            (*frame).uc = UContext {
                uc_onstack: on_alt as i32,
                uc_sigmask: carried,
                uc_stack: libc::stack_t {
                    ss_sp: ptr::null_mut(),
                    ss_size: 0,
                    ss_flags: 0,
                },
                uc_link: ptr::null_mut(),
                uc_mcsize: mem::size_of::<MContext64>(),
                uc_mcontext: &raw mut (*frame).mc,
            };
        }
        self.saved_ctx.push(SavedCtx {
            frame: frame as u64,
        });

        // Block the handler's declared mask, plus the signal itself unless
        // `SA_NODEFER`, for the handler's duration.
        let mut during = self.blocked | act.mask;
        if act.flags & libc::SA_NODEFER == 0 {
            during |= 1 << (signo - 1);
        }
        self.set_blocked(during);

        // Enter the handler with the C ABI the kernel-side trampoline would
        // use — `(signo, siginfo *, ucontext *)` — returning to the sentinel.
        state.regs[0] = signo as u64;
        state.regs[1] = unsafe { &raw mut (*frame).info } as u64;
        state.regs[2] = unsafe { &raw mut (*frame).uc } as u64;
        state.regs[30] = crate::arch::dispatch::SIGRETURN_SENTINEL;
        state.sp = sp;
        state.pc = act.handler;

        // SA_RESETHAND: one delivery, then back to the default disposition, on
        // the guest table and the host slot alike.
        if act.flags & libc::SA_RESETHAND != 0 {
            self.table.lock().unwrap()[signo as usize - 1].handler = SIG_DFL;
            install_host(signo as usize, SIG_DFL);
        }
        true
    }

    /// The handler returned to the sentinel: restore the interrupted context
    /// from the frame it saw (honoring any `ucontext` edits) and reinstate the
    /// mask the frame carries.
    pub fn restore(&mut self, state: &mut ThreadState) {
        let Some(saved) = self.saved_ctx.pop() else {
            // A sigreturn with no delivery in flight is a wild guest jump.
            super::fault::die(libc::SIGSEGV);
        };
        let frame = saved.frame as *const SigFrame;
        unsafe {
            let mc = &(*frame).mc;
            state.regs[..29].copy_from_slice(&mc.x);
            state.regs[29] = mc.fp;
            state.regs[30] = mc.lr;
            state.sp = mc.sp;
            state.pc = mc.pc;
            state.nzcv = mc.cpsr as u64;
            state.fpstate = mc.ns;
            self.set_blocked((*frame).uc.uc_sigmask);
        }
    }
}

/// The record-and-return host catcher installed for every signal the guest
/// handles: capture the `siginfo` into the receiving thread's pending set (or
/// the stray set if this host thread runs no guest code) and return; the
/// dispatch loop delivers at the next block boundary.
extern "C" fn chimera_sigcatch(
    signo: libc::c_int,
    info: *const libc::siginfo_t,
    _ucontext: *mut libc::c_void,
) {
    let ctx = crate::arch::dispatch::current_ctx();
    let mut pending: &PendingSet = &STRAY_PENDING;
    if !ctx.is_null() {
        let p = unsafe { (*ctx).pending_set } as *const PendingSet;
        if !p.is_null() {
            pending = unsafe { &*p };
        }
    }
    pending.record(signo as u32, info);
    // Arm this thread's safepoint so a fully linked guest loop — which never
    // returns to the run loop on its own — exits at its next loop-closing
    // poll and delivers. The interrupted thread may be executing translated
    // code right now, so the run loop cannot do this itself. A plain relaxed
    // store on an `AtomicU32`: no allocation and no locking, keeping the
    // catcher async-signal-safe. The run loop re-derives the flag at every
    // boundary (`refresh_exit_requested`), so a signal that turns out not to
    // be deliverable only costs one extra trip out of the cache.
    if !ctx.is_null() {
        unsafe { (*ctx).exit_requested.store(1, Ordering::Relaxed) };
    }
}

/// Install the host disposition for `signo`: the real `SIG_DFL`/`SIG_IGN`
/// (the kernel's own default and ignore actions are faithful and involve no
/// guest code), or the catcher for a custom guest handler. No `SA_RESTART`,
/// so a forwarded blocking syscall is interrupted and the dispatch loop
/// regains control to deliver.
fn install_host(signo: usize, handler: u64) {
    // SIGSEGV/SIGBUS belong to the fault handler; the reserved interrupt signal
    // (see crate::sys::thread) belongs to the process-wide-stop primitive. The
    // guest's disposition for any of them is recorded in the table but never
    // installed on the host slot, so the guest never receives them natively.
    if signo == SIGSEGV || signo == SIGBUS || signo as i32 == crate::sys::thread::reserved_signal()
    {
        return;
    }
    let (host, flags) = match handler {
        SIG_DFL => (libc::SIG_DFL, 0),
        SIG_IGN => (libc::SIG_IGN, 0),
        _ => (
            chimera_sigcatch as *const () as usize,
            libc::SA_SIGINFO | libc::SA_ONSTACK,
        ),
    };
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = host;
        sa.sa_flags = flags;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(signo as i32, &sa, ptr::null_mut());
    }
}

/// Whether `signo`'s default action is to discard it: SIGURG, SIGCONT (to a
/// running process), SIGCHLD, SIGWINCH, SIGINFO. Stop-default signals never
/// reach [`Signals::deliver`] — their host disposition stays `SIG_DFL`, so the
/// kernel stops the process natively.
fn default_action_discards(signo: u32) -> bool {
    const SIGINFO: i32 = 29;
    matches!(
        signo as i32,
        libc::SIGURG | libc::SIGCONT | libc::SIGCHLD | libc::SIGWINCH | SIGINFO
    )
}

/// Mirror the guest's blocked mask onto the host thread. `SIGSEGV`/`SIGBUS`
/// stay unblocked — the synchronous fault handler must always be able to run.
fn mirror_host_blocked(mask: u32) {
    let reserved = crate::sys::thread::reserved_signal();
    unsafe {
        let mut set: libc::sigset_t = mem::zeroed();
        libc::sigemptyset(&mut set);
        for i in 0..NSIG {
            let signo = (i + 1) as i32;
            // Never mask the fault or interrupt signals: the fault handler and
            // the process-wide-stop primitive must always be able to run.
            if mask & (1 << i) != 0
                && signo != SIGSEGV as i32
                && signo != SIGBUS as i32
                && signo != reserved
            {
                libc::sigaddset(&mut set, signo);
            }
        }
        libc::pthread_sigmask(libc::SIG_SETMASK, &set, ptr::null_mut());
    }
}

/// Snapshot the host kernel's pending set for this thread as a mask.
fn host_pending_snapshot() -> u32 {
    unsafe {
        let mut s: libc::sigset_t = mem::zeroed();
        libc::sigemptyset(&mut s);
        libc::sigpending(&mut s);
        let mut mask = 0u32;
        for i in 0..NSIG {
            if libc::sigismember(&s, (i + 1) as i32) == 1 {
                mask |= 1 << i;
            }
        }
        mask
    }
}

/// Block the calling host thread until a signal deliverable under `blocked`
/// has been recorded, or the process group starts dissolving (`exit_group`,
/// a committed `execve`). Race-free: all signals are blocked first, the
/// pending sets and stop flags are checked, then `sigsuspend` atomically
/// unblocks the watched ones and waits. The reserved interrupt signal is
/// never in `wait`, so a group stop's `pthread_kill` always interrupts the
/// suspend — a stop published after the flag check is not lost.
fn wait_for_signal(pending: &PendingSet, blocked: u32, process: &Process) {
    unsafe {
        let mut all: libc::sigset_t = mem::zeroed();
        libc::sigfillset(&mut all);
        let mut prev: libc::sigset_t = mem::zeroed();
        libc::pthread_sigmask(libc::SIG_BLOCK, &all, &mut prev);
        let mut wait: libc::sigset_t = mem::zeroed();
        libc::sigemptyset(&mut wait);
        let reserved = crate::sys::thread::reserved_signal();
        for i in 0..NSIG {
            // The reserved interrupt signal stays out of the suspend mask even
            // when the guest's mask names it (`SIGURG` is an ordinary maskable
            // signal on Darwin): the slot is runtime-owned — the guest never
            // observes it — and masking it here would let a guest that blocks
            // everything suppress the group-stop wake entirely.
            if blocked & (1 << i) != 0 && (i + 1) as i32 != reserved {
                libc::sigaddset(&mut wait, (i + 1) as i32);
            }
        }
        while (pending.snapshot() | STRAY_PENDING.snapshot()) & !blocked == 0 {
            if process.is_exiting() || process.exec_pending() {
                break;
            }
            libc::sigsuspend(&wait);
        }
        libc::pthread_sigmask(libc::SIG_SETMASK, &prev, ptr::null_mut());
    }
}

// The Darwin arm64 signal-frame ABI, from `mach/arm/_structs.h`. The libc
// crate does not define these for `aarch64-apple-darwin`, so they are mirrored
// here; the layout is kernel ABI and stable. [`super::fault`] reads the
// kernel-built equivalents out of a fault's `ucontext`.

/// `stack_t` as the raw `sigaltstack` syscall lays it out.
#[repr(C)]
#[derive(Clone, Copy)]
struct StackT {
    ss_sp: u64,
    ss_size: u64,
    ss_flags: i32,
}

#[repr(C)]
pub struct UContext {
    pub uc_onstack: libc::c_int,
    pub uc_sigmask: u32,
    pub uc_stack: libc::stack_t,
    pub uc_link: *mut UContext,
    pub uc_mcsize: usize,
    pub uc_mcontext: *mut MContext64,
}

/// `_STRUCT_MCONTEXT64`: exception state, thread state, then FPSIMD state.
/// The `ns` byte layout (32 Q-registers, then `FPSR`/`FPCR`) matches
/// `ThreadState::fpstate` exactly.
#[repr(C)]
pub struct MContext64 {
    /// `__darwin_arm_exception_state64`: fault address, syndrome, class.
    pub far: u64,
    pub esr: u32,
    pub exception: u32,
    /// `__darwin_arm_thread_state64`.
    pub x: [u64; 29],
    pub fp: u64,
    pub lr: u64,
    pub sp: u64,
    pub pc: u64,
    pub cpsr: u32,
    pub pad: u32,
    /// `__darwin_arm_neon_state64`.
    pub ns: [u8; 520],
}

/// The synthesized guest signal frame, pushed on the guest (or alternate)
/// stack: the `siginfo` and `ucontext` the handler receives, and the machine
/// context the ucontext points into.
#[repr(C)]
struct SigFrame {
    info: libc::siginfo_t,
    uc: UContext,
    mc: MContext64,
}
