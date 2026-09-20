//! The guest's signal state, mirrored rather than delegated.
//!
//! The host's own signal state cannot be the guest's, for two reasons. The
//! mask must never really block the signals Chimera runs on — `SIGSYS` above
//! all, which *is* the dispatch trap — so what the kernel enforces is always
//! the guest's mask minus [`UNBLOCKABLE`], and the guest's own view has to be
//! kept here to be reported back. And a signal that arrives while Chimera is
//! midway through servicing a syscall cannot be delivered where it lands: the
//! interrupted context is the runtime's, not the guest's. Those are recorded
//! per thread and delivered at the next safepoint, by building a
//! kernel-shaped `rt_sigframe` on the guest's own stack (see [`deliver`]).
//! Deferring is also what leaves a forwarded blocking syscall interruptible:
//! Chimera's handler carries no `SA_RESTART`, so the kernel hands the
//! interruption back as `EINTR` for [`restart_wanted`] to rule on.
//!
//! Dispositions live in [`Process::actions`]; everything else here belongs
//! to the thread.

use std::{
    cell::{Cell, UnsafeCell},
    mem, ptr,
};

use crate::{
    SyscallResult, SystemCall,
    sys::mmap::{copy_from_guest, copy_to_guest},
};

use super::{
    super::{
        signal::{KernelSigaction, SS_DISABLE, SS_ONSTACK},
        syscall::host_syscall,
    },
    SigsysInfo,
    thread::{Thread, chimera_altstack, current_fs, set_fs, this_thread},
};

/// Signal numbers run 1..=64, so a table indexed by number needs 65 entries.
pub const NSIG: usize = 65;

/// A kernel `siginfo_t` is 128 bytes. Chimera copies them around opaquely —
/// it forwards the kernel's bytes to the guest rather than interpreting the
/// union — so the raw array is the honest type.
const SIGINFO_SIZE: usize = 128;
pub type RawSiginfo = [u8; SIGINFO_SIZE];

pub const fn sig_bit(signo: i32) -> u64 {
    1u64 << (signo as u64 - 1)
}

/// The signals Chimera never lets the kernel block, whatever the guest asks
/// for. `SIGSYS` is the dispatch trap itself: blocked, the next guest syscall
/// takes the signal's default action and kills the process instead of
/// trapping. `SIGSEGV` and `SIGBUS` are synchronous faults that the runtime
/// takes on the guest's behalf whenever a guarded copy reads an unmapped
/// guest address, and a blocked synchronous fault is fatal too. The guest's
/// own view of its mask is kept in [`Signals::mask`] and reports these as the
/// guest set them, so the substitution is invisible.
pub const UNBLOCKABLE: u64 = sig_bit(libc::SIGSYS) | sig_bit(libc::SIGSEGV) | sig_bit(libc::SIGBUS);

/// The signal frame Chimera builds on the guest's stack when it delivers a
/// signal itself, laid out exactly as the kernel's `rt_sigframe` so the
/// kernel's own `rt_sigreturn` can restore it (see [`deliver`]).
#[repr(C)]
struct RtSigFrame {
    /// The return address the handler pops: Chimera's restorer, which sits
    /// in the exempt range and can therefore reach `rt_sigreturn`.
    pretcode: u64,
    uc: libc::ucontext_t,
    info: RawSiginfo,
}

/// `FP_XSTATE_MAGIC1`, and the offset of `_fpx_sw_bytes` within the legacy
/// `fxsave` area: how the kernel records the size of the extended FP state it
/// appended to a signal frame. Chimera copies that state verbatim into the
/// frame it builds, so it has to know how long it is.
const FP_XSTATE_MAGIC1: u32 = 0x4650_5853;
const FP_SW_BYTES_OFFSET: usize = 464;
const FXSAVE_SIZE: usize = 512;

// The signal-return trampoline handed to the kernel for every guest
// `rt_sigaction`: two instructions in Chimera's text, and therefore inside
// the exempt range — the guest's own restorer sits below the exempt floor,
// where its `rt_sigreturn` would itself trap.
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

/// One guest signal disposition, in the kernel's `rt_sigaction` shape minus
/// the restorer, which Chimera substitutes and never reports back.
#[derive(Clone, Copy)]
pub struct GuestAction {
    handler: u64,
    flags: u64,
    mask: u64,
}

impl GuestAction {
    fn is_caught(&self) -> bool {
        self.handler != libc::SIG_DFL as u64 && self.handler != libc::SIG_IGN as u64
    }
}

impl Default for GuestAction {
    fn default() -> Self {
        Self {
            handler: libc::SIG_DFL as u64,
            flags: 0,
            mask: 0,
        }
    }
}

/// One signal disposition. A `Cell` of a `Copy` value rather than a
/// `RefCell`: [`on_guest_signal`] reads the table from a handler that can
/// interrupt `rt_sigaction` mid-update, and a borrow held across that
/// interruption would panic.
pub struct ActionSlot(Cell<GuestAction>);

impl ActionSlot {
    pub fn new() -> Self {
        Self(Cell::new(GuestAction::default()))
    }

    fn load(&self) -> GuestAction {
        self.0.get()
    }

    fn store(&self, action: GuestAction) {
        self.0.set(action);
    }
}

/// One thread's signal state.
///
/// Every field is a `Cell` rather than the whole struct a `RefCell`: the
/// signal handlers that touch this state interrupt each other by nature, and
/// a `RefCell` borrow held across a `host_syscall` would panic the moment one
/// did. Scalar `Cell`s have no borrow to outlive the interruption.
pub struct Signals {
    /// The mask the guest believes is installed. What the kernel enforces is
    /// this minus [`UNBLOCKABLE`].
    pub mask: Cell<u64>,
    /// Signals caught while Chimera was inside the runtime, awaiting the next
    /// safepoint. Distinct from the kernel's pending set, which holds the
    /// signals the *mask* is keeping undelivered; `rt_sigpending` reports the
    /// union.
    pub pending: PendingQueue,
    /// The guest's alternate signal stack, virtualized: the host's belongs to
    /// Chimera's own handlers, and letting a guest `sigaltstack` through
    /// would move the runtime's trap handler onto guest memory.
    alt: Cell<libc::stack_t>,
    /// Whether the guest was on its alternate stack when control last left
    /// it, refreshed at every trap entry. Derived rather than remembered:
    /// nothing tells Chimera when a handler returns — the guest's
    /// `rt_sigreturn` goes straight to the kernel through the restorer — so a
    /// flag set at delivery would never be cleared, and the guest's alternate
    /// stack would read as occupied forever after its first use.
    on_alt: Cell<bool>,
    /// Whether control is inside the runtime — the window in which an
    /// arriving signal must be deferred rather than delivered.
    pub in_runtime: Cell<bool>,
}

impl Signals {
    pub fn new() -> Self {
        Self {
            mask: Cell::new(0),
            pending: PendingQueue::new(),
            alt: Cell::new(libc::stack_t {
                ss_sp: ptr::null_mut(),
                ss_flags: SS_DISABLE,
                ss_size: 0,
            }),
            on_alt: Cell::new(false),
            in_runtime: Cell::new(false),
        }
    }

    /// Re-derive the guest's mask from the context the kernel handed over,
    /// and whether it is on its alternate stack from the stack pointer.
    ///
    /// Chimera loses control at the end of a guest signal handler: the
    /// restorer issues `rt_sigreturn`, the kernel restores the mask from the
    /// frame, and no code of Chimera's runs in between. A mirrored mask
    /// maintained only by `rt_sigprocmask` would therefore stay stuck at the
    /// handler's mask forever after the first delivery, and every later
    /// signal would be filtered out as blocked and never delivered at all.
    ///
    /// So the kernel is the authority, and the mirror only carries what the
    /// kernel cannot: the guest's intent for the [`UNBLOCKABLE`] signals,
    /// which are never really blocked and so never appear in a context's
    /// mask. Every entry into Chimera refreshes the rest from the interrupted
    /// context, whose `uc_sigmask` is exactly the mask that was in force.
    pub fn refresh_from(&self, uc: &libc::ucontext_t) {
        self.mask
            .set(sigmask_of(&uc.uc_sigmask) | (self.mask.get() & UNBLOCKABLE));
        self.on_alt
            .set(self.on_sig_stack(uc.uc_mcontext.gregs[libc::REG_RSP as usize] as u64));
    }

    /// Whether `rsp` lies within the guest's alternate signal stack — the
    /// kernel's `on_sig_stack`, and the same answer `sigaltstack` reports as
    /// `SS_ONSTACK`.
    fn on_sig_stack(&self, rsp: u64) -> bool {
        let alt = self.alt.get();
        if alt.ss_flags & SS_DISABLE != 0 {
            return false;
        }
        let base = alt.ss_sp as u64;
        (base..base + alt.ss_size as u64).contains(&rsp)
    }
}

/// How many deferred signals Chimera will hold. The kernel's own limit is
/// `RLIMIT_SIGPENDING`, in the thousands; this queue only ever holds what
/// arrived inside a single syscall's service window, so a short one is ample
/// and a full queue degrades the way the kernel's does — the signal is
/// dropped, which for a standard signal is indistinguishable from coalescing.
const PENDING_MAX: usize = 64;

/// The signals Chimera caught inside the runtime and has not yet delivered.
///
/// A bitmask would do for standard signals, which coalesce, but not for
/// real-time ones: those queue, each instance carrying its own `si_value`,
/// and are delivered lowest-numbered first with instances of one number in
/// the order they were sent. The queue keeps them in arrival order and
/// [`PendingQueue::take_last`] imposes the rest.
pub struct PendingQueue {
    entries: UnsafeCell<[(i32, RawSiginfo); PENDING_MAX]>,
    len: Cell<usize>,
    /// The set of signal numbers held, for `rt_sigpending` to report and for
    /// the deliverable test, which would otherwise walk the queue.
    mask: Cell<u64>,
}

impl PendingQueue {
    fn new() -> Self {
        Self {
            entries: UnsafeCell::new([(0, [0; SIGINFO_SIZE]); PENDING_MAX]),
            len: Cell::new(0),
            mask: Cell::new(0),
        }
    }

    fn mask(&self) -> u64 {
        self.mask.get()
    }

    pub fn clear(&self) {
        self.len.set(0);
        self.mask.set(0);
    }

    /// Record an arrival. A standard signal already held is dropped — they do
    /// not queue — while a real-time one is appended.
    fn push(&self, signo: i32, info: &RawSiginfo) {
        if signo < libc::SIGRTMIN() && self.mask.get() & sig_bit(signo) != 0 {
            return;
        }
        let len = self.len.get();
        if len == PENDING_MAX {
            return;
        }
        unsafe { (*self.entries.get())[len] = (signo, *info) };
        self.len.set(len + 1);
        self.mask.set(self.mask.get() | sig_bit(signo));
    }

    /// Remove and return the entry that must be *built* first, which is the
    /// one that must *run* last: the highest-numbered deliverable signal, and
    /// among instances of that number the one queued latest. Delivery stacks
    /// frames, so the last frame built is the first the guest enters — which
    /// makes this reversal what produces ascending, first-sent-first order.
    fn take_last(&self, deliverable: u64) -> Option<(i32, RawSiginfo)> {
        let entries = unsafe { &mut *self.entries.get() };
        let len = self.len.get();
        let mut best: Option<usize> = None;
        for i in 0..len {
            if deliverable & sig_bit(entries[i].0) == 0 {
                continue;
            }
            match best {
                Some(b) if entries[b].0 >= entries[i].0 => {}
                _ => best = Some(i),
            }
        }
        let idx = best?;
        let taken = entries[idx];
        entries.copy_within(idx + 1..len, idx);
        self.len.set(len - 1);
        let mut mask = 0;
        for e in entries.iter().take(len - 1) {
            mask |= sig_bit(e.0);
        }
        self.mask.set(mask);
        Some(taken)
    }
}

/// The mask the kernel actually enforces for a guest that asked for `mask`.
fn host_mask(mask: u64) -> u64 {
    mask & !UNBLOCKABLE
}

/// Install the kernel-enforced mask for the guest's current one. Called
/// whenever [`Signals::mask`] changes, so an unblocked signal the kernel has
/// been holding is delivered right away rather than at the next safepoint.
pub fn sync_host_mask(t: &Thread) {
    let set = host_mask(t.sig.mask.get());
    host_syscall(&SystemCall::new(
        libc::SYS_rt_sigprocmask as u64,
        [
            libc::SIG_SETMASK as u64,
            &set as *const u64 as u64,
            0,
            8,
            0,
            0,
        ],
    ));
}

/// The kernel-enforced mask the trap's `sigreturn` must restore: `sigreturn`
/// takes the mask from the context, so the guest's own — filtered — mask has
/// to be published there rather than left as the one the trap entered with.
pub fn publish_host_mask(t: &Thread, uc: &mut libc::ucontext_t) {
    uc.uc_sigmask = sigset_from(host_mask(t.sig.mask.get()));
}

fn read_guest_sigset(ptr: u64) -> Option<u64> {
    let mut raw = [0u8; 8];
    copy_from_guest(ptr, &mut raw).then(|| u64::from_ne_bytes(raw))
}

/// `rt_sigprocmask`, serviced against the mirrored mask: the guest's own view
/// is composed and reported here, and only the filtered result reaches the
/// kernel.
pub fn do_sigprocmask(t: &Thread, call: &mut SystemCall) {
    if call.args[3] != 8 {
        call.set_result(SyscallResult::Error(libc::EINVAL));
        return;
    }
    let old = t.sig.mask.get();
    if call.args[1] != 0 {
        let Some(set) = read_guest_sigset(call.args[1]) else {
            call.set_result(SyscallResult::Error(libc::EFAULT));
            return;
        };
        let new = match call.args[0] as i32 {
            libc::SIG_BLOCK => old | set,
            libc::SIG_UNBLOCK => old & !set,
            libc::SIG_SETMASK => set,
            _ => {
                call.set_result(SyscallResult::Error(libc::EINVAL));
                return;
            }
        };
        // SIGKILL and SIGSTOP are never blockable, by the kernel's rule
        // rather than Chimera's; it drops them silently and so does this.
        t.sig
            .mask
            .set(new & !(sig_bit(libc::SIGKILL) | sig_bit(libc::SIGSTOP)));
        sync_host_mask(t);
    }
    if call.args[2] != 0 && !copy_to_guest(call.args[2], &old.to_ne_bytes()) {
        call.set_result(SyscallResult::Error(libc::EFAULT));
        return;
    }
    call.set_result(SyscallResult::Ok(0));
}

/// `rt_sigpending` reports the union of the two pending sets: the kernel's,
/// holding what the mask keeps undelivered, and Chimera's, holding what
/// arrived while the runtime was mid-syscall and has not reached a safepoint.
pub fn do_sigpending(t: &Thread, call: &mut SystemCall) {
    if call.args[1] != 8 {
        call.set_result(SyscallResult::Error(libc::EINVAL));
        return;
    }
    let mut host: u64 = 0;
    let result = host_syscall(&SystemCall::new(
        libc::SYS_rt_sigpending as u64,
        [&mut host as *mut u64 as u64, 8, 0, 0, 0, 0],
    ));
    if let SyscallResult::Error(errno) = result {
        call.set_result(SyscallResult::Error(errno));
        return;
    }
    let set = host | t.sig.pending.mask();
    if copy_to_guest(call.args[0], &set.to_ne_bytes()) {
        call.set_result(SyscallResult::Ok(0));
    } else {
        call.set_result(SyscallResult::Error(libc::EFAULT));
    }
}

/// `sigaltstack`, virtualized. The host's alternate stack is Chimera's, where
/// its own trap handler runs; letting the guest's request through would move
/// the runtime onto guest memory that an `execve` then unmaps. The guest's
/// choice is recorded instead and honored by [`deliver`] when it places a
/// frame for an `SA_ONSTACK` handler.
pub fn do_sigaltstack(t: &Thread, call: &mut SystemCall) {
    let old = t.sig.alt.get();
    if call.args[0] != 0 {
        let mut raw = [0u8; mem::size_of::<libc::stack_t>()];
        if !copy_from_guest(call.args[0], &mut raw) {
            call.set_result(SyscallResult::Error(libc::EFAULT));
            return;
        }
        let new: libc::stack_t = unsafe { mem::transmute(raw) };
        // Changing the alt stack from a handler running on it would pull the
        // stack out from under the handler; the kernel refuses, and so does
        // this.
        if t.sig.on_alt.get() {
            call.set_result(SyscallResult::Error(libc::EPERM));
            return;
        }
        if new.ss_flags & !(SS_DISABLE | SS_ONSTACK) != 0 {
            call.set_result(SyscallResult::Error(libc::EINVAL));
            return;
        }
        if new.ss_flags & SS_DISABLE != 0 {
            t.sig.alt.set(libc::stack_t {
                ss_sp: ptr::null_mut(),
                ss_flags: SS_DISABLE,
                ss_size: 0,
            });
        } else {
            if new.ss_size < libc::MINSIGSTKSZ {
                call.set_result(SyscallResult::Error(libc::ENOMEM));
                return;
            }
            t.sig.alt.set(libc::stack_t {
                ss_sp: new.ss_sp,
                ss_flags: 0,
                ss_size: new.ss_size,
            });
        }
    }
    if call.args[1] != 0 {
        let mut reported = old;
        // The flags word is a status on the way out, not the stored value:
        // a stack the guest is currently running on reads back SS_ONSTACK.
        if t.sig.on_alt.get() && old.ss_flags & SS_DISABLE == 0 {
            reported.ss_flags = SS_ONSTACK;
        }
        let raw: [u8; mem::size_of::<libc::stack_t>()] = unsafe { mem::transmute(reported) };
        if !copy_to_guest(call.args[1], &raw) {
            call.set_result(SyscallResult::Error(libc::EFAULT));
            return;
        }
    }
    call.set_result(SyscallResult::Ok(0));
}

/// `rt_sigsuspend`: install the temporary mask, park until a signal arrives,
/// then restore. The wait is the host's, so the kernel does the parking; the
/// signal that ends it is caught by [`on_guest_signal`] and deferred, because
/// the runtime is mid-syscall.
///
/// The deferred signal is delivered here rather than at the usual safepoint,
/// because only here is the suspend mask still the one in force. POSIX runs
/// the handler under it and hands the *caller's* mask back afterwards, which
/// is the split [`deliver_pending`] takes as `base` and `restore`: without
/// it, a signal the caller had blocked — the whole point of the call — would
/// be filtered out at the safepoint and never delivered at all.
pub fn do_sigsuspend(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t) {
    if call.args[1] != 8 {
        call.set_result(SyscallResult::Error(libc::EINVAL));
        return;
    }
    let Some(set) = read_guest_sigset(call.args[0]) else {
        call.set_result(SyscallResult::Error(libc::EFAULT));
        return;
    };
    let saved = t.sig.mask.get();
    let suspend = set & !(sig_bit(libc::SIGKILL) | sig_bit(libc::SIGSTOP));
    t.sig.mask.set(suspend);
    let filtered = host_mask(suspend);
    let result = host_syscall(&SystemCall::new(
        libc::SYS_rt_sigsuspend as u64,
        [&filtered as *const u64 as u64, 8, 0, 0, 0, 0],
    ));
    // The result register belongs to the frame the delivery is about to save,
    // so it has to be in place first: `sigsuspend` always reports `EINTR`, and
    // that is what the guest resumes on once its handler returns.
    uc.uc_mcontext.gregs[libc::REG_RAX as usize] = match result {
        SyscallResult::Ok(v) => v,
        SyscallResult::Error(errno) => -(errno as i64),
    } as libc::greg_t;
    if deliver_pending(t, uc, suspend, saved) == 0 {
        t.sig.mask.set(saved);
    }
    sync_host_mask(t);
    call.set_result(result);
}

/// `rt_sigaction`, serviced against the mirrored disposition table. The guest
/// never reaches the kernel's table: Chimera installs its own handler for
/// every signal the guest catches, so what the guest reads back has to come
/// from here.
pub fn do_sigaction(t: &Thread, call: &mut SystemCall) {
    let signo = call.args[0] as i32;
    if signo < 1 || signo >= NSIG as i32 || call.args[3] != 8 {
        call.set_result(SyscallResult::Error(libc::EINVAL));
        return;
    }
    let old = t.process.action(signo).load();
    if call.args[1] != 0 {
        if signo == libc::SIGKILL || signo == libc::SIGSTOP {
            call.set_result(SyscallResult::Error(libc::EINVAL));
            return;
        }
        let mut raw = [0u8; mem::size_of::<KernelSigaction>()];
        if !copy_from_guest(call.args[1], &mut raw) {
            call.set_result(SyscallResult::Error(libc::EFAULT));
            return;
        }
        let act: KernelSigaction = unsafe { mem::transmute(raw) };
        t.process.action(signo).store(GuestAction {
            handler: act.handler,
            flags: act.flags,
            mask: act.mask,
        });
        install_host_action(t, signo);
    }
    if call.args[2] != 0 {
        let reported = KernelSigaction {
            handler: old.handler,
            flags: old.flags,
            mask: old.mask,
            restorer: 0,
        };
        let raw: [u8; mem::size_of::<KernelSigaction>()] = unsafe { mem::transmute(reported) };
        if !copy_to_guest(call.args[2], &raw) {
            call.set_result(SyscallResult::Error(libc::EFAULT));
            return;
        }
    }
    call.set_result(SyscallResult::Ok(0));
}

/// POSIX `execve` resets caught signals to their default disposition and
/// leaves ignored ones ignored; `clone3`'s `CLONE_CLEAR_SIGHAND` asks for the
/// same in the child. The mask and the pending set survive both, so only the
/// disposition table is swept.
pub fn reset_guest_signals(t: &Thread) {
    for signo in 1..NSIG as i32 {
        if t.process.action(signo).load().is_caught() {
            t.process.action(signo).store(GuestAction::default());
            install_host_action(t, signo);
        }
    }
}

/// Install the host disposition matching the guest's recorded one. A caught
/// signal gets Chimera's [`on_guest_signal`], which decides whether the guest
/// can take it here or must take it at the next safepoint; `SIG_DFL` and
/// `SIG_IGN` are installed as themselves, so the kernel keeps doing the
/// default action or dropping the signal without a trip through userspace.
///
/// `SIGSYS` is never installed: it is the dispatch trap, and the guest's
/// disposition for it is recorded but never honored.
fn install_host_action(t: &Thread, signo: i32) {
    if signo == libc::SIGSYS {
        return;
    }
    let action = t.process.action(signo).load();
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        if action.is_caught() {
            sa.sa_sigaction = on_guest_signal as *const () as usize;
            // Chimera's handler is not re-entered: it either delivers or
            // defers, both of which touch the one signal state.
            libc::sigfillset(&mut sa.sa_mask);
            libc::sigdelset(&mut sa.sa_mask, libc::SIGSEGV);
            libc::sigdelset(&mut sa.sa_mask, libc::SIGBUS);
            // Deliberately no `SA_RESTART`: the guest's own flag decides
            // whether an interrupted syscall restarts, and Chimera can only
            // apply it if the kernel hands the interruption back (see
            // `restart_syscall`).
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        } else {
            sa.sa_sigaction = action.handler as usize;
            libc::sigemptyset(&mut sa.sa_mask);
            sa.sa_flags = 0;
        }
        libc::sigaction(signo, &sa, ptr::null_mut());
    }
}

/// A guest signal arrived.
///
/// Where it can be taken depends on what it interrupted. Guest code can take
/// it immediately, and does. Runtime code cannot: the interrupted context is
/// Chimera's, so a frame built on it would return the guest into the middle
/// of a syscall it never made, and the handler would run against the
/// runtime's TLS. Those are recorded and taken at the next safepoint, which
/// is the tail of the trap handler — by which point the syscall being
/// serviced has a result and the context describes the guest again.
extern "C" fn on_guest_signal(
    signo: libc::c_int,
    info: *mut libc::siginfo_t,
    uc: *mut libc::c_void,
) {
    let t = this_thread();
    let entry_fs = current_fs();
    set_fs(t.runtime_fs);

    let uc = unsafe { &mut *(uc as *mut libc::ucontext_t) };
    let raw_info: RawSiginfo = unsafe { ptr::read(info as *const RawSiginfo) };

    if t.sig.in_runtime.get() {
        t.sig.pending.push(signo, &raw_info);
    } else {
        // Guest code was interrupted, so this context carries the guest's own
        // mask — including any restored by a handler's `rt_sigreturn`, which
        // Chimera never sees.
        t.sig.refresh_from(uc);
        let mask = t.sig.mask.get();
        deliver(t, signo, &raw_info, uc, mask, mask);
    }

    set_fs(entry_fs);
}

/// Take every deferred signal `base` leaves unblocked, building a frame for
/// each onto `uc`. Called at the safepoint on the way out of the trap
/// handler. Returns how many were delivered.
///
/// `base` is the mask the guest is under while the signals are taken, and
/// `restore` the one the last handler to run returns to — the same value,
/// except after a `sigsuspend`, whose caller gets its original mask back
/// rather than the one it waited under.
///
/// Frames stack, so the last one built is the first the guest enters. Each
/// handler returns to the mask the *next* one to run needs, and the last
/// returns to `restore`, which is why the chain walks backwards from it.
pub fn deliver_pending(t: &Thread, uc: &mut libc::ucontext_t, base: u64, restore: u64) -> usize {
    let mut delivered = 0;
    let mut restore = restore;
    while let Some((signo, info)) = t.sig.pending.take_last(!base) {
        deliver(t, signo, &info, uc, base, restore);
        restore = t.sig.mask.get();
        delivered += 1;
    }
    delivered
}

/// Build a signal frame on the guest's stack and point `uc` at the guest's
/// handler, so the `sigreturn` that ends the interruption enters the handler
/// rather than resuming what it interrupted.
///
/// The frame is laid out as the kernel's `rt_sigframe`, because it is the
/// kernel's own `rt_sigreturn` that will consume it: Chimera substitutes its
/// restorer for the guest's — the guest's sits below the exempt floor, where
/// its `rt_sigreturn` would trap into a dispatch handler with no way to
/// complete it — and that restorer issues the real syscall. The extended FP
/// state is copied verbatim out of the frame the kernel built for Chimera and
/// the pointer to it relocated, which is the only part whose size is not
/// known up front.
fn deliver(
    t: &Thread,
    signo: i32,
    info: &RawSiginfo,
    uc: &mut libc::ucontext_t,
    base: u64,
    restore: u64,
) {
    let action = t.process.action(signo).load();
    if !action.is_caught() {
        // The disposition changed out from under a deferred signal (an
        // `SA_RESETHAND` delivery, or the guest's own `sigaction`). Ignoring
        // is right for SIG_IGN; for SIG_DFL, re-raising lets the kernel apply
        // the default action against the guest rather than emulating it here.
        if action.handler == libc::SIG_DFL as u64 {
            host_syscall(&SystemCall::new(
                libc::SYS_kill as u64,
                [unsafe { libc::getpid() } as u64, signo as u64, 0, 0, 0, 0],
            ));
        }
        return;
    }

    let saved = uc.uc_mcontext;
    let fp_size = unsafe { fpstate_size(saved.fpregs as *const u8) };

    // Place the frame where the kernel would: below the interrupted stack
    // pointer past the red zone, or at the top of the guest's alternate stack
    // when the handler asked for one and is not already running on it.
    // Whether to switch stacks is decided against the context being saved,
    // not against trap entry: a second frame built at the same safepoint sees
    // the first one's stack pointer and stacks onto it rather than starting
    // over at the top and overwriting it.
    let alt = t.sig.alt.get();
    let use_alt = action.flags & libc::SA_ONSTACK as u64 != 0
        && alt.ss_flags & SS_DISABLE == 0
        && !t
            .sig
            .on_sig_stack(saved.gregs[libc::REG_RSP as usize] as u64);
    let mut sp = if use_alt {
        alt.ss_sp as u64 + alt.ss_size as u64
    } else {
        saved.gregs[libc::REG_RSP as usize] as u64 - 128
    };
    sp = (sp - fp_size as u64) & !63;
    let fp_addr = sp;
    sp -= mem::size_of::<RtSigFrame>() as u64;
    // The handler is entered with the return address pushed, so this
    // alignment is what leaves `rsp` 16-byte aligned inside it.
    let frame_addr = (sp & !15) - 8;

    let mut frame: RtSigFrame = unsafe { mem::zeroed() };
    frame.pretcode = chimera_sud_restorer as *const () as u64;
    frame.uc.uc_flags = uc.uc_flags;
    frame.uc.uc_link = ptr::null_mut();
    // What `rt_sigreturn` restores as the alternate stack. It has to be
    // Chimera's, since the runtime's own handlers keep running on it after
    // the guest's returns; the guest's view of `sigaltstack` is answered from
    // the mirrored state instead, so this is invisible to it.
    frame.uc.uc_stack = chimera_altstack();
    frame.uc.uc_mcontext = saved;
    frame.uc.uc_mcontext.fpregs = fp_addr as *mut _;
    frame.uc.uc_sigmask = sigset_from(restore);
    frame.info = *info;

    let frame_bytes = unsafe {
        std::slice::from_raw_parts(
            &frame as *const RtSigFrame as *const u8,
            mem::size_of::<RtSigFrame>(),
        )
    };
    let fp_bytes = unsafe { std::slice::from_raw_parts(saved.fpregs as *const u8, fp_size) };
    if !copy_to_guest(frame_addr, frame_bytes) || !copy_to_guest(fp_addr, fp_bytes) {
        // The guest's stack will not take a frame — the classic stack
        // overflow with no alternate stack registered. The kernel kills the
        // process with the signal's default action; so does this.
        force_default(signo);
    }

    // The mask the handler runs under, and the one `rt_sigreturn` will
    // restore: the guest's, plus the handler's own mask, plus the signal
    // itself unless it asked to stay re-entrant.
    let mut new_mask = base | action.mask;
    if action.flags & libc::SA_NODEFER as u64 == 0 {
        new_mask |= sig_bit(signo);
    }
    t.sig.mask.set(new_mask);
    // A one-shot handler is spent: the kernel resets it before the handler
    // runs, so a second signal arriving inside it takes the default action.
    if action.flags & libc::SA_RESETHAND as u64 != 0 {
        t.process.action(signo).store(GuestAction::default());
        install_host_action(t, signo);
    }
    let gregs = &mut uc.uc_mcontext.gregs;
    gregs[libc::REG_RSP as usize] = frame_addr as libc::greg_t;
    gregs[libc::REG_RIP as usize] = action.handler as libc::greg_t;
    gregs[libc::REG_RDI as usize] = signo as libc::greg_t;
    gregs[libc::REG_RSI as usize] =
        (frame_addr + mem::offset_of!(RtSigFrame, info) as u64) as libc::greg_t;
    gregs[libc::REG_RDX as usize] =
        (frame_addr + mem::offset_of!(RtSigFrame, uc) as u64) as libc::greg_t;
    gregs[libc::REG_RAX as usize] = 0;
    // The ABI hands a handler a cleared direction flag, and single-step and
    // resume must not carry into it.
    gregs[libc::REG_EFL as usize] &= !(0x400 | 0x100 | 0x10000);
    publish_host_mask(t, uc);
}

/// Whether every deferred signal wants the interrupted syscall restarted.
/// The kernel's rule: a handler carrying `SA_RESTART` resumes the call, one
/// without it lets `EINTR` through, and a signal with no handler at all (a
/// deferred one whose disposition has since been reset) restarts.
pub fn restart_wanted(t: &Thread) -> bool {
    let deliverable = t.sig.pending.mask() & !t.sig.mask.get();
    if deliverable == 0 {
        return false;
    }
    let mut rest = deliverable;
    while rest != 0 {
        let signo = rest.trailing_zeros() as i32 + 1;
        rest &= rest - 1;
        let action = t.process.action(signo).load();
        if action.is_caught() && action.flags & libc::SA_RESTART as u64 == 0 {
            return false;
        }
    }
    true
}

/// Rewind the trapped context onto the `syscall` instruction so the guest
/// re-issues the interrupted call once its handler returns — the kernel's own
/// restart, which rewinds `rip` by the two bytes of the instruction and puts
/// the call number back in `rax`. The dispatch `siginfo` reports the address
/// *after* the instruction, which is what makes the rewind exact.
pub fn restart_syscall(uc: &mut libc::ucontext_t, info: &SigsysInfo, nr: u64) {
    const SYSCALL_INSN_LEN: u64 = 2;
    uc.uc_mcontext.gregs[libc::REG_RIP as usize] =
        (info.call_addr - SYSCALL_INSN_LEN) as libc::greg_t;
    uc.uc_mcontext.gregs[libc::REG_RAX as usize] = nr as libc::greg_t;
}

/// The size of the extended FP state the kernel appended to a signal frame.
/// The `_fpx_sw_bytes` record inside the legacy `fxsave` area carries it; an
/// absent magic means no extended state, just the 512-byte legacy area.
unsafe fn fpstate_size(fpregs: *const u8) -> usize {
    if fpregs.is_null() {
        return 0;
    }
    unsafe {
        let magic = ptr::read_unaligned(fpregs.add(FP_SW_BYTES_OFFSET) as *const u32);
        if magic != FP_XSTATE_MAGIC1 {
            return FXSAVE_SIZE;
        }
        ptr::read_unaligned(fpregs.add(FP_SW_BYTES_OFFSET + 4) as *const u32) as usize
    }
}

/// The low 64 bits of a `sigset_t` — signals 1..=64, which is all of them.
fn sigmask_of(set: &libc::sigset_t) -> u64 {
    unsafe { ptr::read(set as *const libc::sigset_t as *const u64) }
}

/// Widen a mask into the `sigset_t` shape `ucontext_t` carries. The kernel
/// uses the low 64 bits for signals 1..=64 and leaves the rest zero.
fn sigset_from(mask: u64) -> libc::sigset_t {
    let mut set: libc::sigset_t = unsafe { mem::zeroed() };
    unsafe { ptr::write(&mut set as *mut libc::sigset_t as *mut u64, mask) };
    set
}

/// Kill the guest with a signal's default action, for the case a frame cannot
/// be built. Resetting the disposition first is what makes the re-raise
/// terminal rather than another trip through the handler.
fn force_default(signo: i32) -> ! {
    unsafe {
        let mut dfl: libc::sigaction = mem::zeroed();
        dfl.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut dfl.sa_mask);
        libc::sigaction(signo, &dfl, ptr::null_mut());
        let empty: u64 = 0;
        host_syscall(&SystemCall::new(
            libc::SYS_rt_sigprocmask as u64,
            [
                libc::SIG_SETMASK as u64,
                &empty as *const u64 as u64,
                0,
                8,
                0,
                0,
            ],
        ));
        libc::raise(signo);
        libc::_exit(128 + signo);
    }
}
