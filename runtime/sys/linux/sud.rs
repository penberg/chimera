//! Native execution behind Linux syscall user dispatch (SUD).
//!
//! An alternative to the translating backend: the guest's instructions run
//! unmodified on the CPU, and interception happens at the syscall boundary
//! only. `prctl(PR_SET_SYSCALL_USER_DISPATCH)` (Linux 5.11) makes the kernel
//! deliver `SIGSYS` for any syscall instruction executed outside a single
//! exempt address range; Chimera puts its own image — runtime text, libc,
//! vdso, and with them every runtime syscall site and the signal-return
//! trampoline — inside that range and loads the guest below it, so every
//! guest syscall traps into [`on_sigsys`], which drives the same
//! [`SystemCalls`] embedder hooks as the translating dispatcher.
//!
//! The address-space contract: the exempt range is everything at or above
//! [`EXEMPT_FLOOR`]. The kernel links a PIE and its libraries above that line
//! (`ELF_ET_DYN_BASE` is `0x5555_5555_4000`), which [`execv`] verifies rather
//! than assumes. Guest images and guest `NULL`-hint mappings are placed in a
//! bump-allocated arena at [`GUEST_ARENA_BASE`], below the line, so guest
//! code — a JIT's fresh pages included — can never issue an unintercepted
//! syscall. Guest *data* the kernel places on its own (the initial stack,
//! `brk` growth) may sit above the line; the range exempts instruction
//! addresses, and data is not fetched.
//!
//! What this backend trades away, compared to translation: the guest executes
//! natively, so a *hostile* guest can branch straight to a syscall
//! instruction inside the exempt range (Chimera's own libc) and bypass
//! interception — SUD confines syscall *sites*, not control flow. The
//! translating backend has no such hole and remains the default; this one
//! suits observation and compatibility work (an strace, a VFS overlay) on
//! guests that are not adversarial, at native speed.
//!
//! Signals are Chimera's rather than the kernel's, for two reasons that
//! together shape [`Signals`]. The mask cannot be the guest's: `SIGSYS` *is*
//! the dispatch trap, and a guest that blocks it — `sigfillset` around a
//! critical section is the ordinary way — would make its own next syscall
//! fatal. So what the kernel enforces is always the guest's mask minus
//! [`UNBLOCKABLE`], and the guest's own view is mirrored here to be reported
//! back. And a signal arriving while a syscall is being serviced cannot be
//! delivered where it lands: the interrupted context is the runtime's, not
//! the guest's. Those are deferred to the safepoint at the tail of
//! [`on_sigsys`], where the context describes the guest again, and delivered
//! by building a kernel-shaped `rt_sigframe` on the guest's own stack (see
//! [`deliver`]). Deferring is also what leaves a forwarded blocking syscall
//! interruptible, since Chimera's handler carries no `SA_RESTART` and the
//! kernel hands the interruption back as `EINTR` for [`restart_wanted`] to
//! rule on.
//!
//! Every guest thread is a host thread. A `clone` in the thread shape cannot
//! be forwarded — the task the kernel made would come back from the syscall
//! inside this trap handler, with no state of its own and no interception at
//! all — so Chimera creates the host thread itself and lets it build its own
//! before any guest instruction runs on it (see [`spawn_thread`]). What the
//! threads share lives in [`Process`]; what belongs to one lives in
//! [`Thread`], reached through the `gs` base. `fork` and the `posix_spawn`
//! shape are forwarded, `execve` is emulated in place, and a group-wide stop
//! travels by signal, since a guest thread running natively has no safepoint
//! to poll.

use std::{
    cell::{Cell, UnsafeCell},
    ffi::OsString,
    io, mem,
    os::fd::AsRawFd,
    path::Path,
    ptr,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering},
    },
};

use crate::{
    Error, SyscallResult, SystemCall, SystemCalls,
    sys::mmap::{copy_from_guest, copy_to_guest},
};

use super::{
    elf::{LoadedElf, PAGE_SIZE, ParsedElf, map_elf_native, parse_elf},
    exec::{
        ExecRequest, PreparedExec, build_stack, close_cloexec_fds, exec_errno, initial_request,
        prepare_exec,
    },
    fault,
    syscall::host_syscall,
};

const PR_SET_SYSCALL_USER_DISPATCH: libc::c_int = 59;
const PR_SYS_DISPATCH_OFF: libc::c_ulong = 0;
const PR_SYS_DISPATCH_ON: libc::c_ulong = 1;

/// Everything at or above this address is exempt from dispatch: the runtime,
/// its libraries, and the vdso live here (see the module comment).
const EXEMPT_FLOOR: u64 = 0x5500_0000_0000;

/// The guest arena: where guest images and guest `NULL`-hint mappings are
/// bump-allocated, safely below [`EXEMPT_FLOOR`].
const GUEST_ARENA_BASE: u64 = 0x5100_0000_0000;
const GUEST_ARENA_CEILING: u64 = 0x5400_0000_0000;

/// Gap left after each image placed in the arena, room for `brk`-less heaps
/// and a guard against off-by-a-page neighbors.
const ARENA_IMAGE_GAP: u64 = 2 * 1024 * 1024;

const ARCH_SET_FS: u64 = 0x1002;
const ARCH_GET_FS: u64 = 0x1003;

/// `clone3`'s `CLONE_CLEAR_SIGHAND`. Defined locally: the flag lives in bit
/// 32 and the libc crate's `c_int` constant truncates to 0.
const CLONE_CLEAR_SIGHAND: u64 = 1 << 32;

/// Signal numbers run 1..=64, so a table indexed by number needs 65 entries.
const NSIG: usize = 65;

/// A kernel `siginfo_t` is 128 bytes. Chimera copies them around opaquely —
/// it forwards the kernel's bytes to the guest rather than interpreting the
/// union — so the raw array is the honest type.
const SIGINFO_SIZE: usize = 128;
type RawSiginfo = [u8; SIGINFO_SIZE];

fn sig_bit(signo: i32) -> u64 {
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
const UNBLOCKABLE: u64 =
    sig_bit_const(libc::SIGSYS) | sig_bit_const(libc::SIGSEGV) | sig_bit_const(libc::SIGBUS);

const fn sig_bit_const(signo: i32) -> u64 {
    1u64 << (signo as u64 - 1)
}

const SS_ONSTACK: i32 = 1;
const SS_DISABLE: i32 = 2;

/// The `siginfo` layout the kernel uses to describe a dispatch trap (the
/// `_sigsys` arm of its union), which the libc crate does not expose.
#[repr(C)]
struct SigsysInfo {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _pad: i32,
    call_addr: u64,
    syscall: i32,
    arch: u32,
}

/// The kernel's `rt_sigaction` argument layout.
#[repr(C)]
#[derive(Clone, Copy)]
struct KernelSigaction {
    handler: u64,
    flags: u64,
    restorer: u64,
    mask: u64,
}

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
// the exempt range — the guest's own restorer sits below `EXEMPT_FLOOR`,
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

/// One guest thread. Every guest thread is a host thread, and this is the
/// state that belongs to it alone: the two `fs` bases it switches between,
/// its signal mask and deferred signals, its alternate stack, and the frame
/// its `exit` unwinds to.
///
/// Reached from the trap handler through the `gs` base — see [`this_thread`].
/// A `fork` child inherits its copy, contexts and all, so the child unwinds
/// through its own frame exactly like the parent.
#[repr(C)]
struct Thread {
    /// A pointer to this very struct, at offset 0 so the trap handler can
    /// load it with a single `gs:[0]`. Written by [`set_this_thread`] once
    /// the struct is at its final address — it cannot be filled in during
    /// construction, where the value would be the address of a local about to
    /// move. `Cell` is `repr(transparent)`, so the field is still a bare
    /// pointer at offset 0 as far as the load is concerned.
    self_ptr: Cell<*const Thread>,
    /// The state shared with every other thread of the guest process.
    process: Arc<Process>,
    /// The runtime's `fs` base, restored on every [`on_sigsys`] entry so the
    /// handler's Rust code sees its own TLS; the guest owns the real `fs`
    /// while it runs (its TLS accesses are native). Per thread, since each
    /// host thread has TLS of its own.
    runtime_fs: u64,
    /// The guest's `fs` base, kept by the virtualized
    /// `arch_prctl(ARCH_SET_FS)` and reinstated when the handler returns.
    guest_fs: Cell<u64>,
    /// This thread's kernel TID, which is also the TID the guest sees. A
    /// `Cell` because a fork child keeps the struct and takes a new TID.
    tid: Cell<i32>,
    /// Whether this is the thread group's leader — the one whose run
    /// returning ends the process, and the one an `exit_group` from a sibling
    /// hands the status to. A fork child is promoted to leader whichever
    /// thread forked, since it is its new process's only thread.
    is_leader: Cell<bool>,
    /// The `CLONE_CHILD_CLEARTID` word to zero and wake on exit, which is
    /// what a `pthread_join` blocks on.
    clear_child_tid: Cell<Option<u64>>,
    /// The write end of the pipe a `posix_spawn` child reports its `execve`
    /// outcome on; set only in such a child. See [`spawned`].
    spawn_report_fd: Cell<Option<i32>>,
    /// The errno of this spawn child's most recent failed `execve`, reported
    /// to the blocked parent only if the child exits without ever committing
    /// one.
    spawn_exec_errno: Cell<Option<i32>>,
    /// A group stop that arrived while this thread was inside the runtime and
    /// could not be taken where it landed; honored at the next safepoint.
    /// See [`on_stop`].
    stop_requested: Cell<bool>,
    /// Set by the `exit`/`exit_group` intercept just before unwinding.
    exit: Cell<Option<i32>>,
    /// Where the unwind lands: the frame that entered the guest, captured
    /// with `getcontext`. Boxed so the `fpregs` self-pointer `getcontext`
    /// plants stays valid.
    exit_ctx: Box<UnsafeCell<libc::ucontext_t>>,
    /// The guest's per-thread signal state: mask, deferred signals, and
    /// alternate stack. Dispositions are process-wide and live in
    /// [`Process::actions`]. See [`Signals`].
    sig: Signals,
}

/// The state every thread of the guest process shares: the embedder's
/// handler, the signal dispositions POSIX keeps process-wide, the guest
/// address arena, and the bookkeeping a group-wide stop needs.
struct Process {
    /// The embedder's system-call handler. `SystemCalls` is `Send + Sync` and
    /// dispatched by `&self`, so every guest thread drives the one instance.
    handler: Box<dyn SystemCalls>,
    /// The guest's signal dispositions. Process-wide, as POSIX requires: a
    /// handler installed on one thread is the one every thread takes the
    /// signal with.
    actions: [ActionSlot; NSIG],
    /// Bump pointer into the guest arena, shared because the arena is one
    /// address space.
    bump: AtomicU64,
    /// Mappings owned by the current guest image outside the arena — `ET_EXEC`
    /// segments at their fixed low addresses and thread stacks — torn down
    /// together with the arena when an `execve` replaces the image. Only
    /// touched with the group quiesced, so a plain mutex is safe here.
    regions: Mutex<Vec<(u64, u64)>>,
    /// The live guest threads, by kernel TID, with the leader first. A
    /// group-wide stop reaches its siblings through this, and a leader that
    /// outlives its own guest waits on it.
    threads: Mutex<Vec<i32>>,
    /// Signalled whenever `threads` shrinks, so a leader parked in
    /// [`Process::wait_for_others`] wakes.
    threads_cv: Condvar,
    /// An image a non-leader thread's `execve` committed, waiting for the
    /// leader to install and run. See [`Process::publish_exec`].
    exec_request: Mutex<Option<PreparedExec>>,
    /// Latched once an exec has been committed, and cleared only when the new
    /// image is in place. The slot going empty means the leader has *taken*
    /// the image, not that another exec may start: without the latch a second
    /// racing thread would publish into the empty slot and stop the leader
    /// again, mid-install.
    exec_committed: AtomicBool,
    /// Set when any thread issues `exit_group`, with the status in
    /// `exit_code`: the whole group ends, not just the caller.
    exiting: AtomicBool,
    exit_code: AtomicI32,
    /// The guest exit status of the most recent thread to finish. Absent an
    /// `exit_group`, the kernel reports the *last* thread's status as the
    /// process's, so every thread records its own on the way out.
    last_exit_status: AtomicI32,
}

/// One signal disposition, published for lock-free reads.
///
/// The trap handler cannot take a lock to read this. A guest signal arriving
/// mid-service runs [`on_guest_signal`] on the same thread, which reads the
/// disposition to decide what to do with it; if the interrupted code held a
/// mutex over the table, that read would deadlock against itself. So the slot
/// is a seqlock: writers — `rt_sigaction`, which is rare — bump `seq` to an
/// odd value, store, and bump it to even, while a reader retries until it
/// sees one even value twice with no change across the load. That is enough
/// to make a torn read impossible without any reader ever blocking.
struct ActionSlot {
    seq: AtomicU32,
    handler: AtomicU64,
    flags: AtomicU64,
    mask: AtomicU64,
}

impl ActionSlot {
    fn new() -> Self {
        Self {
            seq: AtomicU32::new(0),
            handler: AtomicU64::new(libc::SIG_DFL as u64),
            flags: AtomicU64::new(0),
            mask: AtomicU64::new(0),
        }
    }

    fn load(&self) -> GuestAction {
        loop {
            let before = self.seq.load(Ordering::Acquire);
            if before & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let action = GuestAction {
                handler: self.handler.load(Ordering::Relaxed),
                flags: self.flags.load(Ordering::Relaxed),
                mask: self.mask.load(Ordering::Relaxed),
            };
            if self.seq.load(Ordering::Acquire) == before {
                return action;
            }
        }
    }

    fn store(&self, action: GuestAction) {
        let seq = self.seq.load(Ordering::Relaxed);
        self.seq.store(seq.wrapping_add(1), Ordering::Release);
        self.handler.store(action.handler, Ordering::Relaxed);
        self.flags.store(action.flags, Ordering::Relaxed);
        self.mask.store(action.mask, Ordering::Relaxed);
        self.seq.store(seq.wrapping_add(2), Ordering::Release);
    }
}

impl Process {
    fn new(handler: Box<dyn SystemCalls>) -> Self {
        Self {
            handler,
            actions: std::array::from_fn(|_| ActionSlot::new()),
            bump: AtomicU64::new(GUEST_ARENA_BASE),
            regions: Mutex::new(Vec::new()),
            exec_request: Mutex::new(None),
            exec_committed: AtomicBool::new(false),
            threads: Mutex::new(Vec::new()),
            threads_cv: Condvar::new(),
            exiting: AtomicBool::new(false),
            exit_code: AtomicI32::new(0),
            last_exit_status: AtomicI32::new(0),
        }
    }

    fn register(&self, tid: i32) {
        self.threads.lock().unwrap().push(tid);
    }

    fn unregister(&self, tid: i32) {
        let mut threads = self.threads.lock().unwrap();
        threads.retain(|&t| t != tid);
        self.threads_cv.notify_all();
    }

    fn is_exiting(&self) -> bool {
        self.exiting.load(Ordering::Acquire)
    }

    /// Hand a committed image to the leader, which installs and runs it (see
    /// the sibling-exec path in [`do_execve`]).
    ///
    /// First committer wins: a second concurrent exec publishes nothing and
    /// is told so. Concurrent execs race natively too, and only one survives
    /// — the loser is killed by the winner's `de_thread` and never observes a
    /// return value. Letting the second one through would be worse than
    /// losing it: it would replace an image the leader may already be
    /// installing.
    fn publish_exec(&self, prepared: PreparedExec) -> Option<PreparedExec> {
        let mut request = self.exec_request.lock().unwrap();
        if self.exec_committed.swap(true, Ordering::AcqRel) || self.is_exiting() {
            return Some(prepared);
        }
        *request = Some(prepared);
        None
    }

    fn take_exec_request(&self) -> Option<PreparedExec> {
        self.exec_request.lock().unwrap().take()
    }

    /// Reopen the group to execs once the new image is running: its own
    /// threads may exec again.
    fn exec_installed(&self) {
        self.exec_committed.store(false, Ordering::Release);
    }

    /// Block until this thread is the group's only one, without asking anyone
    /// to stop — the asking has already been done by whoever published the
    /// exec.
    fn wait_quiesce(&self, self_tid: i32) {
        let mut threads = self.threads.lock().unwrap();
        while !threads.iter().all(|&t| t == self_tid) {
            threads = self.threads_cv.wait(threads).unwrap();
        }
    }

    /// Rebuild the bookkeeping in the child of a fork. The copied roster
    /// still names the parent's whole group, but the child has exactly one
    /// thread — the caller — and is no participant in whatever stop the
    /// parent had in flight.
    fn reset_after_fork(&self, self_tid: i32) {
        *self.threads.lock().unwrap() = vec![self_tid];
        self.exiting.store(false, Ordering::Release);
        self.exec_committed.store(false, Ordering::Release);
        self.exit_code.store(0, Ordering::Relaxed);
        self.last_exit_status.store(0, Ordering::Relaxed);
    }

    /// Block until every guest thread but `self_tid` has left the roster.
    /// POSIX keeps a process alive until its last thread ends, so a leader
    /// whose own guest called `exit` waits here; the status it then reports
    /// is the last thread's, or the group's if an `exit_group` set one.
    fn wait_for_others(&self, self_tid: i32) -> i32 {
        let mut threads = self.threads.lock().unwrap();
        loop {
            if self.is_exiting() {
                return self.exit_code.load(Ordering::Relaxed);
            }
            if threads.iter().all(|&t| t == self_tid) {
                return self.last_exit_status.load(Ordering::Relaxed);
            }
            threads = self.threads_cv.wait(threads).unwrap();
        }
    }

    /// Block until every guest thread but `self_tid` is gone, having asked
    /// each to stop. Used by `execve`, whose image install must not pull
    /// mappings out from under a sibling still running guest code — Linux's
    /// `de_thread`, which kills the group before a new image is installed.
    fn quiesce_others(&self, self_tid: i32) {
        self.stop_others(self_tid);
        let mut threads = self.threads.lock().unwrap();
        while !threads.iter().all(|&t| t == self_tid) {
            threads = self.threads_cv.wait(threads).unwrap();
        }
    }

    /// Ask every thread but `self_tid` to end, by sending the reserved stop
    /// signal. A guest thread runs natively, with no safepoint to poll, so a
    /// signal is the only way to reach one spinning in a compute loop; its
    /// handler unwinds the thread wherever it lands.
    fn stop_others(&self, self_tid: i32) {
        let pid = unsafe { libc::getpid() };
        let threads = self.threads.lock().unwrap();
        for &tid in threads.iter() {
            if tid != self_tid {
                unsafe { libc::syscall(libc::SYS_tgkill, pid, tid, stop_signal()) };
            }
        }
    }
}

/// The signal Chimera reserves to stop a guest thread. A guest thread
/// executes natively, so nothing polls a flag; the highest real-time signal
/// is the one least likely to collide with something the guest installs, and
/// the guest's own `rt_sigaction` for it is recorded but never honored.
fn stop_signal() -> i32 {
    libc::SIGRTMAX()
}

/// The guest's signal state, mirrored rather than delegated.
///
/// The host's own signal state cannot be the guest's, for two reasons. The
/// mask must never really block the signals Chimera runs on — `SIGSYS` above
/// all, which *is* the dispatch trap — so what the kernel enforces is always
/// the guest's mask minus [`UNBLOCKABLE`], and the guest's own view has to be
/// kept here to be reported back. And a signal that arrives while Chimera is
/// midway through servicing a syscall cannot be delivered where it lands: the
/// interrupted context is the runtime's, not the guest's. Those are recorded
/// in `pending` and delivered at the next safepoint (see [`deliver_pending`]).
///
/// Every field is a `Cell` rather than the whole struct a `RefCell`: the
/// signal handlers that touch this state interrupt each other by nature, and
/// a `RefCell` borrow held across a `host_syscall` would panic the moment one
/// did. Scalar `Cell`s have no borrow to outlive the interruption.
struct Signals {
    /// The mask the guest believes is installed. What the kernel enforces is
    /// this minus [`UNBLOCKABLE`].
    mask: Cell<u64>,
    /// Signals caught while Chimera was inside the runtime, awaiting the next
    /// safepoint. Distinct from the kernel's pending set, which holds the
    /// signals the *mask* is keeping undelivered; `rt_sigpending` reports the
    /// union.
    pending: PendingQueue,
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
    in_runtime: Cell<bool>,
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
struct PendingQueue {
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

    fn clear(&self) {
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

/// One guest signal disposition, in the kernel's `rt_sigaction` shape.
#[derive(Clone, Copy)]
struct GuestAction {
    handler: u64,
    flags: u64,
    mask: u64,
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

impl Thread {
    /// Build a thread's state for the calling host thread. `runtime_fs` and
    /// `tid` are read here, so this must run *on* the thread it describes.
    fn new(process: Arc<Process>, is_leader: bool) -> Self {
        let runtime_fs = current_fs();
        Self {
            self_ptr: Cell::new(ptr::null()),
            process,
            runtime_fs,
            // Until the guest sets its own, its thread pointer is the
            // runtime's: an image that has not reached `ARCH_SET_FS` yet has
            // no TLS of its own, and leaving the base coherent keeps the
            // host thread usable in the meantime.
            guest_fs: Cell::new(runtime_fs),
            tid: Cell::new(unsafe { libc::syscall(libc::SYS_gettid) } as i32),
            is_leader: Cell::new(is_leader),
            clear_child_tid: Cell::new(None),
            spawn_report_fd: Cell::new(None),
            spawn_exec_errno: Cell::new(None),
            stop_requested: Cell::new(false),
            exit: Cell::new(None),
            exit_ctx: Box::new(UnsafeCell::new(unsafe { mem::zeroed() })),
            sig: Signals::new(),
        }
    }
}

impl Signals {
    fn new() -> Self {
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
}

/// The calling thread's [`Thread`], read out of the `gs` base.
///
/// The trap handler cannot use ordinary thread-local storage to find this.
/// It is entered with `fs` still holding the *guest's* thread pointer, so
/// every Rust thread-local — and `errno`, and the allocator's per-thread
/// state — would resolve against guest memory; and the runtime `fs` base it
/// needs to restore is itself per-thread, so the lookup that would tell it
/// what to restore cannot itself depend on TLS. `gs` closes the circle:
/// Linux x86-64 userspace leaves it unused (thread pointers live in `fs`),
/// so Chimera claims it, points it at each thread's own state, and reads the
/// self-pointer parked at offset 0 with a single instruction that touches no
/// TLS at all. The guest's own `arch_prctl(ARCH_SET_GS)` is refused for the
/// same reason the translating backend refuses it.
fn this_thread() -> &'static Thread {
    let t: *const Thread;
    unsafe {
        std::arch::asm!("mov {}, gs:[0]", out(reg) t, options(nostack, preserves_flags, readonly));
        &*t
    }
}

/// Publish `thread` as the calling host thread's, by pointing the `gs` base
/// at it. The struct's first field is a pointer to itself, so [`this_thread`]
/// is one load.
fn set_this_thread(thread: &'static Thread) -> Result<(), Error> {
    const ARCH_SET_GS: u64 = 0x1001;
    let base = thread as *const Thread;
    thread.self_ptr.set(base);
    let base = base as u64;
    match host_syscall(&SystemCall::new(
        libc::SYS_arch_prctl as u64,
        [ARCH_SET_GS, base, 0, 0, 0, 0],
    )) {
        SyscallResult::Ok(_) => Ok(()),
        SyscallResult::Error(errno) => Err(Error::io(
            "arch_prctl(ARCH_SET_GS)",
            io::Error::from_raw_os_error(errno),
        )),
    }
}

/// Run `program` natively behind syscall user dispatch; returns the guest's
/// exit code. The counterpart of the translating `exec::execv`.
pub fn execv(
    program: &Path,
    args: &[OsString],
    envs: Option<&[(OsString, OsString)]>,
    handler: Box<dyn SystemCalls>,
) -> Result<i32, Error> {
    // The exempt-range contract is load-address dependent; verify it against
    // this process rather than trusting the kernel's usual PIE placement.
    if (execv as *const () as u64) < EXEMPT_FLOOR
        || (libc::getpid as *const () as u64) < EXEMPT_FLOOR
    {
        return Err(Error::io(
            "syscall user dispatch",
            io::Error::new(
                io::ErrorKind::Unsupported,
                "runtime loaded below the dispatch-exempt floor",
            ),
        ));
    }
    // Probe support up front: switching dispatch off is idempotent, so this
    // fails only on a kernel without SUD.
    if sud_off() != 0 {
        return Err(Error::io(
            "syscall user dispatch",
            io::Error::new(
                io::ErrorKind::Unsupported,
                "kernel lacks PR_SET_SYSCALL_USER_DISPATCH (Linux 5.11+)",
            ),
        ));
    }
    // The fault handler backs `copy_from_guest`, which reads exec requests
    // out of untrusted guest memory.
    fault::install();

    let req = initial_request(program, args, envs, &*handler)?;
    let mut bump = GUEST_ARENA_BASE;
    let main = load_native(&parse_elf(&req.path)?, &mut bump)?;
    let (rip, interp_base, interp) = match &main.interp {
        Some(interp_path) => {
            let interp = load_native(&parse_elf(interp_path)?, &mut bump)?;
            (interp.entry, interp.base, Some(interp))
        }
        None => (main.entry, 0, None),
    };
    handler.on_execve(&req.path);
    let (rsp, stack_start, stack_len) =
        build_stack(&req.argv, &req.envp, &req.raw, &main, interp_base)?;

    let process = Arc::new(Process::new(handler));
    {
        let mut owned = process.regions.lock().unwrap();
        owned.extend(&main.regions);
        if let Some(interp) = &interp {
            owned.extend(&interp.regions);
        }
        owned.push((stack_start as u64, stack_len as u64));
    }
    process.bump.store(bump, Ordering::Relaxed);

    install_sigsys_handler();
    install_stop_handler();

    // The leader's `Thread` is pinned for the process's whole life, so the
    // `gs` base and the self-pointer both stay valid; a clone child's lives
    // for its host thread's closure.
    let leader = Box::leak(Box::new(Thread::new(Arc::clone(&process), true)));
    enter(leader, rip, rsp)
}

/// Bring a host thread up as a guest thread and run its guest to completion:
/// publish it for the trap handler, install the alternate stack, arm
/// dispatch, and enter guest code at `rip`/`rsp`. Returns the guest's exit
/// status when its `exit`/`exit_group` unwinds back here.
fn enter(thread: &'static Thread, rip: u64, rsp: u64) -> Result<i32, Error> {
    set_this_thread(thread)?;
    install_altstack()?;
    thread.process.register(thread.tid.get());

    let mut next = Some((rip, rsp));
    // The back edge is invisible to the compiler — control returns to the
    // `getcontext` below through a `setcontext` in a signal handler, not by
    // falling off the end — so the body does read as straight-line code that
    // ends in a diverging call.
    #[allow(clippy::never_loop)]
    loop {
        // The unwind target: `exit`/`exit_group` and the group-stop handler
        // `setcontext` back here, and the pass that follows takes one of the
        // branches below.
        unsafe { libc::getcontext(thread.exit_ctx.get()) };

        // A sibling's `execve` committed and handed the image over. This
        // thread is the group's survivor: wait out the stragglers, install,
        // and run the new program here.
        if let Some(prepared) = thread.process.take_exec_request() {
            // The stop that brought this thread here was the exec's doing,
            // not an exit; clearing both is what lets the new image run
            // instead of ending at its first syscall.
            thread.exit.set(None);
            thread.stop_requested.set(false);
            thread.process.wait_quiesce(thread.tid.get());
            next = Some(install_image(thread, prepared)?);
            thread.process.exec_installed();
        } else if let Some(code) = thread.exit.get() {
            return Ok(finish(thread, code));
        }

        let (rip, rsp) = next
            .take()
            .expect("a resumed leader always has an image to enter");
        if sud_on() != 0 {
            return Err(Error::last_os_error("enabling syscall user dispatch"));
        }
        unsafe { enter_guest(rip, rsp) }
    }
}

/// Leave guest code for good on this thread, with `code` as its status: jump
/// to the frame that entered the guest, which retires the thread through
/// [`finish`]. Never returns.
fn unwind(t: &Thread, code: i32) -> ! {
    // A spawn child ending without a committed exec is the failure case its
    // parent is still blocked on.
    report_spawn(t, t.spawn_exec_errno.get().unwrap_or(0));
    t.exit.set(Some(code));
    unsafe {
        libc::setcontext(t.exit_ctx.get());
        libc::abort();
    }
}

/// Retire a guest thread whose `exit` has unwound: honor its
/// `CLONE_CHILD_CLEARTID` word, leave the roster, and settle the status the
/// process reports. A leader that outlives its own guest waits for the last
/// sibling first, since POSIX keeps the process alive until then and reports
/// that last thread's status.
fn finish(thread: &Thread, code: i32) -> i32 {
    clear_tid_and_wake(thread);
    thread
        .process
        .last_exit_status
        .store(code, Ordering::Relaxed);
    thread.process.unregister(thread.tid.get());
    if !thread.is_leader.get() {
        return code;
    }
    let status = thread.process.wait_for_others(thread.tid.get());
    sud_off();
    status
}

/// Honor `CLONE_CHILD_CLEARTID`/`set_tid_address` on the way out: zero the
/// registered word and wake one futex waiter on it, exactly as the kernel
/// does for a real task, which is what a `pthread_join` is blocked on. The
/// word is guest memory that may already be unmapped, so the store is
/// best-effort — the kernel's own `put_user` there is unchecked too.
fn clear_tid_and_wake(thread: &Thread) {
    let Some(addr) = thread.clear_child_tid.get() else {
        return;
    };
    copy_to_guest(addr, &0u32.to_ne_bytes());
    unsafe {
        libc::syscall(libc::SYS_futex, addr, libc::FUTEX_WAKE, 1, 0, 0, 0);
    }
}

/// Arm dispatch for the calling task: every syscall issued outside
/// `[EXEMPT_FLOOR, 2^64)` traps to `SIGSYS`. The selector is null, which
/// makes dispatch unconditionally on — the guest gets no per-thread switch
/// it could flip. A raw syscall, since a `fork` child re-arms from inside
/// the `SIGSYS` handler.
fn sud_on() -> i64 {
    match host_syscall(&SystemCall::new(
        libc::SYS_prctl as u64,
        [
            PR_SET_SYSCALL_USER_DISPATCH as u64,
            PR_SYS_DISPATCH_ON,
            EXEMPT_FLOOR,
            u64::MAX - EXEMPT_FLOOR,
            0,
            0,
        ],
    )) {
        SyscallResult::Ok(v) => v,
        SyscallResult::Error(e) => -(e as i64),
    }
}

fn sud_off() -> i64 {
    match host_syscall(&SystemCall::new(
        libc::SYS_prctl as u64,
        [
            PR_SET_SYSCALL_USER_DISPATCH as u64,
            PR_SYS_DISPATCH_OFF,
            0,
            0,
            0,
            0,
        ],
    )) {
        SyscallResult::Ok(v) => v,
        SyscallResult::Error(e) => -(e as i64),
    }
}

/// Map an image for native execution, drawing `ET_DYN` placement from the
/// arena bump pointer and advancing it past whatever landed there.
fn load_native(parsed: &ParsedElf, bump: &mut u64) -> Result<LoadedElf, Error> {
    let elf = map_elf_native(parsed, *bump)?;
    for &(start, len) in &elf.regions {
        if (GUEST_ARENA_BASE..GUEST_ARENA_CEILING).contains(&start) {
            let end = (start + len + ARENA_IMAGE_GAP + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
            if end > *bump {
                *bump = end;
            }
        }
    }
    Ok(elf)
}

/// Jump into the guest: capture a context, aim it at the guest entry with the
/// guest stack and the zeroed registers a fresh `execve` presents (`rdx`
/// doubles as the atexit-function register, so a stale value would be
/// registered and called), and resume it. `setcontext` never returns here —
/// the run ends through the `exit_ctx` unwind.
unsafe fn enter_guest(rip: u64, rsp: u64) -> ! {
    unsafe {
        let mut ctx: libc::ucontext_t = mem::zeroed();
        libc::getcontext(&mut ctx);
        aim_context(&mut ctx.uc_mcontext.gregs, rip, rsp);
        libc::setcontext(&ctx);
        libc::abort();
    }
}

/// Point a captured register set at a fresh image: entry `rip`, initial
/// `rsp`, and every register `setcontext`/`sigreturn` will restore zeroed,
/// the state a native `execve` hands over.
fn aim_context(gregs: &mut [libc::greg_t; 23], rip: u64, rsp: u64) {
    for r in [
        libc::REG_RBX,
        libc::REG_RBP,
        libc::REG_R12,
        libc::REG_R13,
        libc::REG_R14,
        libc::REG_R15,
        libc::REG_RDI,
        libc::REG_RSI,
        libc::REG_RDX,
        libc::REG_RCX,
        libc::REG_R8,
        libc::REG_R9,
        libc::REG_R10,
        libc::REG_R11,
        libc::REG_RAX,
    ] {
        gregs[r as usize] = 0;
    }
    gregs[libc::REG_RSP as usize] = rsp as libc::greg_t;
    gregs[libc::REG_RIP as usize] = rip as libc::greg_t;
}

/// Whether the CPU and kernel expose `rdfsbase`/`wrfsbase` to userspace
/// (`CPUID.7.0:EBX.FSGSBASE[0]` plus `CR4.FSGSBASE`, which Linux sets when it
/// advertises the `fsgsbase` hwcap). Read once: [`on_sigsys`] swaps the `fs`
/// base twice per dispatched syscall, and a pair of `arch_prctl` calls there
/// costs more than the trap itself.
fn fsgsbase_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let leaf = std::arch::x86_64::__cpuid_count(7, 0);
        if leaf.ebx & 1 == 0 {
            return false;
        }
        // The instruction faults with #UD unless the kernel enabled
        // `CR4.FSGSBASE`; the auxv hwcap2 bit is how it says so.
        unsafe { libc::getauxval(libc::AT_HWCAP2) & (1 << 1) != 0 }
    })
}

fn current_fs() -> u64 {
    if fsgsbase_available() {
        let base: u64;
        unsafe { std::arch::asm!("rdfsbase {}", out(reg) base, options(nomem, nostack)) };
        return base;
    }
    let mut base: u64 = 0;
    host_syscall(&SystemCall::new(
        libc::SYS_arch_prctl as u64,
        [ARCH_GET_FS, &mut base as *mut u64 as u64, 0, 0, 0, 0],
    ));
    base
}

/// Install `base` as the thread's `fs`. The fallback is a raw syscall through
/// [`host_syscall`], not glibc, because it is called from the `SIGSYS` handler
/// before TLS is usable.
fn set_fs(base: u64) {
    if fsgsbase_available() {
        unsafe { std::arch::asm!("wrfsbase {}", in(reg) base, options(nomem, nostack)) };
        return;
    }
    host_syscall(&SystemCall::new(
        libc::SYS_arch_prctl as u64,
        [ARCH_SET_FS, base, 0, 0, 0, 0],
    ));
}

/// The handler needs a stack of its own: an `execve` intercept unmaps the
/// old guest stack — the very stack the handler would otherwise be running
/// on.
fn install_altstack() -> Result<(), Error> {
    const ALT_STACK_SIZE: usize = 1024 * 1024;
    let stack = unsafe {
        libc::mmap(
            ptr::null_mut(),
            ALT_STACK_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if stack == libc::MAP_FAILED {
        return Err(Error::last_os_error("SUD altstack mmap"));
    }
    let ss = libc::stack_t {
        ss_sp: stack,
        ss_flags: 0,
        ss_size: ALT_STACK_SIZE,
    };
    if unsafe { libc::sigaltstack(&ss, ptr::null_mut()) } != 0 {
        return Err(Error::last_os_error("sigaltstack"));
    }
    Ok(())
}

/// Install the dispatch trap handler.
///
/// Its `sa_mask` is empty, so guest signals stay deliverable for the duration
/// of the trap: that is what lets one interrupt a *forwarded* blocking
/// syscall, so an unhandled `SIGINT` arriving while the guest is parked in
/// `read` is felt rather than waited out. What arrives goes to
/// [`on_guest_signal`], which defers it to the safepoint at the tail of
/// [`on_sigsys`] rather than letting the guest's handler run against the
/// runtime's context. The kernel blocks `SIGSYS` inside its own handler,
/// which is harmless: every syscall the runtime issues comes from the exempt
/// range and traps nothing.
fn install_sigsys_handler() {
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = on_sigsys as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigaction(libc::SIGSYS, &sa, ptr::null_mut());
    }
}

/// Install the handler for the reserved stop signal (see [`stop_signal`]).
///
/// A guest thread executes natively, so there is no safepoint for a
/// group-wide stop to be noticed at — a thread spinning in a compute loop
/// would never reach one. The signal is the safepoint: wherever it lands, its
/// handler unwinds that thread out of guest code and into the frame that
/// entered it, which retires the thread the way its own `exit` would.
///
/// `SA_ONSTACK` puts the unwind on Chimera's alternate stack rather than
/// whatever guest stack was interrupted, and the mask is full because the
/// handler never returns to what it interrupted.
fn install_stop_handler() {
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = on_stop as *const () as usize;
        libc::sigfillset(&mut sa.sa_mask);
        libc::sigdelset(&mut sa.sa_mask, libc::SIGSEGV);
        libc::sigdelset(&mut sa.sa_mask, libc::SIGBUS);
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigaction(stop_signal(), &sa, ptr::null_mut());
    }
}

/// Block the reserved stop signal for as long as this value lives.
///
/// The stop handler does not return to what it interrupted — it unwinds the
/// thread — so any lock held at the moment it lands is abandoned still
/// locked, and the next thread to want it waits forever. The `execve` install
/// path holds exactly such a lock while it tears the old image down, and it
/// is also the one path a concurrent exec might try to stop. Blocking the
/// signal makes the teardown uninterruptible; the stop is merely deferred,
/// and lands at the release.
struct StopBlocked(u64);

impl StopBlocked {
    fn new() -> Self {
        let set = sig_bit(stop_signal());
        let mut old: u64 = 0;
        host_syscall(&SystemCall::new(
            libc::SYS_rt_sigprocmask as u64,
            [
                libc::SIG_BLOCK as u64,
                &set as *const u64 as u64,
                &mut old as *mut u64 as u64,
                8,
                0,
                0,
            ],
        ));
        Self(old)
    }
}

impl Drop for StopBlocked {
    fn drop(&mut self) {
        let set = self.0;
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
}

/// End this guest thread on the group's behalf: a sibling issued `exit_group`
/// or committed an `execve`.
///
/// Where the thread can be ended depends on what the signal interrupted, the
/// same split [`on_guest_signal`] makes. Guest code holds nothing of
/// Chimera's, so a thread interrupted there unwinds on the spot — which is
/// the whole point of the mechanism, since a guest spinning without a syscall
/// in sight has no other way to be reached. Runtime code is different: this
/// handler never returns to what it interrupted, so a lock held at that
/// moment — one of the runtime's, or one inside the embedder's handler —
/// would be abandoned still locked and strand every thread that wants it
/// next. Those are flagged and taken at the safepoint in [`on_sigsys`],
/// where the syscall being serviced has finished and nothing is held.
extern "C" fn on_stop(_signo: libc::c_int, _info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    let t = this_thread();
    let entry_fs = current_fs();
    set_fs(t.runtime_fs);
    let uc = unsafe { &mut *(uc as *mut libc::ucontext_t) };
    let rip = uc.uc_mcontext.gregs[libc::REG_RIP as usize] as u64;
    if t.sig.in_runtime.get() || rip >= EXEMPT_FLOOR {
        t.stop_requested.set(true);
        set_fs(entry_fs);
        return;
    }
    unwind(t, t.process.exit_code.load(Ordering::Relaxed));
}

/// Chimera's own alternate stack, as a `stack_t`. Installed once by
/// [`install_altstack`] and recorded so a frame Chimera builds can name it
/// for `rt_sigreturn` to restore.
fn chimera_altstack() -> libc::stack_t {
    let mut ss: libc::stack_t = unsafe { mem::zeroed() };
    unsafe { libc::sigaltstack(ptr::null(), &mut ss) };
    ss.ss_flags = 0;
    ss
}

/// One trapped guest syscall. The first statements run with the *guest's*
/// `fs` base, so nothing before `set_fs` may touch TLS — no libc wrappers, no
/// `errno`, no thread locals.
///
/// The tail is the backend's safepoint. A guest signal that arrived while the
/// syscall was being serviced was deferred by [`on_guest_signal`], because
/// the context it interrupted was the runtime's; here the context describes
/// the guest again — the syscall has its result — so the deferred signals can
/// be delivered against it.
extern "C" fn on_sigsys(_signo: libc::c_int, info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    let t = this_thread();
    set_fs(t.runtime_fs);
    t.sig.in_runtime.set(true);

    let uc = unsafe { &mut *(uc as *mut libc::ucontext_t) };
    let info = unsafe { &*(info as *const SigsysInfo) };
    let nr = info.syscall as u32 as u64;
    refresh_mask(t, uc);
    let gregs = &uc.uc_mcontext.gregs;
    t.sig
        .on_alt
        .set(on_sig_stack(t, gregs[libc::REG_RSP as usize] as u64));
    let args = [
        gregs[libc::REG_RDI as usize] as u64,
        gregs[libc::REG_RSI as usize] as u64,
        gregs[libc::REG_RDX as usize] as u64,
        gregs[libc::REG_R10 as usize] as u64,
        gregs[libc::REG_R8 as usize] as u64,
        gregs[libc::REG_R9 as usize] as u64,
    ];
    let mut call = SystemCall::new(nr, args);
    dispatch(t, &mut call, uc, info);
    uc.uc_mcontext.gregs[libc::REG_RAX as usize] = call.return_value() as libc::greg_t;

    // A syscall the kernel handed back as `EINTR` was interrupted by a signal
    // Chimera caught and deferred; whether the guest ever sees the `EINTR` is
    // its own `SA_RESTART` choice, applied before the frame is built so the
    // handler returns onto the restarted call.
    if call.return_value() == -(libc::EINTR as i64) && restart_wanted(t) {
        restart_syscall(uc, info, nr);
    }

    t.sig.in_runtime.set(false);
    // The safepoint a group stop deferred to (see `on_stop`): nothing of the
    // runtime's or the embedder's is held here, so the thread can end.
    if t.stop_requested.get() {
        unwind(t, t.process.exit_code.load(Ordering::Relaxed));
    }
    // `sigreturn` restores the mask from the context, so the guest's own —
    // filtered — mask has to be published here rather than left as the one
    // the trap entered with. A delivery overwrites it with the handler's.
    let mask = t.sig.mask.get();
    uc.uc_sigmask = sigset_from(host_mask(mask));
    deliver_pending(t, uc, mask, mask);

    set_fs(t.guest_fs.get());
}

/// Whether every deferred signal wants the interrupted syscall restarted.
/// The kernel's rule: a handler carrying `SA_RESTART` resumes the call, one
/// without it lets `EINTR` through, and a signal with no handler at all (a
/// deferred one whose disposition has since been reset) restarts.
fn restart_wanted(t: &Thread) -> bool {
    let deliverable = t.sig.pending.mask() & !t.sig.mask.get();
    if deliverable == 0 {
        return false;
    }
    let mut rest = deliverable;
    while rest != 0 {
        let signo = rest.trailing_zeros() as usize + 1;
        rest &= rest - 1;
        let action = t.process.actions[signo].load();
        if action.handler != libc::SIG_DFL as u64
            && action.handler != libc::SIG_IGN as u64
            && action.flags & libc::SA_RESTART as u64 == 0
        {
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
fn restart_syscall(uc: &mut libc::ucontext_t, info: &SigsysInfo, nr: u64) {
    const SYSCALL_INSN_LEN: u64 = 2;
    uc.uc_mcontext.gregs[libc::REG_RIP as usize] =
        (info.call_addr - SYSCALL_INSN_LEN) as libc::greg_t;
    uc.uc_mcontext.gregs[libc::REG_RAX as usize] = nr as libc::greg_t;
}

/// Drive one trapped syscall: the intercepts this backend owns, then the
/// embedder hooks — the same shape as the translating driver
/// (`crate::syscall`), minus everything that exists only to protect a code
/// cache.
fn dispatch(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t, info: &SigsysInfo) {
    let handler = &*t.process.handler;
    handler.pre_syscall(call);

    let nr = call.number as i64;
    match nr {
        // Thread-local: end this thread alone. Its host thread unwinds to
        // the frame that entered the guest and retires there; the rest of the
        // group runs on, and a leader that exits this way waits out its
        // siblings, since POSIX keeps the process alive until the last one
        // ends. Forwarding would end the embedder's thread, not the guest's.
        libc::SYS_exit => unwind(t, call.args[0] as i32),
        // Process-wide: end the whole group from whichever thread called it.
        // The status is published first, then every sibling is stopped —
        // guest threads run natively, so a signal is the only thing that
        // reaches one spinning without a syscall in sight.
        libc::SYS_exit_group => {
            let code = call.args[0] as i32;
            t.process.exit_code.store(code, Ordering::Relaxed);
            t.process.exiting.store(true, Ordering::Release);
            t.process.stop_others(t.tid.get());
            unwind(t, code);
        }
        // Forwarding an exec would replace the whole process image — and the
        // kernel clears syscall user dispatch across a real `execve`, so the
        // replacement would run unintercepted. Emulate it in place instead:
        // tear the guest image down, load the new one, and point the trapped
        // context at its entry.
        libc::SYS_execve | libc::SYS_execveat => do_execve(t, call, uc),
        libc::SYS_arch_prctl => match call.args[0] {
            // The guest owns the real `fs` while it runs, but the *handler*
            // must run on the runtime's, so the base is virtualized: recorded
            // here, installed by `on_sigsys` on its way out.
            ARCH_SET_FS => {
                t.guest_fs.set(call.args[1]);
                call.set_result(SyscallResult::Ok(0));
            }
            ARCH_GET_FS => {
                let base = t.guest_fs.get();
                if copy_to_guest(call.args[1], &base.to_ne_bytes()) {
                    call.set_result(SyscallResult::Ok(0));
                } else {
                    call.set_result(SyscallResult::Error(libc::EFAULT));
                }
            }
            _ => handler.do_syscall(call),
        },
        // The guest reconfiguring dispatch is the sandbox turning itself off.
        libc::SYS_prctl if call.args[0] == PR_SET_SYSCALL_USER_DISPATCH as u64 => {
            call.set_result(SyscallResult::Error(libc::EPERM));
        }
        libc::SYS_rt_sigaction => do_sigaction(t, call),
        libc::SYS_rt_sigprocmask => do_sigprocmask(t, call),
        libc::SYS_rt_sigsuspend => do_sigsuspend(t, call, uc),
        libc::SYS_rt_sigpending => do_sigpending(t, call),
        libc::SYS_sigaltstack => do_sigaltstack(t, call),
        libc::SYS_clone => do_clone(t, call, uc, info),
        libc::SYS_clone3 => do_clone3(t, call, uc, info),
        // A real vfork child shares the arena bump pointer and `guest_fs`
        // cells with a suspended parent; degrade to fork, whose
        // copy-on-write child owns its copies.
        libc::SYS_vfork | libc::SYS_fork => {
            let mut forked = SystemCall::new(libc::SYS_fork as u64, [0; 6]);
            forward_fork(t, &mut forked, uc, None);
            call.set_result(forked.result().expect("fork always sets a result"));
        }
        libc::SYS_mmap => do_mmap(t, call),
        // io_uring queues syscalls the kernel services without ever passing
        // them back through this driver.
        libc::SYS_io_uring_setup | libc::SYS_io_uring_enter | libc::SYS_io_uring_register => {
            call.set_result(SyscallResult::Error(libc::EPERM));
        }
        _ => handler.do_syscall(call),
    }

    handler.post_syscall(call);
}

/// Emulated `execve`: validate and parse in place (a failure reports
/// `-errno` and resumes the old image untouched), then commit — tear down
/// the old guest, map the new one, and rewrite the trapped context so
/// `sigreturn` resumes at the fresh entry point.
fn do_execve(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t) {
    let prepared = match prepare_exec(call.number, &call.args, &*t.process.handler) {
        Ok(prepared) => prepared,
        Err(err) => {
            let errno = exec_errno(&err).unwrap_or(libc::EIO);
            // Remembered, not reported: `posix_spawnp` walks `$PATH` inside
            // the child, so a failed attempt is routinely followed by one
            // that succeeds. Only the child's exit makes this final.
            t.spawn_exec_errno.set(Some(errno));
            call.set_result(SyscallResult::Error(errno));
            return;
        }
    };
    // A spawn child reaching a loadable image is a successful spawn: unblock
    // the parent now, before the install, so it returns the child's PID while
    // the report pipe is still open — the install's close-on-exec sweep is
    // about to close it.
    report_spawn(t, 0);
    match Some(prepared) {
        Some(prepared) if !t.is_leader.get() => {
            // Linux hands the exec'ing thread the leader's identity, so the
            // new image's only thread has `tid == pid`. Chimera cannot move a
            // TID between host threads, so it moves the *image* instead: the
            // leader is stopped, picks the request up in `enter`, and runs
            // the new program on the host thread whose TID already is the
            // pid. This thread's own guest ends here, like every other
            // sibling `de_thread` takes.
            // A refused publish means a sibling's exec is already dissolving
            // this group, this thread with it. Nothing more to do: the stop
            // already in flight takes it, exactly like any other sibling.
            match t.process.publish_exec(prepared) {
                None => t.process.stop_others(t.tid.get()),
                // Refused: a sibling's exec is already dissolving this group,
                // this thread with it, so there is nothing more to do — the
                // stop already in flight takes it like any other sibling. The
                // image it prepared is deliberately leaked rather than
                // dropped: closing its files here would race the winner's
                // close-on-exec sweep, which is enumerating descriptors on
                // another thread, and a number freed mid-sweep can be reissued
                // to something the runtime still owns and then closed out from
                // under it. The winner's sweep closes these instead, exactly
                // once. Every other local this thread holds is abandoned the
                // same way, since `unwind` runs no destructors.
                Some(rejected) => mem::forget(rejected),
            }
            unwind(t, 0);
        }
        Some(prepared) => match install_image(t, prepared) {
            Ok((rip, rsp)) => {
                aim_context(&mut uc.uc_mcontext.gregs, rip, rsp);
                call.set_result(SyscallResult::Ok(0))
            }
            // Past teardown there is no image to resume; end the run the way
            // a shell reports an exec that died mid-flight.
            Err(err) => {
                eprintln!("chimera: execve: {err}");
                unwind(t, 127);
            }
        },
        None => unreachable!("the prepared image was taken above"),
    }
}

fn install_image(t: &Thread, prepared: PreparedExec) -> Result<(u64, u64), Error> {
    let _uninterruptible = StopBlocked::new();
    let PreparedExec {
        req,
        parsed,
        parsed_interp,
    } = prepared;
    let ExecRequest {
        path, argv, envp, ..
    } = req;

    // Linux's `de_thread`: every other thread of the group dies before a new
    // image is installed, whichever thread called exec. Here it is also a
    // safety requirement — the teardown below unmaps the arena, and a sibling
    // still executing guest code out of it would fault on the next
    // instruction.
    t.process.quiesce_others(t.tid.get());
    // The exec'ing thread takes the group over. If it was not the leader, the
    // old leader has just unwound and is waiting for the group to end; this
    // thread is now the group, and its status is the process's.
    t.is_leader.set(true);

    t.process.handler.on_execve(&path);
    let mut keep = vec![parsed.as_raw_fd()];
    if let Some(interp) = &parsed_interp {
        keep.push(interp.as_raw_fd());
    }
    close_cloexec_fds(&keep)?;

    // Tear down the old guest address space: the tracked out-of-arena
    // regions, then the arena wholesale up to its watermark. Guest mappings
    // the kernel placed on its own (an explicit high hint) are the leak this
    // proof of concept accepts.
    for (start, len) in t.process.regions.lock().unwrap().drain(..) {
        unsafe { libc::munmap(start as *mut libc::c_void, len as usize) };
    }
    let watermark = t.process.bump.load(Ordering::Relaxed);
    if watermark > GUEST_ARENA_BASE {
        unsafe {
            libc::munmap(
                GUEST_ARENA_BASE as *mut libc::c_void,
                (watermark - GUEST_ARENA_BASE) as usize,
            )
        };
    }
    t.process.bump.store(GUEST_ARENA_BASE, Ordering::Relaxed);

    let mut bump = GUEST_ARENA_BASE;
    let main = load_native(&parsed, &mut bump)?;
    let (rip, interp_base, interp) = match &parsed_interp {
        Some(parsed_interp) => {
            let interp = load_native(parsed_interp, &mut bump)?;
            (interp.entry, interp.base, Some(interp))
        }
        None => (main.entry, 0, None),
    };
    let (rsp, stack_start, stack_len) = build_stack(
        &argv,
        &envp,
        path.as_os_str().as_encoded_bytes(),
        &main,
        interp_base,
    )?;
    t.process.bump.store(bump, Ordering::Relaxed);
    let mut regions = t.process.regions.lock().unwrap();
    regions.extend(&main.regions);
    if let Some(interp) = &interp {
        regions.extend(&interp.regions);
    }
    regions.push((stack_start as u64, stack_len as u64));

    reset_guest_signals(t);
    // A fresh image has no TLS yet; hand the handler epilogue a base that at
    // least keeps the host thread coherent until the new libc sets its own.
    t.guest_fs.set(t.runtime_fs);
    Ok((rip, rsp))
}

/// POSIX `execve` resets caught signals to their default disposition and
/// leaves ignored ones ignored. The mask and the pending set survive an exec,
/// so only the disposition table is swept.
fn reset_guest_signals(t: &Thread) {
    for sig in 1..NSIG as i32 {
        let action = t.process.actions[sig as usize].load();
        if action.handler != libc::SIG_DFL as u64 && action.handler != libc::SIG_IGN as u64 {
            t.process.actions[sig as usize].store(GuestAction::default());
            install_host_action(t, sig);
        }
    }
}

/// The mask the kernel actually enforces for a guest that asked for `mask`.
fn host_mask(mask: u64) -> u64 {
    mask & !UNBLOCKABLE
}

/// Install the kernel-enforced mask for the guest's current one. Called
/// whenever [`Signals::mask`] changes, so an unblocked signal the kernel has
/// been holding is delivered right away rather than at the next safepoint.
fn sync_host_mask(t: &Thread) {
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

fn read_guest_sigset(ptr: u64) -> Option<u64> {
    let mut raw = [0u8; 8];
    copy_from_guest(ptr, &mut raw).then(|| u64::from_ne_bytes(raw))
}

/// `rt_sigprocmask`, serviced against the mirrored mask: the guest's own view
/// is composed and reported here, and only the filtered result reaches the
/// kernel.
fn do_sigprocmask(t: &Thread, call: &mut SystemCall) {
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
fn do_sigpending(t: &Thread, call: &mut SystemCall) {
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
fn do_sigaltstack(t: &Thread, call: &mut SystemCall) {
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
fn do_sigsuspend(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t) {
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
fn do_sigaction(t: &Thread, call: &mut SystemCall) {
    let signo = call.args[0] as i32;
    if signo < 1 || signo >= NSIG as i32 || call.args[3] != 8 {
        call.set_result(SyscallResult::Error(libc::EINVAL));
        return;
    }
    let old = t.process.actions[signo as usize].load();
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
        t.process.actions[signo as usize].store(GuestAction {
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
            // The guest's own restorer is what it gave us; Chimera's
            // substitution is an implementation detail of delivery and is
            // not reported back.
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
    let action = t.process.actions[signo as usize].load();
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        if action.handler == libc::SIG_DFL as u64 || action.handler == libc::SIG_IGN as u64 {
            sa.sa_sigaction = action.handler as usize;
            libc::sigemptyset(&mut sa.sa_mask);
            sa.sa_flags = 0;
        } else {
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
/// is the tail of [`on_sigsys`] — by which point the syscall being serviced
/// has a result and the context describes the guest again.
///
/// Deferring is also what makes a forwarded blocking syscall interruptible.
/// Chimera's handler carries no `SA_RESTART`, so the kernel hands the
/// interrupted `read` back as `EINTR` rather than resuming it, and the
/// dispatch path decides whether to restart it on the guest's behalf.
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
        refresh_mask(t, uc);
        let mask = t.sig.mask.get();
        deliver(t, signo, &raw_info, uc, mask, mask);
    }

    set_fs(entry_fs);
}

/// Take every deferred signal `base` leaves unblocked, building a frame for
/// each onto `uc`. Called at the safepoint on the way out of [`on_sigsys`].
/// Returns how many were delivered.
///
/// `base` is the mask the guest is under while the signals are taken, and
/// `restore` the one the last handler to run returns to — the same value,
/// except after a `sigsuspend`, whose caller gets its original mask back
/// rather than the one it waited under.
///
/// Frames stack, so the last one built is the first the guest enters. Each
/// handler returns to the mask the *next* one to run needs, and the last
/// returns to `restore`, which is why the chain walks backwards from it.
fn deliver_pending(t: &Thread, uc: &mut libc::ucontext_t, base: u64, restore: u64) -> usize {
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
    let action = t.process.actions[signo as usize].load();
    if action.handler == libc::SIG_DFL as u64 || action.handler == libc::SIG_IGN as u64 {
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
    let alt = t.sig.alt.get();
    // Whether to switch stacks is decided against the context being saved,
    // not against trap entry: a second frame built at the same safepoint sees
    // the first one's stack pointer and stacks onto it rather than starting
    // over at the top and overwriting it.
    let use_alt = action.flags & libc::SA_ONSTACK as u64 != 0
        && alt.ss_flags & SS_DISABLE == 0
        && !on_sig_stack(t, saved.gregs[libc::REG_RSP as usize] as u64);
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
        t.process.actions[signo as usize].store(GuestAction::default());
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
    uc.uc_sigmask = sigset_from(host_mask(new_mask));
}

/// Whether `rsp` lies within the guest's alternate signal stack — the
/// kernel's `on_sig_stack`, and the same answer `sigaltstack` reports as
/// `SS_ONSTACK`.
fn on_sig_stack(t: &Thread, rsp: u64) -> bool {
    let alt = t.sig.alt.get();
    if alt.ss_flags & SS_DISABLE != 0 {
        return false;
    }
    let base = alt.ss_sp as u64;
    (base..base + alt.ss_size as u64).contains(&rsp)
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

/// Re-derive the guest's mask from the context the kernel handed over.
///
/// Chimera loses control at the end of a guest signal handler: the restorer
/// issues `rt_sigreturn`, the kernel restores the mask from the frame, and no
/// code of Chimera's runs in between. A mirrored mask maintained only by
/// `rt_sigprocmask` would therefore stay stuck at the handler's mask forever
/// after the first delivery, and every later signal would be filtered out as
/// blocked and never delivered at all.
///
/// So the kernel is the authority, and the mirror only carries what the
/// kernel cannot: the guest's intent for the [`UNBLOCKABLE`] signals, which
/// are never really blocked and so never appear in a context's mask. Every
/// entry into Chimera refreshes the rest from the interrupted context, whose
/// `uc_sigmask` is exactly the mask that was in force.
fn refresh_mask(t: &Thread, uc: &libc::ucontext_t) {
    t.sig
        .mask
        .set(sigmask_of(&uc.uc_sigmask) | (t.sig.mask.get() & UNBLOCKABLE));
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

/// Forward a fork-shaped call and re-arm dispatch in the child.
///
/// The kernel does **not** inherit the syscall-user-dispatch configuration
/// across `fork`/`clone`: the child's `SYSCALL_WORK_SYSCALL_USER_DISPATCH`
/// work flag is cleared, so without this its every syscall would go straight
/// to the host kernel — the guest's child escaping the sandbox entirely, and
/// silently, since an escaped syscall succeeds. The child re-arms here,
/// before it returns to guest code, so the first guest instruction it
/// executes is already intercepted. This is the one place a fork is
/// forwarded, and the whole backend's confinement of child processes rests
/// on it.
///
/// The handler's locks are held across the copy, the `pthread_atfork`
/// discipline the translating backend applies for the same reason (see
/// `SystemCalls::lock_for_fork`).
/// The `posix_spawn` shape, which needs more than a fork.
///
/// glibc issues `clone(CLONE_VM | CLONE_VFORK)` and relies on both flags: it
/// stays suspended until the child execs or exits, and reads the child's
/// error out of the memory they share. A fork gives neither, so the outcome
/// travels back over a pipe instead and the parent blocks on it, which is
/// what makes a missing program fail `posix_spawn` synchronously with
/// `ENOENT` rather than only surfacing as the child's exit status.
///
/// The report waits for the child's *exit*, not its first failed `execve`:
/// `posix_spawnp` walks `$PATH` inside the child, one `execve` per candidate,
/// and an early failure is routinely followed by one that succeeds.
fn spawned(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t, child_stack: Option<u64>) {
    let mut fds = [0i32; 2];
    // Without the pipe the spawn still works; it just loses the synchronous
    // error report.
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        forward_fork(t, call, uc, child_stack);
        return;
    }
    let read_fd = fds[0];
    // Move the write end clear of the low descriptors a spawn's file actions
    // typically remap, so the child's own `dup2`/`close` cannot clobber it.
    let mut write_fd = fds[1];
    let moved = unsafe { libc::fcntl(write_fd, libc::F_DUPFD_CLOEXEC, 100) };
    if moved >= 0 {
        unsafe { libc::close(write_fd) };
        write_fd = moved;
    }

    forward_fork(t, call, uc, child_stack);
    let result = call.result();

    if let Some(SyscallResult::Ok(0)) = result {
        unsafe { libc::close(read_fd) };
        t.spawn_report_fd.set(Some(write_fd));
        return;
    }
    unsafe { libc::close(write_fd) };
    let Some(SyscallResult::Ok(child_pid)) = result else {
        unsafe { libc::close(read_fd) };
        return;
    };

    let mut buf = [0u8; 4];
    let n = unsafe { libc::read(read_fd, buf.as_mut_ptr().cast(), buf.len()) };
    unsafe { libc::close(read_fd) };
    if n == buf.len() as isize {
        let errno = i32::from_ne_bytes(buf);
        if errno != 0 {
            // The child's exec failed and it is about to `_exit`; reap it so
            // it leaves no zombie — the caller never gets a PID to wait on —
            // and report the errno the way the shared-memory path would have.
            unsafe { libc::waitpid(child_pid as libc::pid_t, ptr::null_mut(), 0) };
            call.set_result(SyscallResult::Error(errno));
        }
    }
}

/// Report a spawn child's `execve` outcome to its blocked parent. A committed
/// exec closes the pipe with nothing written, which the parent reads as EOF
/// and takes for success; a child that exits without one writes the errno of
/// its last failed attempt.
fn report_spawn(t: &Thread, errno: i32) {
    let Some(fd) = t.spawn_report_fd.take() else {
        return;
    };
    if errno != 0 {
        let buf = errno.to_ne_bytes();
        unsafe { libc::write(fd, buf.as_ptr().cast(), buf.len()) };
    }
    unsafe { libc::close(fd) };
}

fn forward_fork(
    t: &Thread,
    call: &mut SystemCall,
    uc: &mut libc::ucontext_t,
    child_stack: Option<u64>,
) {
    let hold = t.process.handler.lock_for_fork();
    let result = host_syscall(call);
    if let SyscallResult::Ok(0) = result {
        // The guest asked for its child to run on a stack of its own (the
        // `posix_spawn` shape); the kernel was not allowed to install it, so
        // it goes into the context the child resumes through.
        if let Some(sp) = child_stack {
            uc.uc_mcontext.gregs[libc::REG_RSP as usize] = sp as libc::greg_t;
        }
        sud_on();
        // The pid the guest-memory writes are aimed at is cached, and the
        // cache is the parent's. Left stale, every `copy_to_guest` in the
        // child would land in the *parent's* address space and report
        // success — the child's own writes silently lost and the parent's
        // memory corrupted. libc's `pthread_atfork` hook does not cover this
        // fork: the guest's `clone` is forwarded as a raw syscall and never
        // runs libc's handlers.
        crate::sys::mmap::reset_cached_pid();
        // POSIX hands the child an empty pending set. The kernel clears its
        // own; the deferred set Chimera keeps is ordinary memory the fork
        // copied, so it has to be cleared by hand or the child would take a
        // signal only its parent was sent.
        t.sig.pending.clear();
        // A fork copies only the calling thread, so whatever this thread was
        // in the parent, in the child it is the whole process: its TID is new,
        // it is the leader, and the roster it inherited — describing the
        // parent's group — describes threads that do not exist here. The
        // parent's group-wide stop, if one was in flight, is not the child's
        // to finish either.
        t.tid.set(unsafe { libc::syscall(libc::SYS_gettid) } as i32);
        t.is_leader.set(true);
        t.process.reset_after_fork(t.tid.get());
    }
    drop(hold);
    call.set_result(result);
}

/// `clone` shapes: a plain fork forwards (the copy-on-write child carries the
/// runtime and its own copy of the [`Task`]); the `posix_spawn` shape
/// (`CLONE_VM | CLONE_VFORK`) degrades to fork, since a child sharing the
/// arena bump pointer and `guest_fs` cells would race its parent; any other
/// shared-memory shape — a thread — is refused, since a second native guest
/// thread would race the single-task state here.
fn do_clone(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t, info: &SigsysInfo) {
    let flags = call.args[0];
    let vm = flags & libc::CLONE_VM as u64 != 0;
    let vfork = flags & libc::CLONE_VFORK as u64 != 0;
    if flags & libc::CLONE_THREAD as u64 != 0 {
        let result = spawn_thread(
            t,
            uc,
            info,
            CloneRequest {
                flags,
                child_stack: call.args[1],
                parent_tid: call.args[2],
                child_tid: call.args[3],
                tls: call.args[4],
            },
        );
        call.set_result(result);
        return;
    }
    // `CLONE_VM` without `CLONE_THREAD` is a second process sharing this
    // address space — and with it the arena bump pointer and every thread's
    // state, which two processes cannot share. The `posix_spawn` shape pairs
    // it with `CLONE_VFORK` and only ever runs to an `execve`, so it degrades
    // to a fork; anything else is refused.
    if vm && !vfork {
        call.set_result(SyscallResult::Error(libc::EPERM));
        return;
    }
    let mut child_stack = None;
    if vm && vfork {
        call.args[0] = flags & !(libc::CLONE_VM as u64 | libc::CLONE_VFORK as u64);
        // The stack argument must not reach the kernel with it. `clone` sets
        // the child's stack pointer whatever the flags, so a forwarded fork
        // carrying one comes back *inside Chimera's own trap handler* running
        // on the guest's spawn stack — a few pages with no frame under them —
        // and the first thing the runtime touches faults. Dropped here, the
        // child keeps the parent's stack, copy-on-write, the way a real fork
        // does; the guest still needs to resume on the stack it asked for, so
        // the value is installed into the child's resume context instead.
        child_stack = (call.args[1] != 0).then_some(call.args[1]);
        call.args[1] = 0;
        spawned(t, call, uc, child_stack);
        return;
    }
    forward_fork(t, call, uc, child_stack);
}

/// The arguments a thread-creating `clone` carries, in whichever shape it
/// arrived.
struct CloneRequest {
    flags: u64,
    child_stack: u64,
    parent_tid: u64,
    child_tid: u64,
    tls: u64,
}

/// Create a guest thread.
///
/// The kernel's own `clone` cannot be forwarded for this. The task it makes
/// would come back from the syscall *inside Chimera's trap handler*, on the
/// guest's thread stack, with no `gs` of its own, no alternate stack, and —
/// since dispatch configuration does not survive a clone any more than it
/// survives a fork — no interception at all. So Chimera creates the host
/// thread itself and lets that thread build its own state before any guest
/// instruction runs on it.
///
/// The child enters guest code exactly where the kernel would have put it:
/// at the instruction after the guest's own `syscall`, with the parent's
/// register file, `rax` zeroed to report the child's side of the clone, and
/// its own stack. The parent gets the child's kernel TID, which is the TID
/// the guest sees, so its later `futex` and `tgkill` reach this host thread.
fn spawn_thread(
    t: &Thread,
    uc: &libc::ucontext_t,
    info: &SigsysInfo,
    req: CloneRequest,
) -> SyscallResult {
    let process = Arc::clone(&t.process);
    // The child resumes with the parent's registers, its own stack, and the
    // clone's zero return.
    let mut child_ctx = uc.uc_mcontext.gregs;
    child_ctx[libc::REG_RAX as usize] = 0;
    child_ctx[libc::REG_RSP as usize] = req.child_stack as libc::greg_t;
    child_ctx[libc::REG_RIP as usize] = info.call_addr as libc::greg_t;
    // `CLONE_SETTLS` gives the child its own thread pointer; without it the
    // child inherits the parent's, as the kernel does.
    let guest_fs = if req.flags & libc::CLONE_SETTLS as u64 != 0 {
        req.tls
    } else {
        t.guest_fs.get()
    };
    let clear_child_tid = (req.flags & libc::CLONE_CHILD_CLEARTID as u64 != 0
        && req.child_tid != 0)
        .then_some(req.child_tid);
    let inherited_mask = t.sig.mask.get();
    let req_flags = req.flags;
    let (parent_tid_word, child_tid_word) = (req.parent_tid, req.child_tid);

    // The parent must return the child's TID, but only the child can read its
    // own; hand it back over a one-shot channel and wait for it.
    let (tx, rx) = std::sync::mpsc::channel::<i32>();
    let spawned = std::thread::Builder::new()
        .name("chimera-guest".to_string())
        .spawn(move || {
            // Leaked, not stack-held: the `gs` base points at this for as long
            // as the thread runs guest code, and the trap handler dereferences
            // it from contexts that know nothing of this frame.
            let child: &'static Thread = Box::leak(Box::new(Thread::new(process, false)));
            child.guest_fs.set(guest_fs);
            child.clear_child_tid.set(clear_child_tid);
            // A new thread inherits its creator's signal mask.
            child.sig.mask.set(inherited_mask);

            // Replicate the kernel's set-TID writes before any guest code
            // runs: the kernel fills these at clone time, so the child must
            // observe its own TID from its first instruction. glibc points
            // them at the thread's control block and reads the value during
            // early thread setup and as the thread's identity for, among
            // other things, `pthread_rwlock` writer ownership. Both are
            // guest-controlled addresses, so the stores are best-effort — the
            // kernel's own `put_user` there is unchecked.
            if req_flags & libc::CLONE_PARENT_SETTID as u64 != 0 {
                copy_to_guest(parent_tid_word, &child.tid.get().to_ne_bytes());
            }
            if req_flags & libc::CLONE_CHILD_SETTID as u64 != 0 {
                copy_to_guest(child_tid_word, &child.tid.get().to_ne_bytes());
            }
            let _ = tx.send(child.tid.get());

            let code = match enter_thread(child, &child_ctx) {
                Ok(code) => code,
                Err(err) => {
                    eprintln!("chimera: guest thread failed: {err}");
                    127
                }
            };
            // A `fork` in this thread made it the only thread — and the
            // leader — of a whole new process (see `forward_fork`). This host
            // thread is all that process has, so its guest's status is the
            // process's, and simply returning would end the thread and leave
            // the process to exit 0 behind it.
            if child.is_leader.get() {
                std::process::exit(code);
            }
        });

    match spawned {
        // The handle is dropped: the host thread is detached and reclaims
        // itself when its closure returns, and the child is tracked by its
        // kernel TID rather than by a retained handle, which under thread
        // churn would only accumulate.
        Ok(_handle) => match rx.recv() {
            Ok(tid) => SyscallResult::Ok(tid as i64),
            Err(_) => SyscallResult::Error(libc::EAGAIN),
        },
        Err(_) => SyscallResult::Error(libc::EAGAIN),
    }
}

/// Bring a clone child up and run its guest, resuming from the register file
/// its parent's `clone` was trapped with. The counterpart of [`enter`] for
/// the leader, which starts from a fresh image instead.
fn enter_thread(thread: &'static Thread, gregs: &[libc::greg_t; 23]) -> Result<i32, Error> {
    set_this_thread(thread)?;
    install_altstack()?;
    thread.process.register(thread.tid.get());
    sync_host_mask(thread);

    unsafe { libc::getcontext(thread.exit_ctx.get()) };
    if let Some(code) = thread.exit.get() {
        return Ok(finish(thread, code));
    }

    if sud_on() != 0 {
        return Err(Error::last_os_error("enabling syscall user dispatch"));
    }
    unsafe {
        let mut ctx: libc::ucontext_t = mem::zeroed();
        libc::getcontext(&mut ctx);
        ctx.uc_mcontext.gregs = *gregs;
        set_fs(thread.guest_fs.get());
        libc::setcontext(&ctx);
        libc::abort();
    }
}

fn do_clone3(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t, info: &SigsysInfo) {
    // `clone_args` begins with flags, exit_signal at offset 32; read enough
    // to patch the shape and forward a private copy.
    const CLONE_ARGS_SIZE_MIN: usize = 64;
    const CLONE_ARGS_SIZE_MAX: usize = 4096;
    let size = call.args[1] as usize;
    if !(CLONE_ARGS_SIZE_MIN..=CLONE_ARGS_SIZE_MAX).contains(&size) {
        call.set_result(SyscallResult::Error(libc::EINVAL));
        return;
    }
    let mut buf = vec![0u8; size];
    if !copy_from_guest(call.args[0], &mut buf) {
        call.set_result(SyscallResult::Error(libc::EFAULT));
        return;
    }
    let mut flags = u64::from_ne_bytes(buf[..8].try_into().unwrap());
    if flags & libc::CLONE_THREAD as u64 != 0 {
        // The `clone_args` fields, in uapi order: flags, pidfd, child_tid,
        // parent_tid, exit_signal, stack, stack_size, tls. Unlike `clone`,
        // `stack` is the *lowest* address and `stack_size` its length, so the
        // child's stack pointer is their sum. This is the path modern glibc's
        // `pthread_create` takes.
        let field = |i: usize| u64::from_ne_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
        let result = spawn_thread(
            t,
            uc,
            info,
            CloneRequest {
                flags,
                child_stack: field(5).wrapping_add(field(6)),
                parent_tid: field(3),
                child_tid: field(2),
                tls: field(7),
            },
        );
        call.set_result(result);
        return;
    }
    let vm = flags & libc::CLONE_VM as u64 != 0;
    let vfork = flags & libc::CLONE_VFORK as u64 != 0;
    if vm && !vfork {
        call.set_result(SyscallResult::Error(libc::EPERM));
        return;
    }
    let mut child_stack = None;
    let is_spawn = vm && vfork;
    if is_spawn {
        flags &= !(libc::CLONE_VM as u64 | libc::CLONE_VFORK as u64);
        // See `do_clone`: the stack must not reach the kernel with a
        // fork-shaped clone. `clone_args` carries base and length, so the
        // child's stack pointer is their sum.
        let base = u64::from_ne_bytes(buf[40..48].try_into().unwrap());
        let len = u64::from_ne_bytes(buf[48..56].try_into().unwrap());
        child_stack = (base != 0).then(|| base.wrapping_add(len));
        buf[40..48].copy_from_slice(&0u64.to_ne_bytes());
        buf[48..56].copy_from_slice(&0u64.to_ne_bytes());
    }
    // `CLONE_CLEAR_SIGHAND` must not reach the host: there it would flush
    // Chimera's own handlers out of the child's slots, and the child's first
    // syscall would take `SIGSYS`'s default action instead of trapping. The
    // flag is emulated on the guest's virtual table in the child instead,
    // where it means what the guest asked for — caught handlers revert to
    // `SIG_DFL`, ignored ones stay ignored.
    let clear_sighand = flags & CLONE_CLEAR_SIGHAND != 0;
    flags &= !CLONE_CLEAR_SIGHAND;
    buf[..8].copy_from_slice(&flags.to_ne_bytes());
    let mut patched = SystemCall::new(call.number, [buf.as_ptr() as u64, size as u64, 0, 0, 0, 0]);
    if is_spawn {
        spawned(t, &mut patched, uc, child_stack);
    } else {
        forward_fork(t, &mut patched, uc, child_stack);
    }
    if clear_sighand && matches!(patched.result(), Some(SyscallResult::Ok(0))) {
        reset_guest_signals(t);
    }
    call.set_result(patched.result().expect("clone3 always sets a result"));
}

/// The runtime-owned `mmap`: resolve a virtualized fd like the translating
/// driver, and steer `NULL`-hint requests into the guest arena so fresh
/// guest pages — code the guest may write and jump to — stay below the
/// exempt floor. Explicitly placed requests forward untouched.
fn do_mmap(t: &Thread, call: &mut SystemCall) {
    let fd = call.args[4] as i32;
    if fd >= 0
        && let Some(host_fd) = t.process.handler.resolve_fd(fd)
    {
        call.args[4] = host_fd as u64;
    }
    let flags = call.args[3] as libc::c_int;
    let fixed = flags & (libc::MAP_FIXED | libc::MAP_FIXED_NOREPLACE) != 0;
    if call.args[0] != 0 || fixed {
        call.set_result(host_syscall(call));
        return;
    }
    let len = (call.args[1] + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    loop {
        let hint = t.process.bump.load(Ordering::Relaxed);
        if hint + len > GUEST_ARENA_CEILING {
            // Arena exhausted; let the kernel place it and accept that a
            // syscall from such a page would go unintercepted.
            call.set_result(host_syscall(call));
            return;
        }
        let placed = SystemCall::new(
            call.number,
            [
                hint,
                call.args[1],
                call.args[2],
                (flags | libc::MAP_FIXED_NOREPLACE) as u64,
                call.args[4],
                call.args[5],
            ],
        );
        let result = host_syscall(&placed);
        match result {
            SyscallResult::Error(libc::EEXIST) => {
                t.process
                    .bump
                    .store(hint + len.max(ARENA_IMAGE_GAP), Ordering::Relaxed);
            }
            _ => {
                if matches!(result, SyscallResult::Ok(_)) {
                    t.process.bump.store(hint + len, Ordering::Relaxed);
                }
                call.set_result(result);
                return;
            }
        }
    }
}
