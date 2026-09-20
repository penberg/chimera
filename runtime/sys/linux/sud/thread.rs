//! One guest thread: its state, and how a host thread enters and leaves
//! guest code.
//!
//! Every guest thread is a host thread. What belongs to one thread alone —
//! the two `fs` bases it switches between, its signal mask and deferred
//! signals, its alternate stack, the frame its `exit` unwinds to — lives in
//! [`Thread`], reached from the trap handler through the `gs` base (see
//! [`this_thread`]). A thread enters guest code through [`enter`] (the
//! leader, from a fresh image) or [`enter_thread`] (a clone child, from its
//! parent's registers), and leaves it for good through [`unwind`], which
//! lands back in the entering frame to retire the thread in [`finish`].
//!
//! A group-wide stop travels by signal. A guest thread runs natively, so
//! there is no safepoint for it to poll; the reserved stop signal is the
//! safepoint, taken on the spot in guest code and deferred to the trap's
//! safepoint inside the runtime (see [`on_stop`]).

use std::{
    cell::{Cell, UnsafeCell},
    io, mem, ptr,
    sync::{Arc, atomic::Ordering},
};

use crate::{Error, SyscallResult, SystemCall, sys::mmap::copy_to_guest};

use super::{
    super::syscall::host_syscall,
    EXEMPT_FLOOR, Process,
    exec::install_image,
    signal::{Signals, sig_bit, sync_host_mask},
    sud_off, sud_on,
};

const ARCH_SET_FS: u64 = 0x1002;
const ARCH_GET_FS: u64 = 0x1003;
const ARCH_SET_GS: u64 = 0x1001;

/// One guest thread. A `fork` child inherits its copy, contexts and all, so
/// the child unwinds through its own frame exactly like the parent.
#[repr(C)]
pub struct Thread {
    /// A pointer to this very struct, at offset 0 so the trap handler can
    /// load it with a single `gs:[0]`. Written by [`set_this_thread`] once
    /// the struct is at its final address — it cannot be filled in during
    /// construction, where the value would be the address of a local about to
    /// move. `Cell` is `repr(transparent)`, so the field is still a bare
    /// pointer at offset 0 as far as the load is concerned.
    self_ptr: Cell<*const Thread>,
    /// The state shared with every other thread of the guest process.
    pub process: Arc<Process>,
    /// The runtime's `fs` base, restored on every trap entry so the handler's
    /// Rust code sees its own TLS; the guest owns the real `fs` while it runs
    /// (its TLS accesses are native). Per thread, since each host thread has
    /// TLS of its own.
    pub runtime_fs: u64,
    /// The guest's `fs` base, kept by the virtualized
    /// `arch_prctl(ARCH_SET_FS)` and reinstated when the handler returns.
    pub guest_fs: Cell<u64>,
    /// This thread's kernel TID, which is also the TID the guest sees. A
    /// `Cell` because a fork child keeps the struct and takes a new TID.
    pub tid: Cell<i32>,
    /// Whether this is the thread group's leader — the one whose run
    /// returning ends the process, and the one an `exit_group` from a sibling
    /// hands the status to. A fork child is promoted to leader whichever
    /// thread forked, since it is its new process's only thread.
    pub is_leader: Cell<bool>,
    /// The `CLONE_CHILD_CLEARTID` word to zero and wake on exit, which is
    /// what a `pthread_join` blocks on.
    pub clear_child_tid: Cell<Option<u64>>,
    /// The write end of the pipe a `posix_spawn` child reports its `execve`
    /// outcome on; set only in such a child. See `clone::spawned`.
    pub spawn_report_fd: Cell<Option<i32>>,
    /// The errno of this spawn child's most recent failed `execve`, reported
    /// to the blocked parent only if the child exits without ever committing
    /// one.
    pub spawn_exec_errno: Cell<Option<i32>>,
    /// A group stop that arrived while this thread was inside the runtime and
    /// could not be taken where it landed; honored at the next safepoint.
    /// See [`on_stop`].
    pub stop_requested: Cell<bool>,
    /// Set by the `exit`/`exit_group` intercept just before unwinding.
    exit: Cell<Option<i32>>,
    /// Where the unwind lands: the frame that entered the guest, captured
    /// with `getcontext`. Boxed so the `fpregs` self-pointer `getcontext`
    /// plants stays valid.
    exit_ctx: Box<UnsafeCell<libc::ucontext_t>>,
    /// The guest's per-thread signal state: mask, deferred signals, and
    /// alternate stack. Dispositions are process-wide and live in
    /// [`Process::actions`].
    pub sig: Signals,
}

impl Thread {
    /// Build a thread's state for the calling host thread. `runtime_fs` and
    /// `tid` are read here, so this must run *on* the thread it describes.
    pub fn new(process: Arc<Process>, is_leader: bool) -> Self {
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
            tid: Cell::new(gettid()),
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

    /// The virtualized `arch_prctl`: the guest owns the real `fs` while it
    /// runs, but the handler must run on the runtime's, so the base is
    /// recorded here and installed on the way back to the guest. `gs` is
    /// Chimera's (see [`this_thread`]), so the guest's requests for it fall
    /// through to the embedder, which sees the same `EINVAL` the translating
    /// backend gives.
    pub fn arch_prctl(&self, call: &mut SystemCall) {
        match call.args[0] {
            ARCH_SET_FS => {
                self.guest_fs.set(call.args[1]);
                call.set_result(SyscallResult::Ok(0));
            }
            ARCH_GET_FS => {
                let base = self.guest_fs.get();
                if copy_to_guest(call.args[1], &base.to_ne_bytes()) {
                    call.set_result(SyscallResult::Ok(0));
                } else {
                    call.set_result(SyscallResult::Error(libc::EFAULT));
                }
            }
            _ => self.process.handler.do_syscall(call),
        }
    }

    /// Re-describe this thread as the only one of a fork child: a fork
    /// copies just the calling thread, so whatever it was in the parent, in
    /// the child it is the whole process — a new TID, the leader, and a
    /// roster of one.
    pub fn become_fork_child(&self) {
        self.tid.set(gettid());
        self.is_leader.set(true);
        self.process.reset_after_fork(self.tid.get());
    }
}

pub fn gettid() -> i32 {
    unsafe { libc::syscall(libc::SYS_gettid) as i32 }
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
/// TLS at all.
pub fn this_thread() -> &'static Thread {
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
    let base = thread as *const Thread;
    thread.self_ptr.set(base);
    match host_syscall(&SystemCall::new(
        libc::SYS_arch_prctl as u64,
        [ARCH_SET_GS, base as u64, 0, 0, 0, 0],
    )) {
        SyscallResult::Ok(_) => Ok(()),
        SyscallResult::Error(errno) => Err(Error::io(
            "arch_prctl(ARCH_SET_GS)",
            io::Error::from_raw_os_error(errno),
        )),
    }
}

/// Bring the leader up as a guest thread and run its guest to completion:
/// publish it for the trap handler, install the alternate stack, arm
/// dispatch, and enter guest code at `rip`/`rsp`. Returns the guest's exit
/// status when its `exit`/`exit_group` unwinds back here.
pub fn enter(thread: &'static Thread, rip: u64, rsp: u64) -> Result<i32, Error> {
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

/// Bring a clone child up and run its guest, resuming from the register file
/// its parent's `clone` was trapped with. The counterpart of [`enter`] for
/// the leader, which starts from a fresh image instead.
pub fn enter_thread(thread: &'static Thread, gregs: &[libc::greg_t; 23]) -> Result<i32, Error> {
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

/// Jump into the guest: capture a context, aim it at the guest entry with the
/// guest stack and the zeroed registers a fresh `execve` presents, and resume
/// it. `setcontext` never returns here — the run ends through the `exit_ctx`
/// unwind.
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
/// the state a native `execve` hands over (`rdx` doubles as the
/// atexit-function register, so a stale value would be registered and
/// called).
pub fn aim_context(gregs: &mut [libc::greg_t; 23], rip: u64, rsp: u64) {
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

/// Leave guest code for good on this thread, with `code` as its status: jump
/// to the frame that entered the guest, which retires the thread through
/// [`finish`]. Never returns, and runs no destructors — whatever the caller
/// holds is abandoned.
pub fn unwind(t: &Thread, code: i32) -> ! {
    // A spawn child ending without a committed exec is the failure case its
    // parent is still blocked on.
    super::clone::report_spawn(t, t.spawn_exec_errno.get().unwrap_or(0));
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
    thread.process.record_exit_status(code);
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

/// Whether the CPU and kernel expose `rdfsbase`/`wrfsbase` to userspace
/// (`CPUID.7.0:EBX.FSGSBASE[0]` plus `CR4.FSGSBASE`, which Linux sets when it
/// advertises the `fsgsbase` hwcap). Read once: the trap handler swaps the
/// `fs` base twice per dispatched syscall, and a pair of `arch_prctl` calls
/// there costs more than the trap itself.
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

pub fn current_fs() -> u64 {
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
pub fn set_fs(base: u64) {
    if fsgsbase_available() {
        unsafe { std::arch::asm!("wrfsbase {}", in(reg) base, options(nomem, nostack)) };
        return;
    }
    host_syscall(&SystemCall::new(
        libc::SYS_arch_prctl as u64,
        [ARCH_SET_FS, base, 0, 0, 0, 0],
    ));
}

/// The handlers need a stack of their own: an `execve` intercept unmaps the
/// old guest stack — the very stack the trap handler would otherwise be
/// running on.
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

/// Chimera's own alternate stack, as a `stack_t`. Installed once by
/// [`install_altstack`] and recorded so a frame Chimera builds can name it
/// for `rt_sigreturn` to restore.
pub fn chimera_altstack() -> libc::stack_t {
    let mut ss: libc::stack_t = unsafe { mem::zeroed() };
    unsafe { libc::sigaltstack(ptr::null(), &mut ss) };
    ss.ss_flags = 0;
    ss
}

/// The signal Chimera reserves to stop a guest thread. A guest thread
/// executes natively, so nothing polls a flag; the highest real-time signal
/// is the one least likely to collide with something the guest installs, and
/// the guest's own `rt_sigaction` for it is recorded but never honored.
pub fn stop_signal() -> i32 {
    libc::SIGRTMAX()
}

/// Install the handler for the reserved stop signal. `SA_ONSTACK` puts the
/// unwind on Chimera's alternate stack rather than whatever guest stack was
/// interrupted, and the mask is full because the handler never returns to
/// what it interrupted.
pub fn install_stop_handler() {
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

/// End this guest thread on the group's behalf: a sibling issued `exit_group`
/// or committed an `execve`.
///
/// Where the thread can be ended depends on what the signal interrupted, the
/// same split `signal::on_guest_signal` makes. Guest code holds nothing of
/// Chimera's, so a thread interrupted there unwinds on the spot — which is
/// the whole point of the mechanism, since a guest spinning without a syscall
/// in sight has no other way to be reached. Runtime code is different: this
/// handler never returns to what it interrupted, so a lock held at that
/// moment — one of the runtime's, or one inside the embedder's handler —
/// would be abandoned still locked and strand every thread that wants it
/// next. Those are flagged and taken at the safepoint in the trap handler,
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

/// Block the reserved stop signal for as long as this value lives.
///
/// The stop handler does not return to what it interrupted — it unwinds the
/// thread — so any lock held at the moment it lands is abandoned still
/// locked, and the next thread to want it waits forever. The `execve` install
/// path holds exactly such a lock while it tears the old image down, and it
/// is also the one path a concurrent exec might try to stop. Blocking the
/// signal makes the teardown uninterruptible; the stop is merely deferred,
/// and lands at the release.
pub struct StopBlocked(u64);

impl StopBlocked {
    pub fn new() -> Self {
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
