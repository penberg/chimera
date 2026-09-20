//! The guest thread: its state, and how the host thread enters and leaves
//! guest code.
//!
//! The guest runs on the host thread that called [`super::execv`]. What the
//! trap handler needs from that thread — the two `fs` bases it switches
//! between, the frame its `exit` unwinds to — lives in
//! [`Thread`], reached through the `gs` base (see [`this_thread`]). The
//! thread enters guest code through [`enter`] and leaves it for good through
//! [`unwind`], which lands back in the entering frame to retire the guest in
//! [`finish`].

use std::{
    cell::{Cell, UnsafeCell},
    io, mem, ptr,
};

use crate::{Error, SyscallResult, SystemCall, sys::mmap::copy_to_guest};

use super::{super::syscall::host_syscall, Process, sud_off, sud_on};

const ARCH_SET_FS: u64 = 0x1002;
const ARCH_GET_FS: u64 = 0x1003;
const ARCH_SET_GS: u64 = 0x1001;

/// The guest thread. A `fork` child inherits its copy, contexts and all, so
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
    /// The process-wide state.
    pub process: Process,
    /// The runtime's `fs` base, restored on every trap entry so the handler's
    /// Rust code sees its own TLS; the guest owns the real `fs` while it runs
    /// (its TLS accesses are native).
    pub runtime_fs: u64,
    /// The guest's `fs` base, kept by the virtualized
    /// `arch_prctl(ARCH_SET_FS)` and reinstated when the handler returns.
    pub guest_fs: Cell<u64>,
    /// The write end of the pipe a `posix_spawn` child reports its `execve`
    /// outcome on; set only in such a child. See `clone::spawned`.
    pub spawn_report_fd: Cell<Option<i32>>,
    /// The errno of this spawn child's most recent failed `execve`, reported
    /// to the blocked parent only if the child exits without ever committing
    /// one.
    pub spawn_exec_errno: Cell<Option<i32>>,
    /// Set by the `exit`/`exit_group` intercept just before unwinding.
    exit: Cell<Option<i32>>,
    /// Where the unwind lands: the frame that entered the guest, captured
    /// with `getcontext`. Boxed so the `fpregs` self-pointer `getcontext`
    /// plants stays valid.
    exit_ctx: Box<UnsafeCell<libc::ucontext_t>>,
}

impl Thread {
    /// Build the thread's state for the calling host thread. `runtime_fs` is
    /// read here, so this must run *on* the thread it describes.
    pub fn new(process: Process) -> Self {
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
            spawn_report_fd: Cell::new(None),
            spawn_exec_errno: Cell::new(None),
            exit: Cell::new(None),
            exit_ctx: Box::new(UnsafeCell::new(unsafe { mem::zeroed() })),
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
}

/// The calling thread's [`Thread`], read out of the `gs` base.
///
/// The trap handler cannot use ordinary thread-local storage to find this.
/// It is entered with `fs` still holding the *guest's* thread pointer, so
/// every Rust thread-local — and `errno`, and the allocator's per-thread
/// state — would resolve against guest memory, and the runtime `fs` base it
/// needs to restore has to come from somewhere TLS-free. `gs` is that
/// somewhere: Linux x86-64 userspace leaves it unused (thread pointers live
/// in `fs`), so Chimera claims it, points it at the thread's state, and
/// reads the self-pointer parked at offset 0 with a single instruction that
/// touches no TLS at all.
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

/// Bring the host thread up as the guest thread and run the guest to
/// completion: publish the state for the trap handler, install the alternate
/// stack, arm dispatch, and enter guest code at `rip`/`rsp`. Returns the
/// guest's exit status when its `exit`/`exit_group` unwinds back here.
pub fn enter(thread: &'static Thread, rip: u64, rsp: u64) -> Result<i32, Error> {
    set_this_thread(thread)?;
    install_altstack()?;

    // The unwind target: `exit`/`exit_group` in the trap handler
    // `setcontext`s back here, and the second pass returns the code.
    unsafe { libc::getcontext(thread.exit_ctx.get()) };
    if let Some(code) = thread.exit.get() {
        return Ok(finish(code));
    }

    if sud_on() != 0 {
        return Err(Error::last_os_error("enabling syscall user dispatch"));
    }
    unsafe { enter_guest(rip, rsp) }
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

/// Leave guest code for good, with `code` as its status: jump to the frame
/// that entered the guest, which retires the run through [`finish`]. Never
/// returns, and runs no destructors — whatever the caller holds is
/// abandoned.
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

/// Retire the guest whose `exit` has unwound: disarm dispatch, so the
/// embedder's own syscalls no longer trap, and hand the status back.
fn finish(code: i32) -> i32 {
    sud_off();
    code
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

/// The trap handler needs a stack of its own: an `execve` intercept unmaps
/// the old guest stack — the very stack the handler would otherwise be
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
