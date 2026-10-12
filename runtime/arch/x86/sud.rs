//! A [`Vcpu`] that runs guest code natively and traps its system calls with
//! Linux's syscall user dispatch (`PR_SET_SYSCALL_USER_DISPATCH`).
//!
//! Each guest thread arms dispatch with no exempt address range and with the
//! selector byte in its `ThreadState`. [`Vcpu::run`] loads the guest's
//! register file, sets the selector to `BLOCK`, and jumps to the guest; from
//! then on every system call the thread makes raises SIGSYS. Every signal
//! handler Chimera installs ends a native residency the same way: if the
//! thread was in guest code, the handler copies the interrupted context into
//! the register file, sets the selector back to `ALLOW`, and rewrites the
//! context so that `sigreturn` lands in `chimera_sud_exit`, which returns from
//! `run` with the reason in `exit_kind`. The kernel side never knows the guest
//! ran natively: it sees the same register file and the same exits the
//! translator produces.
//!
//! The selector stays `ALLOW` whenever Chimera's own code runs, including the
//! `rt_sigreturn` each handler returns through, so no part of Chimera has to
//! live in an exempt address range. Chimera's handlers run with every signal
//! blocked while guest code may be running, so one handler never mistakes
//! another handler's context for the guest's.

use std::{
    arch::{asm, global_asm},
    mem::{self, offset_of},
    ptr,
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{
    Error, process,
    sys::linux::signal,
    vcpu::{Exit, Vcpu},
};

use super::dispatch::{EXIT_KIND_BLOCK, EXIT_KIND_SYSCALL, EXIT_KIND_TRAP, ThreadState};

/// `prctl` option and modes (`<linux/prctl.h>`).
const PR_SET_SYSCALL_USER_DISPATCH: libc::c_int = 59;
const PR_SYS_DISPATCH_OFF: libc::c_ulong = 0;
const PR_SYS_DISPATCH_ON: libc::c_ulong = 1;

/// Selector values (`SYSCALL_DISPATCH_FILTER_*`).
const SELECTOR_ALLOW: u8 = 0;

/// `si_code` of a SIGSYS raised by syscall user dispatch.
const SYS_USER_DISPATCH: libc::c_int = 2;

/// `si_code` of a SIGTRAP raised by `int3`.
const SI_KERNEL: libc::c_int = 0x80;

/// `_fpx_sw_bytes::magic1`: the signal frame's FP area is an XSAVE image,
/// and `xstate_size` (at byte 16 of the software-reserved bytes, which start
/// at byte 464 of the legacy area) says how much of it is valid.
const FP_XSTATE_MAGIC1: u32 = 0x4650_5853;
const FP_SW_BYTES: usize = 464;

/// `exit_kind` of native guest code that branched to an address it cannot
/// fetch. Raised by the SIGSEGV/SIGBUS handler.
pub const EXIT_KIND_FETCH_FAULT: u64 = 3;

/// Size of the alternate stack the SIGSYS handler runs on.
const ALT_STACK_SIZE: usize = 64 * 1024;

/// Set while any thread has syscall user dispatch armed. Chimera's other
/// signal handlers then block every signal while they run (see the module
/// comment).
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// The signals whose host slots the backend takes over while it is active.
const HANDLED: [libc::c_int; 2] = [libc::SIGSYS, libc::SIGTRAP];

/// The attached vcpus, and the host dispositions their handlers replaced.
/// The first attach installs the handlers and the last detach restores what
/// was there, so a program that runs a sandbox gets its own handlers back
/// afterwards rather than handlers that reach a thread state that is gone.
struct Installed {
    attached: usize,
    saved: [libc::sigaction; HANDLED.len()],
}

// `sigaction` holds only function addresses and flags.
unsafe impl Send for Installed {}

static INSTALLED: Mutex<Option<Installed>> = Mutex::new(None);

/// [`INSTALLED`], held across a forwarded `fork` (see
/// `Process::lock_for_fork`).
pub struct ForkLock {
    _installed: MutexGuard<'static, Option<Installed>>,
}

pub fn lock_for_fork() -> ForkLock {
    ForkLock {
        _installed: INSTALLED.lock().unwrap(),
    }
}

/// Whether this process runs guest code natively.
pub fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Byte offset of `ThreadState::regs[idx]`.
const fn reg_off(idx: usize) -> usize {
    offset_of!(ThreadState, regs) + idx * 8
}

global_asm!(
    include_str!("sud.S"),
    TS_RAX            = const reg_off(0),
    TS_RBX            = const reg_off(1),
    TS_RCX            = const reg_off(2),
    TS_RDX            = const reg_off(3),
    TS_RSI            = const reg_off(4),
    TS_RDI            = const reg_off(5),
    TS_RBP            = const reg_off(6),
    TS_RSP            = const reg_off(7),
    TS_R8             = const reg_off(8),
    TS_R9             = const reg_off(9),
    TS_R10            = const reg_off(10),
    TS_R11            = const reg_off(11),
    TS_R12            = const reg_off(12),
    TS_R13            = const reg_off(13),
    TS_R14            = const reg_off(14),
    TS_R15            = const reg_off(15),
    TS_RIP            = const offset_of!(ThreadState, rip),
    TS_RFLAGS         = const offset_of!(ThreadState, rflags),
    TS_CHIMERA_RSP    = const offset_of!(ThreadState, chimera_rsp),
    TS_EXIT_KIND      = const offset_of!(ThreadState, exit_kind),
    TS_GUEST_FS       = const offset_of!(ThreadState, guest_fs_base),
    TS_CHIMERA_FS     = const offset_of!(ThreadState, chimera_fs_base),
    TS_EXIT_REQUESTED = const offset_of!(ThreadState, exit_requested),
    TS_FPSTATE        = const offset_of!(ThreadState, fpstate),
    TS_SUD_SELECTOR   = const offset_of!(ThreadState, sud_selector),
    TS_SUD_IN_GUEST   = const offset_of!(ThreadState, sud_in_guest),
    EXIT_KIND_BLOCK   = const EXIT_KIND_BLOCK,
);

unsafe extern "C" {
    fn chimera_sud_enter(ctx: *mut ThreadState);
    fn chimera_sud_enter_window();
    fn chimera_sud_enter_end();
    fn chimera_sud_exit();
}

/// A [`Vcpu`] that runs guest code natively under syscall user dispatch.
pub struct SyscallUserDispatch {
    /// The SIGSYS handler's alternate stack, mapped on first attach.
    alt_stack: *mut libc::c_void,
    /// The alternate stack the host thread had before the first attach,
    /// restored when the vcpu is dropped.
    prev_alt_stack: libc::stack_t,
    /// The host thread dispatch is armed on, or 0.
    armed_tid: libc::pid_t,
}

// The raw pointers name memory this vcpu owns.
unsafe impl Send for SyscallUserDispatch {}

impl SyscallUserDispatch {
    pub fn new() -> Self {
        Self {
            alt_stack: ptr::null_mut(),
            prev_alt_stack: unsafe { mem::zeroed() },
            armed_tid: 0,
        }
    }

    /// Put back the host thread's previous alternate stack and unmap ours.
    fn release_alt_stack(&mut self) {
        unsafe {
            libc::sigaltstack(&self.prev_alt_stack, ptr::null_mut());
            libc::munmap(self.alt_stack, ALT_STACK_SIZE);
        }
        self.alt_stack = ptr::null_mut();
    }
}

impl Vcpu for SyscallUserDispatch {
    fn attach(&mut self, state: &mut ThreadState) -> Result<(), Error> {
        if self.alt_stack.is_null() {
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
                return Err(Error::last_os_error("mmap(sigaltstack)"));
            }
            let ss = libc::stack_t {
                ss_sp: stack,
                ss_flags: 0,
                ss_size: ALT_STACK_SIZE,
            };
            if unsafe { libc::sigaltstack(&ss, &mut self.prev_alt_stack) } != 0 {
                let err = Error::last_os_error("sigaltstack");
                unsafe { libc::munmap(stack, ALT_STACK_SIZE) };
                return Err(err);
            }
            self.alt_stack = stack;
        }
        state.sud_selector = SELECTOR_ALLOW;
        let ret = unsafe {
            libc::prctl(
                PR_SET_SYSCALL_USER_DISPATCH,
                PR_SYS_DISPATCH_ON,
                0 as libc::c_ulong,
                0 as libc::c_ulong,
                &state.sud_selector as *const u8,
            )
        };
        if ret != 0 {
            let err = Error::last_os_error("prctl(PR_SET_SYSCALL_USER_DISPATCH)");
            if self.armed_tid == 0 {
                self.release_alt_stack();
            }
            return Err(err);
        }
        // The selector is still `ALLOW`, so no SIGSYS can arrive before the
        // handlers are in place.
        let tid = unsafe { libc::gettid() };
        if self.armed_tid == 0 {
            install_handlers();
        } else if self.armed_tid != tid {
            // A fork child attaching again: the parent's other threads, and
            // their vcpus, do not exist here.
            if let Some(installed) = INSTALLED.lock().unwrap().as_mut() {
                installed.attached = 1;
            }
        }
        self.armed_tid = tid;
        Ok(())
    }

    fn run(&mut self, state: &mut ThreadState) -> Exit {
        unsafe { chimera_sud_enter(state) };
        match state.exit_kind {
            EXIT_KIND_SYSCALL => Exit::Syscall,
            EXIT_KIND_TRAP => Exit::Breakpoint,
            EXIT_KIND_FETCH_FAULT => Exit::FetchFault,
            _ => Exit::Safepoint,
        }
    }
}

impl Drop for SyscallUserDispatch {
    fn drop(&mut self) {
        if self.armed_tid == 0 {
            return;
        }
        // Dispatch and the alternate stack belong to the host thread that
        // attached; on any other thread there is nothing to disarm.
        if self.armed_tid == unsafe { libc::gettid() } {
            unsafe {
                libc::prctl(
                    PR_SET_SYSCALL_USER_DISPATCH,
                    PR_SYS_DISPATCH_OFF,
                    0 as libc::c_ulong,
                    0 as libc::c_ulong,
                    0 as libc::c_ulong,
                );
            }
            self.release_alt_stack();
        }
        uninstall_handlers();
    }
}

/// Install the SIGSYS and SIGTRAP handlers when the first vcpu attaches,
/// and take their host slots away from the guest's dispositions.
///
/// SIGSYS and the interrupt signal are also kept out of the host mask. The
/// kernel forces the SIGSYS of every trapped system call, and a forced signal
/// that is blocked kills the process. Guest code running natively polls
/// nothing, so the interrupt signal is the only way to stop a thread that
/// never makes a system call, and a guest mask that blocks it would leave an
/// `execve` waiting for that thread forever. Its handler is installed again
/// here: while no vcpu was attached the slot was not reserved, so a guest
/// run by the translator may have replaced it. SIGTRAP can follow the guest's
/// mask: the only SIGTRAP forced on the guest is an `int3`'s, which kills a
/// guest that blocks it natively too, and any other blocked SIGTRAP stays
/// pending in the kernel for the guest's `signalfd`.
fn install_handlers() {
    let mut installed = INSTALLED.lock().unwrap();
    if let Some(installed) = installed.as_mut() {
        installed.attached += 1;
        return;
    }
    let mut saved: [libc::sigaction; HANDLED.len()] = unsafe { mem::zeroed() };
    ACTIVE.store(true, Ordering::Relaxed);
    signal::reserve_host_signal(libc::SIGSYS);
    signal::reserve_host_signal(process::interrupt_signal());
    process::install_interrupt_handler();
    for (i, (signo, handler)) in HANDLED
        .into_iter()
        .zip([
            sud_sigsys as *const () as usize,
            sud_sigtrap as *const () as usize,
        ])
        .enumerate()
    {
        signal::reserve_host_handler(signo);
        unsafe {
            let mut sa: libc::sigaction = mem::zeroed();
            sa.sa_sigaction = handler;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
            libc::sigfillset(&mut sa.sa_mask);
            libc::sigaction(signo, &sa, &mut saved[i]);
        }
    }
    *installed = Some(Installed { attached: 1, saved });
}

/// Undo [`install_handlers`] when the last vcpu detaches.
fn uninstall_handlers() {
    let mut installed = INSTALLED.lock().unwrap();
    let Some(state) = installed.as_mut() else {
        return;
    };
    state.attached -= 1;
    if state.attached > 0 {
        return;
    }
    for (signo, sa) in HANDLED.into_iter().zip(state.saved) {
        unsafe { libc::sigaction(signo, &sa, ptr::null_mut()) };
        signal::release_host_signal(signo);
    }
    signal::release_host_signal(process::interrupt_signal());
    ACTIVE.store(false, Ordering::Relaxed);
    *installed = None;
}

extern "C" fn sud_sigsys(signo: libc::c_int, info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    if unsafe { (*info).si_code } == SYS_USER_DISPATCH {
        if !preempt(uc, EXIT_KIND_SYSCALL) {
            fatal(b"chimera: system call trapped outside guest code\n");
        }
        return;
    }
    signal::chimera_sigcatch(signo, info, uc);
}

extern "C" fn sud_sigtrap(signo: libc::c_int, info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    if unsafe { (*info).si_code } == SI_KERNEL && preempt(uc, EXIT_KIND_TRAP) {
        return;
    }
    signal::chimera_sigcatch(signo, info, uc);
}

fn fatal(msg: &[u8]) -> ! {
    unsafe {
        libc::write(2, msg.as_ptr().cast(), msg.len());
        libc::abort();
    }
}

/// The calling thread's register file, if a signal handler running on it has
/// interrupted native guest code.
fn interrupted_guest() -> Option<*mut ThreadState> {
    let gs: u64;
    unsafe { asm!("rdgsbase {}", out(reg) gs, options(nomem, nostack, preserves_flags)) };
    let state = gs as *mut ThreadState;
    if state.is_null() || unsafe { (*state).sud_in_guest } == 0 {
        return None;
    }
    Some(state)
}

/// Leave native guest code from a signal handler without coming back to it:
/// restore Chimera's FS base and let system calls through, so the handler
/// can run ordinary runtime code. Returns the interrupted FS base.
unsafe fn leave_guest(state: *mut ThreadState) -> u64 {
    let guest_fs: u64;
    unsafe {
        asm!(
            "rdfsbase {g}",
            "wrfsbase {c}",
            g = out(reg) guest_fs,
            c = in(reg) (*state).chimera_fs_base,
            options(nostack, preserves_flags),
        );
        ptr::write_volatile(&raw mut (*state).sud_selector, SELECTOR_ALLOW);
        (*state).sud_in_guest = 0;
    }
    guest_fs
}

/// For a handler that is about to take the process down from a guest fault:
/// if it interrupted native guest code, return to runtime conditions so the
/// handler's own system calls go through.
pub fn abandon_guest() {
    if let Some(state) = interrupted_guest() {
        unsafe { leave_guest(state) };
    }
}

/// End a native residency from a signal handler: if the handler interrupted
/// native guest code, save the interrupted context as the guest's register
/// file, record `exit_kind`, and rewrite the context so that `sigreturn`
/// returns from [`Vcpu::run`]. Returns whether it did. Async-signal-safe;
/// the handler must run with every signal blocked.
pub fn preempt(ucontext: *mut libc::c_void, exit_kind: u64) -> bool {
    let Some(state) = interrupted_guest() else {
        return false;
    };
    unsafe {
        let guest_fs = leave_guest(state);
        let uc = &mut *(ucontext as *mut libc::ucontext_t);
        let gregs = &mut uc.uc_mcontext.gregs;
        let rip = gregs[libc::REG_RIP as usize] as u64;
        let window =
            chimera_sud_enter_window as *const () as u64..chimera_sud_enter_end as *const () as u64;
        if window.contains(&rip) {
            // The entry was interrupted before the guest ran: the register
            // file still describes the guest.
            (*state).exit_kind = EXIT_KIND_BLOCK;
        } else {
            let s = &raw mut *state;
            (*s).regs = [
                gregs[libc::REG_RAX as usize] as u64,
                gregs[libc::REG_RBX as usize] as u64,
                gregs[libc::REG_RCX as usize] as u64,
                gregs[libc::REG_RDX as usize] as u64,
                gregs[libc::REG_RSI as usize] as u64,
                gregs[libc::REG_RDI as usize] as u64,
                gregs[libc::REG_RBP as usize] as u64,
                gregs[libc::REG_RSP as usize] as u64,
                gregs[libc::REG_R8 as usize] as u64,
                gregs[libc::REG_R9 as usize] as u64,
                gregs[libc::REG_R10 as usize] as u64,
                gregs[libc::REG_R11 as usize] as u64,
                gregs[libc::REG_R12 as usize] as u64,
                gregs[libc::REG_R13 as usize] as u64,
                gregs[libc::REG_R14 as usize] as u64,
                gregs[libc::REG_R15 as usize] as u64,
            ];
            (*s).rip = rip;
            (*s).rflags = gregs[libc::REG_EFL as usize] as u64;
            (*s).guest_fs_base = guest_fs;
            save_fpstate(uc.uc_mcontext.fpregs as *const u8, s);
            (*s).exit_kind = exit_kind;
        }
        gregs[libc::REG_RIP as usize] = chimera_sud_exit as *const () as usize as libc::greg_t;
        gregs[libc::REG_RSP as usize] = (*state).chimera_rsp as libc::greg_t;
        // IF set, and DF and TF clear, as the System V ABI expects on entry
        // to Rust code.
        gregs[libc::REG_EFL as usize] = 0x202;
        // No FP state: the kernel resets the FPU on sigreturn, so Chimera
        // resumes with its own default MXCSR and x87 control word rather than
        // whatever the guest left in them.
        uc.uc_mcontext.fpregs = ptr::null_mut();
    }
    true
}

/// Copy the guest's FP/SIMD state out of a signal frame into the register
/// file's XSAVE area, in the standard format both use.
unsafe fn save_fpstate(frame: *const u8, state: *mut ThreadState) {
    if frame.is_null() {
        return;
    }
    unsafe {
        let dst = (&raw mut (*state).fpstate).cast::<u8>();
        let cap = mem::size_of_val(&(*state).fpstate);
        let magic = ptr::read_unaligned(frame.add(FP_SW_BYTES).cast::<u32>());
        if magic == FP_XSTATE_MAGIC1 {
            let size = ptr::read_unaligned(frame.add(FP_SW_BYTES + 16).cast::<u32>()) as usize;
            ptr::copy_nonoverlapping(frame, dst, size.min(cap));
        } else {
            // A legacy FXSAVE frame: keep the x87 and SSE state and mark
            // exactly those components present in the XSAVE header.
            ptr::copy_nonoverlapping(frame, dst, 512);
            ptr::write_bytes(dst.add(512), 0, 64);
            ptr::write_unaligned(dst.add(512).cast::<u64>(), 0b11);
        }
    }
}
