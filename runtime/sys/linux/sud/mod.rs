//! Native execution behind Linux syscall user dispatch.
//!
//! The second execution backend. The guest's instructions run unmodified on
//! the CPU, and interception happens at the syscall boundary only:
//! `prctl(PR_SET_SYSCALL_USER_DISPATCH)` (Linux 5.11) makes the kernel raise
//! `SIGSYS` for any syscall instruction executed outside a single exempt
//! address range, and the trap handler drives the same [`SystemCalls`] hooks
//! the translating dispatcher does. Chimera puts its own image — runtime
//! text, libc, vdso, and with them every runtime syscall site and the
//! signal-return trampoline — inside the exempt range and loads the guest
//! below it.
//!
//! The address-space contract: everything at or above [`EXEMPT_FLOOR`] is
//! exempt. The kernel links a PIE and its libraries above that line
//! (`ELF_ET_DYN_BASE` is `0x5555_5555_4000`), which [`execv`] verifies rather
//! than assumes. Guest images and guest `NULL`-hint mappings are placed in
//! an arena below the line, so guest code — a JIT's fresh pages included —
//! can never issue an unintercepted syscall. Guest *data* the kernel places
//! on its own (the initial stack, `brk` growth) may sit above the line; the
//! range exempts instruction addresses, and data is not fetched.
//!
//! What the backend trades away, compared to translation: the guest executes
//! natively, so a *hostile* guest can branch straight to a syscall
//! instruction inside the exempt range (Chimera's own libc) and bypass
//! interception — dispatch confines syscall *sites*, not control flow. The
//! translating backend has no such hole and remains the default; this one
//! suits observation and compatibility work (an strace, a VFS overlay) on
//! guests that are not adversarial, at native speed.
//!
//! The pieces: [`arena`] owns the guest half of the address space;
//! [`thread`] holds the guest thread's state and how the host thread enters
//! and leaves guest code; [`signal`] mirrors the guest's signal state and
//! delivers signals against the guest's own context; [`clone`] covers the
//! forwarded fork that must re-arm dispatch in the child; and [`exec`]
//! emulates `execve` in place. The guest is one thread: a thread-shaped
//! `clone` is refused, since a second native guest thread would race the
//! state here.

mod arena;
mod clone;
mod exec;
mod signal;
mod thread;

use std::{ffi::OsString, io, mem, path::Path, ptr};

use crate::{Error, SyscallResult, SystemCall, SystemCalls};

use super::{elf::parse_elf, exec::initial_request, fault, syscall::host_syscall};

use arena::Arena;
use signal::{ActionSlot, NSIG};
use thread::{Thread, set_fs, this_thread};

const PR_SET_SYSCALL_USER_DISPATCH: libc::c_int = 59;
const PR_SYS_DISPATCH_OFF: libc::c_ulong = 0;
const PR_SYS_DISPATCH_ON: libc::c_ulong = 1;

/// Everything at or above this address is exempt from dispatch: the runtime,
/// its libraries, and the vdso live here (see the module comment).
pub const EXEMPT_FLOOR: u64 = 0x5500_0000_0000;

/// The `siginfo` layout the kernel uses to describe a dispatch trap (the
/// `_sigsys` arm of its union), which the libc crate does not expose.
#[repr(C)]
pub struct SigsysInfo {
    si_signo: i32,
    si_errno: i32,
    si_code: i32,
    _pad: i32,
    /// The address just past the `syscall` instruction that trapped.
    pub call_addr: u64,
    syscall: i32,
    arch: u32,
}

/// The process-wide guest state: the embedder's handler, the signal
/// dispositions, and the guest arena.
pub struct Process {
    /// The embedder's system-call handler.
    pub handler: Box<dyn SystemCalls>,
    /// The guest's signal dispositions, indexed by signal number.
    actions: [ActionSlot; NSIG],
    pub arena: Arena,
}

impl Process {
    fn new(handler: Box<dyn SystemCalls>) -> Self {
        Self {
            handler,
            actions: std::array::from_fn(|_| ActionSlot::new()),
            arena: Arena::new(),
        }
    }

    /// The disposition slot for `signo`, which the caller has range-checked.
    pub fn action(&self, signo: i32) -> &ActionSlot {
        &self.actions[signo as usize]
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
    handler.on_execve(&req.path);
    let process = Process::new(handler);
    // The parsed images hold their files open, and they must be closed
    // before the guest runs: a guest `execve` sweeps every close-on-exec fd
    // it does not own, and a Rust-owned fd closed out from under its owner
    // aborts the process when the owner drops.
    let (rip, rsp) = {
        let parsed = parse_elf(&req.path)?;
        let parsed_interp = match &parsed.interp {
            Some(interp_path) => Some(parse_elf(interp_path)?),
            None => None,
        };
        process.arena.load_image(
            &parsed,
            parsed_interp.as_ref(),
            &req.argv,
            &req.envp,
            &req.raw,
        )?
    };

    install_sigsys_handler();

    // The `Thread` is pinned for the process's whole life, so the `gs` base
    // and the self-pointer both stay valid.
    let thread = Box::leak(Box::new(Thread::new(process)));
    thread::enter(thread, rip, rsp)
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

/// Install the dispatch trap handler.
///
/// Its `sa_mask` is empty, so guest signals stay deliverable for the duration
/// of the trap: that is what lets one interrupt a *forwarded* blocking
/// syscall, so an unhandled `SIGINT` arriving while the guest is parked in
/// `read` is felt rather than waited out. What arrives goes to
/// `signal::on_guest_signal`, which defers it to the safepoint at the tail
/// of [`on_sigsys`] rather than letting the guest's handler run against the
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

/// One trapped guest syscall. The first statements run with the *guest's*
/// `fs` base, so nothing before `set_fs` may touch TLS — no libc wrappers, no
/// `errno`, no thread locals.
///
/// The tail is the backend's safepoint. A guest signal that arrived while the
/// syscall was being serviced was deferred by `signal::on_guest_signal`,
/// because the context it interrupted was the runtime's; here the context
/// describes the guest again — the syscall has its result — so the deferred
/// signals can be delivered against it.
extern "C" fn on_sigsys(_signo: libc::c_int, info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    let t = this_thread();
    set_fs(t.runtime_fs);
    t.sig.in_runtime.set(true);

    let uc = unsafe { &mut *(uc as *mut libc::ucontext_t) };
    let info = unsafe { &*(info as *const SigsysInfo) };
    let nr = info.syscall as u32 as u64;
    t.sig.refresh_from(uc);
    let gregs = &uc.uc_mcontext.gregs;
    let args = [
        gregs[libc::REG_RDI as usize] as u64,
        gregs[libc::REG_RSI as usize] as u64,
        gregs[libc::REG_RDX as usize] as u64,
        gregs[libc::REG_R10 as usize] as u64,
        gregs[libc::REG_R8 as usize] as u64,
        gregs[libc::REG_R9 as usize] as u64,
    ];
    let mut call = SystemCall::new(nr, args);
    dispatch(t, &mut call, uc);
    uc.uc_mcontext.gregs[libc::REG_RAX as usize] = call.return_value() as libc::greg_t;

    // A syscall the kernel handed back as `EINTR` was interrupted by a signal
    // Chimera caught and deferred; whether the guest ever sees the `EINTR` is
    // its own `SA_RESTART` choice, applied before the frame is built so the
    // handler returns onto the restarted call.
    if call.return_value() == -(libc::EINTR as i64) && signal::restart_wanted(t) {
        signal::restart_syscall(uc, info, nr);
    }

    t.sig.in_runtime.set(false);
    signal::publish_host_mask(t, uc);
    let mask = t.sig.mask.get();
    signal::deliver_pending(t, uc, mask, mask);

    set_fs(t.guest_fs.get());
}

/// Drive one trapped syscall: the intercepts this backend owns, then the
/// embedder hooks — the same shape as the translating driver
/// (`crate::syscall`), minus everything that exists only to protect a code
/// cache.
fn dispatch(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t) {
    let handler = &*t.process.handler;
    handler.pre_syscall(call);

    match call.number as i64 {
        // One guest thread, so a thread-local exit and a group exit end the
        // same run. Unwind to the frame that entered the guest; forwarding
        // either would terminate the embedder.
        libc::SYS_exit | libc::SYS_exit_group => thread::unwind(t, call.args[0] as i32),
        libc::SYS_execve | libc::SYS_execveat => exec::do_execve(t, call, uc),
        libc::SYS_arch_prctl => t.arch_prctl(call),
        // The guest reconfiguring dispatch is the sandbox turning itself off.
        libc::SYS_prctl if call.args[0] == PR_SET_SYSCALL_USER_DISPATCH as u64 => {
            call.set_result(SyscallResult::Error(libc::EPERM));
        }
        libc::SYS_rt_sigaction => signal::do_sigaction(t, call),
        libc::SYS_rt_sigprocmask => signal::do_sigprocmask(t, call),
        libc::SYS_rt_sigsuspend => signal::do_sigsuspend(t, call, uc),
        libc::SYS_rt_sigpending => signal::do_sigpending(t, call),
        libc::SYS_sigaltstack => signal::do_sigaltstack(t, call),
        libc::SYS_clone => clone::do_clone(t, call, uc),
        libc::SYS_clone3 => clone::do_clone3(t, call, uc),
        // A real vfork child shares the arena bump pointer and `guest_fs`
        // cells with a suspended parent; degrade to fork, whose
        // copy-on-write child owns its copies.
        libc::SYS_vfork | libc::SYS_fork => {
            let mut forked = SystemCall::new(libc::SYS_fork as u64, [0; 6]);
            clone::forward_fork(t, &mut forked, uc, None);
            call.set_result(forked.result().expect("fork always sets a result"));
        }
        // The fd is resolved like the translating driver does; the arena
        // decides placement.
        libc::SYS_mmap => {
            let fd = call.args[4] as i32;
            if fd >= 0
                && let Some(host_fd) = handler.resolve_fd(fd)
            {
                call.args[4] = host_fd as u64;
            }
            t.process.arena.mmap(call);
        }
        // io_uring queues syscalls the kernel services without ever passing
        // them back through this driver.
        libc::SYS_io_uring_setup | libc::SYS_io_uring_enter | libc::SYS_io_uring_register => {
            call.set_result(SyscallResult::Error(libc::EPERM));
        }
        _ => handler.do_syscall(call),
    }

    handler.post_syscall(call);
}
