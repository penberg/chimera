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
//! Proof-of-concept scope: single guest thread (`clone(CLONE_THREAD)` is
//! refused — a second native thread would need its own runtime TLS story),
//! `fork` and `posix_spawn` shapes forwarded, `execve` emulated in place,
//! native guest signals via a restorer stub in the exempt range. Guest
//! signal handlers that preempt [`on_sigsys`] mid-service run with the
//! runtime's `fs` base; a long-blocking forwarded syscall is the window.

use std::{
    cell::{Cell, RefCell, UnsafeCell},
    ffi::OsString,
    io, mem,
    os::fd::AsRawFd,
    path::Path,
    ptr,
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

/// The single guest task this backend drives (see the module comment for the
/// single-thread scope). A `fork` child inherits its copy, contexts and all,
/// so the child unwinds through its own `execv` frame exactly like the
/// parent.
struct Task {
    handler: Box<dyn SystemCalls>,
    /// The runtime's `fs` base, restored on every [`on_sigsys`] entry so the
    /// handler's Rust code sees its own TLS; the guest owns the real `fs`
    /// while it runs (its TLS accesses are native).
    runtime_fs: u64,
    /// The guest's `fs` base, kept by the virtualized
    /// `arch_prctl(ARCH_SET_FS)` and reinstated when the handler returns.
    guest_fs: Cell<u64>,
    /// Bump pointer into the guest arena.
    bump: Cell<u64>,
    /// Mappings owned by the current guest image outside the arena — `ET_EXEC`
    /// segments at their fixed low addresses and the initial stack — torn down
    /// together with the arena when an `execve` replaces the image.
    regions: RefCell<Vec<(u64, u64)>>,
    /// Set by the `exit`/`exit_group` intercept just before unwinding.
    exit: Cell<Option<i32>>,
    /// Where the unwind lands: [`execv`]'s frame, captured with `getcontext`
    /// before the guest was entered. Boxed so the `fpregs` self-pointer
    /// `getcontext` plants stays valid.
    exit_ctx: Box<UnsafeCell<libc::ucontext_t>>,
}

/// The task slot. One guest per process, accessed from the signal handler,
/// so a plain static rather than anything TLS-backed.
struct TaskSlot(UnsafeCell<Option<Task>>);
unsafe impl Sync for TaskSlot {}
static TASK: TaskSlot = TaskSlot(UnsafeCell::new(None));

fn task() -> &'static Task {
    unsafe { (*TASK.0.get()).as_ref().expect("SUD task installed") }
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

    let mut regions: Vec<(u64, u64)> = Vec::new();
    regions.extend(&main.regions);
    if let Some(interp) = &interp {
        regions.extend(&interp.regions);
    }
    regions.push((stack_start as u64, stack_len as u64));

    let runtime_fs = current_fs();
    install_altstack()?;
    install_sigsys_handler();
    unsafe {
        *TASK.0.get() = Some(Task {
            handler,
            runtime_fs,
            guest_fs: Cell::new(runtime_fs),
            bump: Cell::new(bump),
            regions: RefCell::new(regions),
            exit: Cell::new(None),
            exit_ctx: Box::new(UnsafeCell::new(mem::zeroed())),
        });
    }
    let t = task();

    // The unwind target: `exit`/`exit_group` in the signal handler
    // `setcontext`s back here, and the second pass returns the code.
    unsafe { libc::getcontext(t.exit_ctx.get()) };
    if let Some(code) = t.exit.get() {
        sud_off();
        return Ok(code);
    }

    if sud_on() != 0 {
        return Err(Error::last_os_error("enabling syscall user dispatch"));
    }
    unsafe { enter_guest(rip, rsp) }
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
/// Guest signals are blocked for the duration of [`on_sigsys`]. Left
/// deliverable, one arriving mid-service preempts the handler and runs guest
/// code in a context built for the runtime: the `fs` base is the runtime's,
/// so the guest's handler reads the wrong TLS, and — since the kernel blocks
/// `SIGSYS` inside its own handler — the first syscall that guest handler
/// makes takes the signal's default action and kills the process. Blocked
/// instead, the signal stays pending until the trap returns, and the kernel
/// delivers it against the guest's own register state. The cost is that a
/// guest signal no longer interrupts a *forwarded* blocking syscall: an
/// unhandled `SIGINT` arriving while the guest is parked in `read` waits for
/// the read to finish. Deferring delivery to a safepoint, the way the
/// translating backend does, is what a full implementation needs here.
///
/// `SIGSEGV`/`SIGBUS` stay unblocked: they are synchronous faults, and the
/// handler itself takes them when a guarded copy reads bad guest memory.
fn install_sigsys_handler() {
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = on_sigsys as *const () as usize;
        libc::sigfillset(&mut sa.sa_mask);
        libc::sigdelset(&mut sa.sa_mask, libc::SIGSEGV);
        libc::sigdelset(&mut sa.sa_mask, libc::SIGBUS);
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigaction(libc::SIGSYS, &sa, ptr::null_mut());
    }
}

/// The `SIGSYS` siginfo fields (`_sigsys` arm of the kernel union), which the
/// libc crate does not expose.
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

/// One trapped guest syscall. The first statements run with the *guest's*
/// `fs` base, so nothing before `set_fs` may touch TLS — no libc wrappers, no
/// `errno`, no thread locals.
extern "C" fn on_sigsys(_signo: libc::c_int, info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    let t = task();
    set_fs(t.runtime_fs);

    let uc = unsafe { &mut *(uc as *mut libc::ucontext_t) };
    let nr = unsafe { (*(info as *const SigsysInfo)).syscall } as u32 as u64;
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

    set_fs(t.guest_fs.get());
}

/// Drive one trapped syscall: the intercepts this backend owns, then the
/// embedder hooks — the same shape as the translating driver
/// (`crate::syscall`), minus everything that exists only to protect a code
/// cache.
fn dispatch(t: &Task, call: &mut SystemCall, uc: &mut libc::ucontext_t) {
    let handler = &*t.handler;
    handler.pre_syscall(call);

    let nr = call.number as i64;
    match nr {
        // One guest thread, so a thread-local exit and a group exit end the
        // same run. Unwind to `execv`'s frame; forwarding either would
        // terminate the embedder.
        libc::SYS_exit | libc::SYS_exit_group => {
            t.exit.set(Some(call.args[0] as i32));
            unsafe {
                libc::setcontext(t.exit_ctx.get());
                libc::abort();
            }
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
        libc::SYS_rt_sigaction => do_sigaction(call),
        libc::SYS_rt_sigprocmask => do_sigprocmask(call),
        libc::SYS_rt_sigsuspend => do_sigsuspend(call),
        libc::SYS_clone => do_clone(t, call),
        libc::SYS_clone3 => do_clone3(t, call),
        // A real vfork child shares the arena bump pointer and `guest_fs`
        // cells with a suspended parent; degrade to fork, whose
        // copy-on-write child owns its copies.
        libc::SYS_vfork | libc::SYS_fork => {
            let mut forked = SystemCall::new(libc::SYS_fork as u64, [0; 6]);
            forward_fork(t, &mut forked);
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
fn do_execve(t: &Task, call: &mut SystemCall, uc: &mut libc::ucontext_t) {
    match prepare_exec(call.number, &call.args, &*t.handler) {
        Ok(prepared) => match install_image(t, prepared, uc) {
            Ok(()) => call.set_result(SyscallResult::Ok(0)),
            // Past teardown there is no image to resume; end the run the way
            // a shell reports an exec that died mid-flight.
            Err(err) => {
                eprintln!("chimera: execve: {err}");
                t.exit.set(Some(127));
                unsafe {
                    libc::setcontext(t.exit_ctx.get());
                    libc::abort();
                }
            }
        },
        Err(err) => {
            let errno = exec_errno(&err).unwrap_or(libc::EIO);
            call.set_result(SyscallResult::Error(errno));
        }
    }
}

fn install_image(t: &Task, prepared: PreparedExec, uc: &mut libc::ucontext_t) -> Result<(), Error> {
    let PreparedExec {
        req,
        parsed,
        parsed_interp,
    } = prepared;
    let ExecRequest {
        path, argv, envp, ..
    } = req;

    t.handler.on_execve(&path);
    let mut keep = vec![parsed.as_raw_fd()];
    if let Some(interp) = &parsed_interp {
        keep.push(interp.as_raw_fd());
    }
    close_cloexec_fds(&keep)?;

    // Tear down the old guest address space: the tracked out-of-arena
    // regions, then the arena wholesale up to its watermark. Guest mappings
    // the kernel placed on its own (an explicit high hint) are the leak this
    // proof of concept accepts.
    for (start, len) in t.regions.borrow_mut().drain(..) {
        unsafe { libc::munmap(start as *mut libc::c_void, len as usize) };
    }
    let watermark = t.bump.get();
    if watermark > GUEST_ARENA_BASE {
        unsafe {
            libc::munmap(
                GUEST_ARENA_BASE as *mut libc::c_void,
                (watermark - GUEST_ARENA_BASE) as usize,
            )
        };
    }
    t.bump.set(GUEST_ARENA_BASE);

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
    t.bump.set(bump);
    let mut regions = t.regions.borrow_mut();
    regions.extend(&main.regions);
    if let Some(interp) = &interp {
        regions.extend(&interp.regions);
    }
    regions.push((stack_start as u64, stack_len as u64));

    reset_guest_signals();
    // A fresh image has no TLS yet; hand the handler epilogue a base that at
    // least keeps the host thread coherent until the new libc sets its own.
    t.guest_fs.set(t.runtime_fs);
    aim_context(&mut uc.uc_mcontext.gregs, rip, rsp);
    Ok(())
}

/// POSIX `execve` resets caught signals to their default disposition and
/// leaves ignored ones ignored. Skip the signals the runtime owns: `SIGSYS`
/// (the dispatch trap) and `SIGSEGV`/`SIGBUS` (the guarded-copy fixup).
fn reset_guest_signals() {
    for sig in 1..=libc::SIGRTMAX() {
        if matches!(
            sig,
            libc::SIGKILL | libc::SIGSTOP | libc::SIGSYS | libc::SIGSEGV | libc::SIGBUS
        ) {
            continue;
        }
        unsafe {
            let mut old: libc::sigaction = mem::zeroed();
            if libc::sigaction(sig, ptr::null(), &mut old) == 0
                && old.sa_sigaction != libc::SIG_DFL
                && old.sa_sigaction != libc::SIG_IGN
            {
                let mut dfl: libc::sigaction = mem::zeroed();
                dfl.sa_sigaction = libc::SIG_DFL;
                libc::sigemptyset(&mut dfl.sa_mask);
                libc::sigaction(sig, &dfl, ptr::null_mut());
            }
        }
    }
}

/// The bit `SIGSYS` occupies in a kernel `sigset_t` (bit `signo - 1`).
const SIGSYS_BIT: u64 = 1 << (libc::SIGSYS as u64 - 1);

/// `SIGSYS` must never be blocked: it *is* the dispatch trap, so a guest that
/// blocks it — `sigfillset` before a critical section is the common way —
/// would have its next syscall kill the process with the signal's default
/// action rather than trap into [`on_sigsys`]. Every mask the guest supplies
/// is filtered here before it reaches the kernel.
///
/// The guest's own view is left honest in one direction only: a mask it reads
/// back (`oldset`, `sigpending`) reports `SIGSYS` unblocked, because it is.
/// A guest that sets a full mask and asserts it reads back full sees the
/// difference; mirroring the intended mask, the way the translating backend
/// mirrors it, is the fix this proof of concept leaves undone.
fn read_guest_sigset(ptr: u64) -> Option<u64> {
    let mut raw = [0u8; 8];
    copy_from_guest(ptr, &mut raw).then(|| u64::from_ne_bytes(raw))
}

/// Forward a syscall whose argument at `arg` is a `sigset_t` pointer, with
/// `SIGSYS` cleared from the set (see [`SIGSYS_BIT`]). A null pointer, and a
/// set that does not block `SIGSYS`, forward untouched.
fn forward_with_filtered_mask(call: &mut SystemCall, arg: usize) {
    let ptr = call.args[arg];
    if ptr == 0 {
        call.set_result(host_syscall(call));
        return;
    }
    let Some(set) = read_guest_sigset(ptr) else {
        call.set_result(SyscallResult::Error(libc::EFAULT));
        return;
    };
    if set & SIGSYS_BIT == 0 {
        call.set_result(host_syscall(call));
        return;
    }
    let filtered = set & !SIGSYS_BIT;
    let mut args = call.args;
    args[arg] = &filtered as *const u64 as u64;
    call.set_result(host_syscall(&SystemCall::new(call.number, args)));
}

fn do_sigprocmask(call: &mut SystemCall) {
    // SIG_UNBLOCK and a query (null set) can only ever clear bits.
    if call.args[0] as i32 == libc::SIG_UNBLOCK {
        call.set_result(host_syscall(call));
        return;
    }
    forward_with_filtered_mask(call, 1);
}

fn do_sigsuspend(call: &mut SystemCall) {
    forward_with_filtered_mask(call, 0);
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

const SA_RESTORER: u64 = 0x0400_0000;

/// Forward a guest `rt_sigaction`, swapping the restorer for
/// [`chimera_sud_restorer`]: the guest's own trampoline sits below the
/// exempt floor, where its `rt_sigreturn` would trap and have no sane
/// emulation. The handler address itself stays the guest's — native delivery
/// into guest code is exactly this backend's model.
fn do_sigaction(call: &mut SystemCall) {
    let sig = call.args[0] as i32;
    // The runtime owns SIGSYS; pretend the installation succeeded so a guest
    // that sets a harness-style catch-all keeps running.
    if sig == libc::SIGSYS {
        call.set_result(SyscallResult::Ok(0));
        return;
    }
    let act_ptr = call.args[1];
    if act_ptr != 0 {
        let mut raw = [0u8; mem::size_of::<KernelSigaction>()];
        if !copy_from_guest(act_ptr, &mut raw) {
            call.set_result(SyscallResult::Error(libc::EFAULT));
            return;
        }
        let mut act: KernelSigaction = unsafe { mem::transmute(raw) };
        if act.handler != libc::SIG_DFL as u64 && act.handler != libc::SIG_IGN as u64 {
            act.flags |= SA_RESTORER;
            act.restorer = chimera_sud_restorer as *const () as u64;
        }
        // The mask applied for the duration of the handler: with `SIGSYS`
        // in it, the first syscall the guest's handler makes would be fatal
        // rather than dispatched (see `SIGSYS_BIT`).
        act.mask &= !SIGSYS_BIT;
        let patched = SystemCall::new(
            call.number,
            [
                call.args[0],
                &act as *const KernelSigaction as u64,
                call.args[2],
                call.args[3],
                0,
                0,
            ],
        );
        call.set_result(host_syscall(&patched));
    } else {
        call.set_result(host_syscall(call));
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
fn forward_fork(t: &Task, call: &mut SystemCall) {
    let hold = t.handler.lock_for_fork();
    let result = host_syscall(call);
    if let SyscallResult::Ok(0) = result {
        sud_on();
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
fn do_clone(t: &Task, call: &mut SystemCall) {
    let flags = call.args[0];
    let vm = flags & libc::CLONE_VM as u64 != 0;
    let vfork = flags & libc::CLONE_VFORK as u64 != 0;
    if vm && !vfork {
        call.set_result(SyscallResult::Error(libc::EPERM));
        return;
    }
    if vm && vfork {
        call.args[0] = flags & !(libc::CLONE_VM as u64 | libc::CLONE_VFORK as u64);
    }
    forward_fork(t, call);
}

fn do_clone3(t: &Task, call: &mut SystemCall) {
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
    let vm = flags & libc::CLONE_VM as u64 != 0;
    let vfork = flags & libc::CLONE_VFORK as u64 != 0;
    if vm && !vfork {
        call.set_result(SyscallResult::Error(libc::EPERM));
        return;
    }
    if vm && vfork {
        flags &= !(libc::CLONE_VM as u64 | libc::CLONE_VFORK as u64);
    }
    // `CLONE_CLEAR_SIGHAND` would reset the child's SIGSYS disposition and
    // the first child syscall would take the default action — death. The
    // dispositions stay; the spawn path resets what it needs at its exec.
    flags &= !CLONE_CLEAR_SIGHAND;
    buf[..8].copy_from_slice(&flags.to_ne_bytes());
    let mut patched = SystemCall::new(call.number, [buf.as_ptr() as u64, size as u64, 0, 0, 0, 0]);
    forward_fork(t, &mut patched);
    call.set_result(patched.result().expect("clone3 always sets a result"));
}

/// The runtime-owned `mmap`: resolve a virtualized fd like the translating
/// driver, and steer `NULL`-hint requests into the guest arena so fresh
/// guest pages — code the guest may write and jump to — stay below the
/// exempt floor. Explicitly placed requests forward untouched.
fn do_mmap(t: &Task, call: &mut SystemCall) {
    let fd = call.args[4] as i32;
    if fd >= 0
        && let Some(host_fd) = t.handler.resolve_fd(fd)
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
        let hint = t.bump.get();
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
                t.bump.set(hint + len.max(ARENA_IMAGE_GAP));
            }
            _ => {
                if matches!(result, SyscallResult::Ok(_)) {
                    t.bump.set(hint + len);
                }
                call.set_result(result);
                return;
            }
        }
    }
}
