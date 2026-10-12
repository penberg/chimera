//! Guest threads on Darwin: the kernel's per-thread state and the loop that
//! drives a thread's [`Vcpu`].
//!
//! [`Thread::run`] is the loop: observe process-wide stops, deliver any
//! pending guest signal, answer the branches that never reach the translator
//! — the return sentinels and the libSystem [`Escapes`] — run the vcpu, and
//! handle the [`Exit`] it returns, until the guest issues BSD `exit` or its
//! `main` returns through the seeded link register.
//!
//! The host-side thread interrupt — the reserved signal that pulls a sibling
//! out of a parked syscall — lives here too, behind the neutral `sys::thread`
//! name; the Linux host supplies its own (`tgkill` on `SIGRTMAX`).

use std::sync::{Arc, Once, atomic::Ordering};

use crate::{
    Error, SystemCall,
    arch::dispatch::{
        CALL_RETURN_SENTINEL, RETURN_SENTINEL, SIGRETURN_SENTINEL, ThreadState, Translator, X0, X1,
        X2, X3, X4, X5, X6, X7, X16, read_tsd_base,
    },
    process::Process,
    sys::darwin::signal::Signals,
    vcpu::{Exit, Vcpu},
};

// === Thread interruption ===
//
// The reserved signal that pulls a guest thread out of a parked host syscall;
// see the module doc.

/// Signal Chimera reserves to interrupt a guest thread parked in a forwarded
/// syscall. Darwin has no real-time signals, so there is no collision-free
/// choice; `SIGURG` is the conventional runtime-preemption signal (the Go
/// runtime uses it for exactly this on macOS) — its default action is Ignore
/// and compute guests rarely touch it. A guest that installs its own `SIGURG`
/// handler never sees it: [`super::signal`] keeps this slot for Chimera, the
/// way it keeps `SIGSEGV`/`SIGBUS` for the fault handler.
pub fn reserved_signal() -> libc::c_int {
    libc::SIGURG
}

/// Darwin `SYS___pthread_kill`: deliver a signal to a thread by its Mach port.
const SYS_PTHREAD_KILL: libc::c_int = 328;

extern "C" fn interrupt_noop(_sig: libc::c_int) {}

static INSTALL_INTERRUPT: Once = Once::new();

/// Install the no-op handler for [`reserved_signal`] once per process. No
/// `SA_RESTART`, so a parked syscall surfaces `EINTR` rather than restarting.
pub fn install_interrupt_handler() {
    INSTALL_INTERRUPT.call_once(|| unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = interrupt_noop as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0;
        libc::sigaction(reserved_signal(), &sa, std::ptr::null_mut());
    });
}

/// Force the guest thread whose Mach port is `tid` to its run-loop boundary by
/// delivering [`reserved_signal`] to it. A thread parked in a blocking host
/// syscall returns `EINTR`; one executing translated code is already pulled out
/// by its armed safepoint slot, which the caller sets before calling here.
/// (`tid` holds the Mach thread port — `pthread_mach_thread_np` — that
/// `Thread::run` published in `ThreadState::tid`.)
pub fn interrupt(tid: i32) {
    if tid != 0 {
        unsafe {
            libc::syscall(SYS_PTHREAD_KILL, tid as u32, reserved_signal());
        }
    }
}

// === Guest threads ===

unsafe extern "C" {
    /// Declared here because `libc`'s binding is deprecated in favor of the
    /// `mach2` crate, which the runtime does not otherwise need.
    fn mach_thread_self() -> libc::mach_port_t;
}

/// The libSystem entry points the run loop answers itself instead of
/// translating, resolved once per process.
///
/// Each is a place where the guest would otherwise observe the *runtime's*
/// state, because the two share one already-initialized libSystem and dyld:
/// the process identity dyld cached at its own startup, and a JIT-protection
/// bit that belongs to the whole host thread. Chimera plays the loader for
/// the guest, so it answers these the way a loader would — the same role the
/// thread-local thunk escape plays for `dyld`'s TLV path.
#[derive(Clone, Copy)]
struct Escapes {
    tlv_thunk: u64,
    jit_write_protect: u64,
    executable_path: u64,
    argv: u64,
    argc: u64,
    progname: u64,
    tlv_finalize: u64,
    tlv_atexit: u64,
    cxa_thread_atexit: u64,
    cxa_atexit: u64,
    atexit: u64,
    dlopen: u64,
    dlsym: u64,
    dlclose: u64,
    malloc_zone_register: u64,
    malloc_zone_unregister: u64,
    malloc_default_zone: u64,
    malloc_get_all_zones: u64,
    analytics_send: u64,
    analytics_send_lazy: u64,
    analytics_send_event: u64,
    analytics_send_event_lazy: u64,
}

impl Escapes {
    /// Resolved once per process: the addresses are process-lifetime
    /// constants, and `dlsym` itself is unsafe on the paths that need them —
    /// a fork child can inherit dyld's API lock mid-acquisition, where the
    /// owner token names a parent thread the child does not have, and the
    /// contending wait spins forever. (A fork child never initializes this:
    /// the parent's `execv` did, single-threaded, before any guest ran.)
    fn resolve() -> Self {
        static RESOLVED: std::sync::OnceLock<Escapes> = std::sync::OnceLock::new();
        *RESOLVED.get_or_init(Self::resolve_uncached)
    }

    fn resolve_uncached() -> Self {
        Self {
            tlv_thunk: crate::sys::darwin::dyld::tlv_thunk_addr(),
            tlv_finalize: crate::sys::darwin::dyld::tlv_finalize_addr(),
            jit_write_protect: symbol(c"pthread_jit_write_protect_np"),
            executable_path: symbol(c"_NSGetExecutablePath"),
            argv: symbol(c"_NSGetArgv"),
            argc: symbol(c"_NSGetArgc"),
            progname: symbol(c"getprogname"),
            tlv_atexit: symbol(c"_tlv_atexit"),
            cxa_thread_atexit: symbol(c"__cxa_thread_atexit"),
            cxa_atexit: symbol(c"__cxa_atexit"),
            atexit: symbol(c"atexit"),
            dlopen: symbol(c"dlopen"),
            dlsym: symbol(c"dlsym"),
            dlclose: symbol(c"dlclose"),
            malloc_zone_register: symbol(c"malloc_zone_register"),
            malloc_zone_unregister: symbol(c"malloc_zone_unregister"),
            malloc_default_zone: symbol(c"malloc_default_zone"),
            malloc_get_all_zones: symbol(c"malloc_get_all_zones"),
            analytics_send: analytics_symbol(c"AnalyticsSendEvent"),
            analytics_send_lazy: analytics_symbol(c"AnalyticsSendEventLazy"),
            analytics_send_event: analytics_symbol(c"analytics_send_event"),
            analytics_send_event_lazy: analytics_symbol(c"analytics_send_event_lazy"),
        }
    }
}

/// Address of a CoreAnalytics export, or zero. The runtime does not link the
/// framework, so it is dlopened here; a guest that calls it (`ld` reports
/// usage telemetry) must be kept off it — CoreAnalytics refuses to run on
/// the child side of a fork with an `abort_with_payload`, and every spawned
/// guest is one (the in-process execve never resets libSystem's fork
/// state). Dropping a telemetry event is the faithful outcome a sandbox
/// wants anyway.
fn analytics_symbol(name: &core::ffi::CStr) -> u64 {
    const PATH: &core::ffi::CStr =
        c"/System/Library/PrivateFrameworks/CoreAnalytics.framework/Versions/A/CoreAnalytics";
    let handle = unsafe { libc::dlopen(PATH.as_ptr(), libc::RTLD_NOW) };
    if handle.is_null() {
        return 0;
    }
    unsafe { libc::dlsym(handle, name.as_ptr()) as u64 }
}

/// Address of a libSystem symbol, or zero if absent — a value no guest branch
/// can match, so a missing symbol simply disables its escape.
fn symbol(name: &core::ffi::CStr) -> u64 {
    const RTLD_DEFAULT: *mut core::ffi::c_void = (-2isize) as *mut core::ffi::c_void;
    unsafe { libc::dlsym(RTLD_DEFAULT, name.as_ptr()) as u64 }
}

/// `SIGTRAP`, raised on a guest `BRK`.
const SIGTRAP: u32 = 5;

/// How a guest run ended.
pub enum ExitReason {
    /// The guest exited with this status code (BSD `exit`, or `main` returning
    /// through a null link register with the code in `x0`).
    Exited(i32),
    /// A committed `execve` dissolved the thread group: the exec driver
    /// installs the published image and re-enters (see
    /// `crate::sys::darwin::exec::drive`).
    Execve,
}

/// A guest thread: its register file plus the shared [`Process`] it runs in.
pub struct Thread {
    pub state: Box<ThreadState>,
    /// The virtual CPU that executes this thread's guest code.
    vcpu: Box<dyn Vcpu>,
    process: Arc<Process>,
    /// Whether the run loop should keep iterating; cleared by BSD `exit`.
    pub running: bool,
    /// The status code the run returns once `running` is cleared.
    pub exit_code: i32,
    /// Whether this is the process's initial thread. The main thread's run
    /// returning ends the process; a `bsdthread_create` child is never main.
    is_main: bool,
    /// Set once this thread's guest-side teardown has begun. From that point
    /// its libpthread has released its thread-local storage, so Chimera must
    /// not run guest code on it — a thread-local access would touch storage
    /// the guest has already given back.
    pub tearing_down: bool,
    /// This thread's guest signal state, over the process-shared disposition
    /// table.
    signals: Signals,
    /// Thread-local destructors the guest registered (`_tlv_atexit`), newest
    /// last. Run as guest calls when this thread's run ends — the role dyld's
    /// `ThreadLocalVariables::finalizeList` plays for a native thread.
    tlv_dtors: Vec<(u64, u64)>,
    /// Set when the most recently serviced syscall was forwarded, restartable,
    /// and returned `EINTR`: `(resume pc after the svc, syscall number,
    /// original x0)`. Consumed at the next signal delivery to honor
    /// `SA_RESTART`; cleared after every syscall, so it reflects only the
    /// immediately preceding one. The original `x0` must be carried because on
    /// arm64 it is both first argument and result register, so the errno
    /// writeback destroys it.
    restart: Option<(u64, u64, u64)>,
}

impl Thread {
    /// Create the guest process's initial thread against a fresh [`Process`],
    /// entering at `pc` with stack `sp`. The caller seeds argument registers
    /// (Darwin's `main` takes `argc`/`argv`/`envp`/`apple` in `x0..x3`) through
    /// the public `state` before calling [`Thread::run`].
    pub fn new(process: Arc<Process>, pc: u64, sp: u64) -> Result<Self, Error> {
        let signals = Signals::new(Arc::clone(&process.sig_table));
        let mut state = ThreadState::new(pc, sp);
        state.pending_set = signals.pending_set_ptr() as u64;
        Ok(Self {
            state,
            vcpu: Box::new(Translator::new(Arc::clone(&process))),
            process,
            running: false,
            exit_code: 0,
            is_main: true,
            tearing_down: false,
            signals,
            restart: None,
            tlv_dtors: Vec::new(),
        })
    }

    /// Rebuild bookkeeping in the child of a forwarded `fork`. The caller is
    /// the child's only thread and its group leader, whatever it was in the
    /// parent — its run returning must end the process with the guest's
    /// status — so it is promoted to main and the copied [`Process`] group
    /// state is reset around it. The interrupt identity is re-read with the
    /// `mach_thread_self` trap: the port `pthread_mach_thread_np` returns is
    /// the parent thread's cached name, stale here until the guest's fork
    /// wrapper runs libSystem's atfork child handler after the trap returns.
    pub fn reset_after_fork(&mut self) {
        self.is_main = true;
        let port = unsafe { mach_thread_self() };
        self.state.tid.store(port as i32, Ordering::Release);
        self.process.reset_after_fork(&self.state);
        // The profiler thread, like every other sibling, did not survive the
        // fork; a child under CHIMERA_PROFILE=1 restarts its own.
        crate::sys::darwin::exec::start_profiler(&self.process);
        // Drop every inherited translation: a `MAP_JIT` page the parent wrote
        // can spuriously lose executability in the fork child (observed as a
        // SIGBUS on the first fetch of a parent-translated block, rare and
        // load-dependent), so the child re-derives its cache — pages the
        // child writes itself are its own and execute reliably. The armed
        // SMC page set is kept: those host page protections were inherited
        // with the address space and are still accurate.
        self.process.lock_addr_space().code.reset();
    }

    pub fn signals_mut(&mut self) -> &mut Signals {
        &mut self.signals
    }

    /// Guest `sigsuspend`, with the shared process supplying the group-stop
    /// flags the wait must honor. A method rather than a `Signals` call so the
    /// signal state and the process handle split borrows cleanly.
    pub fn sigsuspend(&mut self, mask: u64) -> crate::SyscallResult {
        self.signals.sigsuspend(mask, &self.process)
    }

    pub fn process(&self) -> &Process {
        &self.process
    }

    /// Lock and return the shared guest address space (the code cache and guest
    /// mappings). Held only across a resolve/translate, never across `dispatch`.
    pub fn addr_space(&self) -> crate::process::AddrSpaceGuard<'_> {
        self.process.lock_addr_space()
    }

    /// Set the entry registers for a freshly mapped image without touching the
    /// address space, for re-entry after an `execve` once the new image is
    /// mapped.
    pub fn enter(&mut self, pc: u64, sp: u64) {
        self.state.reset(pc, sp);
    }

    /// Run the guest from its current entry state. Returns when it issues BSD
    /// `exit` or its `main` returns to the seeded link register — neither is
    /// forwarded to the host kernel — or when a committed `execve` dissolves
    /// the group.
    pub fn run(&mut self) -> Result<ExitReason, Error> {
        // The main guest thread inherits this host thread's own TSD base as its
        // guest view — libSystem is shared and already initialized, so the main
        // thread's pthread identity genuinely is the host one. Spawned guest
        // threads arrive with the slot pointing at their guest pthread struct.
        if self.state.guest_tsd == 0 {
            self.state.guest_tsd = read_tsd_base();
        }
        // While translated code runs, the host sp is the guest stack; a fault
        // handler without an alternate stack could not deliver a guest
        // stack-overflow fault.
        crate::sys::darwin::fault::install_altstack();
        self.running = true;

        // Publish this thread's Mach port as its interrupt identity, so a
        // sibling's process-wide stop can `__pthread_kill` it out of a parked
        // syscall (see [`interrupt`]). Must be set before register_thread
        // exposes this thread to interrupt_others.
        unsafe {
            let port = libc::pthread_mach_thread_np(libc::pthread_self());
            self.state.tid.store(port as i32, Ordering::Release);
        }
        self.process.register_thread(&self.state);
        let _guard = ThreadGuard {
            process: Arc::clone(&self.process),
            state: &*self.state,
        };

        self.vcpu.attach(&mut self.state)?;
        let escapes = Escapes::resolve();

        while self.running {
            // A sibling requested a process-wide exit (`exit`): stop here at the
            // boundary with the group's status. This thread reaches the check
            // either directly or after the reserved interrupt signal pulled it
            // out of a parked forwarded syscall with EINTR.
            if self.process.is_exiting() {
                self.exit_code = self.process.group_exit_code();
                break;
            }
            // A sibling committed an `execve`: the thread group is dissolving,
            // the way Linux's `de_thread` kills every other thread before a
            // new image is installed. Stop at this boundary — a worker's stop
            // ends its host thread; the main thread hands the run over to the
            // exec driver below.
            if self.process.exec_pending() {
                break;
            }
            // Deliver one pending, unblocked guest signal at this block
            // boundary — the interrupted context is between blocks, so the
            // synthesized frame is clean.
            if let Some((signo, info)) = self.signals.take_deliverable() {
                let restart = self.restart.take();
                self.signals.deliver(&mut self.state, signo, &info, restart);
            }
            self.refresh_exit_requested();
            let pc = self.state.pc;
            // A top-level `ret` jumps to the sentinel LR we seeded at entry;
            // treat it as a clean exit with x0 as the status, the way crt's
            // start glue would have called `exit(retval)`.
            if pc == RETURN_SENTINEL {
                // A `main` that returned: its status is the process's, and
                // the handlers registered along the way still owe a run.
                self.exit_code = self.state.regs[X0] as i32;
                self.run_atexit_handlers();
                break;
            }
            // A guest signal handler returned: restore the interrupted
            // context from its frame — Chimera's `sigreturn`.
            if pc == SIGRETURN_SENTINEL {
                self.signals.restore(&mut self.state);
                continue;
            }
            // A branch to a null or non-canonical pointer (bits 48+ set — e.g.
            // a PAC-poisoned value) is a wild jump. Natively it faults; reflect
            // it as a segfault instead of letting the translator read unmapped
            // memory and crash the runtime itself — a sandbox must contain a
            // guest's bad branch.
            if pc == 0 || pc >> 48 != 0 {
                eprintln!("chimera: guest branched to a bad address {pc:#x}");
                crate::sys::darwin::fault::die(libc::SIGSEGV);
            }
            // A branch into libSystem the runtime answers itself (the TLV
            // thunk, `dlopen`, …) never reaches the translator; see
            // [`Escapes`].
            if self.escape(pc, &escapes)? {
                continue;
            }
            match self.vcpu.run(&mut self.state) {
                Exit::Syscall => self.handle_syscall(),
                Exit::Breakpoint => self.raise_trap(),
                // The guest branched into unmapped memory — a wild indirect
                // branch through a corrupted pointer, say. Natively the fetch
                // faults; raise the same SIGSEGV, which enters the guest's
                // handler or (default action) terminates the process
                // faithfully.
                Exit::FetchFault => self.raise_fetch_fault(),
                Exit::Safepoint => {}
            }
        }
        // The thread's stack outlives its run, so this is the right moment
        // for the destructors it owes.
        self.run_tlv_destructors();
        self.process.record_exit_status(self.exit_code);
        // A committed execve dissolves the group instead of ending the
        // process: hand the main thread's run to the exec driver, which waits
        // out the last sibling and installs the published image. (A main
        // thread that exited on its own parked in the policy's
        // `bsdthread_terminate` arm, whose `wait_for_others` also returns
        // early when a sibling commits an exec.)
        if self.is_main && self.process.exec_pending() {
            return Ok(ExitReason::Execve);
        }
        Ok(ExitReason::Exited(self.exit_code))
    }

    /// The guest branched to an address that cannot be fetched: raise the
    /// `SIGSEGV` the hardware would have delivered.
    fn raise_fetch_fault(&mut self) {
        let restart = self.restart.take();
        let info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        self.signals
            .deliver(&mut self.state, libc::SIGSEGV as u32, &info, restart);
    }

    /// Recompute the safepoint flag the translated loop-closing polls read. It
    /// is set exactly when a signal is pending and not blocked, so a fully
    /// linked guest loop is forced back into this run loop within one
    /// iteration to deliver it; once nothing is deliverable it clears, leaving
    /// warm loops poll-free at runtime. Clear first, then re-arm only if
    /// deliverable — never re-clearing — so a same-thread catcher that sets
    /// the flag between the clear and the recheck is not lost. The
    /// `compiler_fence` keeps the clear ordered before the recheck (signal
    /// delivery on this thread is itself a serialization point, so no CPU
    /// fence is needed). A group stop's own arming of the flag is not
    /// disturbed: `is_exiting`/`exec_pending` are checked at the top of every
    /// iteration, before this runs.
    fn refresh_exit_requested(&mut self) {
        self.state.exit_requested.store(0, Ordering::Relaxed);
        core::sync::atomic::compiler_fence(Ordering::SeqCst);
        if self.signals.has_deliverable() {
            self.state.exit_requested.store(1, Ordering::Relaxed);
        }
    }

    /// Service a guest branch that lands on one of the [`Escapes`]: perform
    /// its effect natively and return through the guest's link register,
    /// exactly as the real function would. Returns whether `pc` was one.
    ///
    /// Translating these instead would run Chimera's own libSystem as guest
    /// code, which for the identity queries would answer with the runtime's
    /// state and for the JIT toggle would strip execute permission from the
    /// code cache the guest is running out of.
    fn escape(&mut self, pc: u64, esc: &Escapes) -> Result<bool, Error> {
        let ts: *mut ThreadState = &mut *self.state;
        let ret = |ts: *mut ThreadState, value: u64| unsafe {
            (*ts).regs[X0] = value;
            (*ts).pc = (*ts).regs[30];
        };
        if pc == esc.tlv_thunk {
            // A first touch also owes this thread the image's thread-local
            // initializers — guest code, run before the slot is handed over,
            // the way dyld's `tlv_allocate_and_initialize` runs them.
            let (slot, init_funcs, install) =
                crate::sys::darwin::dyld::run_tlv_thunk(unsafe { (*ts).regs[X0] });
            // A freshly allocated block is published through the guest's own
            // `pthread_setspecific`, as a guest call: writing the guest's TSD
            // array directly would leave libpthread's per-thread bookkeeping
            // untouched, and its exit-time cleanup would then never reach this
            // key — which is the hook that runs the thread's destructors while
            // a joiner is still waiting (see `chimera_tlv_finalize`).
            if !self.tearing_down {
                if let Some((setspecific, key, block)) = install {
                    self.run_guest_call2(setspecific, key, block)?;
                }
                for func in init_funcs {
                    self.run_guest_call(func, 0)?;
                    if !self.running {
                        break;
                    }
                }
            }
            ret(ts, slot);
        } else if pc == esc.tlv_finalize {
            // libpthread's TSD cleanup, running as translated guest code, has
            // reached this thread's thread-local storage: the guest's exit
            // path is far enough along that the destructors are due, and not
            // so far that a joiner has been released. Run them here; the end
            // of the run loop keeps a backstop for a thread whose cleanup
            // never reaches this key.
            self.run_tlv_destructors();
            self.tearing_down = true;
            ret(ts, 0);
        } else if pc == esc.malloc_zone_register {
            // A guest allocator (rustc's jemalloc, say) registering its zone
            // would put *guest* function pointers on the shared zone list the
            // runtime's own malloc walks — the next native `malloc_size`
            // branches into a guest page and dies. The guest gets a virtual
            // zone list instead (see `sys::darwin::guest_zone_register`),
            // with libmalloc's own append/swap-remove semantics so an
            // allocator's "reorder until ours is the default" constructor
            // converges; the real list never changes.
            crate::sys::darwin::guest_zone_register(unsafe { (*ts).regs[X0] });
            ret(ts, 0);
        } else if pc == esc.malloc_zone_unregister {
            crate::sys::darwin::guest_zone_unregister(unsafe { (*ts).regs[X0] });
            ret(ts, 0);
        } else if pc == esc.malloc_default_zone {
            // Claimed unconditionally: an escape that declines a pc once
            // loses it — the block gets translated, linked, and never
            // consults the dispatcher again. The virtual list starts as a
            // snapshot of the real zones, so the answer is faithful before
            // any guest registration too.
            ret(ts, crate::sys::darwin::guest_zone_default());
        } else if pc == esc.malloc_get_all_zones {
            // `malloc_get_all_zones(task, reader, &addresses, &count)`,
            // answered from the virtual list via a runtime-owned handout
            // buffer — the guest reads runtime memory freely (one address
            // space).
            let (addresses, count) = unsafe { ((*ts).regs[X2], (*ts).regs[X3]) };
            let (array, n) = crate::sys::darwin::guest_zone_list();
            if !crate::sys::mmap::copy_to_guest(addresses, &array.to_ne_bytes())
                || !crate::sys::mmap::copy_to_guest(count, &n.to_ne_bytes())
            {
                return Err(Error::BadAccess(addresses));
            }
            ret(ts, 0);
        } else if pc == esc.analytics_send
            || pc == esc.analytics_send_lazy
            || pc == esc.analytics_send_event
            || pc == esc.analytics_send_event_lazy
        {
            ret(ts, 0);
        } else if pc == esc.jit_write_protect {
            ret(ts, unsafe { (*ts).regs[X0] });
        } else if pc == esc.executable_path {
            let (buf, size) = unsafe { ((*ts).regs[X0], (*ts).regs[X1]) };
            ret(
                ts,
                crate::sys::darwin::ns_get_executable_path(buf, size) as u64,
            );
        } else if pc == esc.argv {
            ret(ts, crate::sys::darwin::ns_get_argv());
        } else if pc == esc.argc {
            ret(ts, crate::sys::darwin::ns_get_argc());
        } else if pc == esc.progname {
            ret(ts, crate::sys::darwin::getprogname());
        } else if pc == esc.dlopen {
            // `dlopen(path, mode)`, serviced natively — the same host call
            // `load_dependent_dylibs` already makes for every linked dylib.
            // When the host dyld refuses the image — a zsh module's
            // flat-namespace reference to a symbol the *guest* executable
            // exports is invisible to it — Chimera's own linker takes over,
            // and owes the image its initializers as guest calls before the
            // guest resumes. On a double failure the guest gets NULL with
            // the host's `dlerror` state still describing its attempt.
            let (path_ptr, mode) = unsafe { ((*ts).regs[X0], (*ts).regs[X1]) };
            let path = if path_ptr == 0 {
                None
            } else {
                match crate::sys::mmap::read_guest_cstr(
                    path_ptr,
                    libc::PATH_MAX as usize,
                    libc::ENAMETOOLONG,
                ) {
                    Ok(bytes) => match std::ffi::CString::new(bytes) {
                        Ok(c) => Some(c),
                        Err(_) => {
                            ret(ts, 0);
                            return Ok(true);
                        }
                    },
                    Err(_) => {
                        ret(ts, 0);
                        return Ok(true);
                    }
                }
            };
            let cpath = path.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
            let handle = unsafe { libc::dlopen(cpath, mode as i32) };
            if !handle.is_null() || path.is_none() {
                ret(ts, handle as u64);
                return Ok(true);
            }
            let path = path.unwrap();
            let guest_path = std::path::Path::new(
                <std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(path.as_bytes()),
            );
            match crate::sys::darwin::dyld::dlopen_guest(guest_path) {
                Ok(module) => {
                    self.addr_space()
                        .add_region(module.region.0 as usize, module.region.1);
                    let frame = crate::sys::darwin::guest_frame();
                    for func in module.initializers {
                        self.run_guest_call4(func, frame)?;
                        if !self.running {
                            break;
                        }
                    }
                    ret(ts, module.handle);
                }
                Err(e) => {
                    if crate::trace::trace() {
                        eprintln!("chimera: dlopen {}: {e}", guest_path.display());
                    }
                    ret(ts, 0);
                }
            }
        } else if pc == esc.dlsym {
            // `dlsym(handle, name)`: a Chimera-loaded image answers from its
            // own symbol table; anything else is the host's to answer, with
            // one addendum — a global search the host misses may name a
            // guest image's export (`RTLD_DEFAULT` reaching the guest's own
            // symbols).
            let (handle, name_ptr) = unsafe { ((*ts).regs[X0], (*ts).regs[X1]) };
            let Ok(name_bytes) = crate::sys::mmap::read_guest_cstr(
                name_ptr,
                libc::PATH_MAX as usize,
                libc::ENAMETOOLONG,
            ) else {
                ret(ts, 0);
                return Ok(true);
            };
            let name = String::from_utf8_lossy(&name_bytes).into_owned();
            if crate::sys::darwin::dyld::is_guest_handle(handle) {
                ret(
                    ts,
                    crate::sys::darwin::dyld::dlsym_guest(handle, &name).unwrap_or(0),
                );
                return Ok(true);
            }
            let Ok(cname) = std::ffi::CString::new(name_bytes) else {
                ret(ts, 0);
                return Ok(true);
            };
            let mut addr =
                unsafe { libc::dlsym(handle as *mut libc::c_void, cname.as_ptr()) } as u64;
            if addr == 0 {
                addr = crate::sys::darwin::dyld::dlsym_guest_global(&name).unwrap_or(0);
            }
            ret(ts, addr);
        } else if pc == esc.dlclose {
            // A Chimera-loaded image is never unloaded — translated code and
            // resolved binds may point into it — so its `dlclose` just
            // reports success, the way dyld treats a leaked image.
            let handle = unsafe { (*ts).regs[X0] };
            if crate::sys::darwin::dyld::is_guest_handle(handle) {
                ret(ts, 0);
            } else {
                ret(ts, unsafe { libc::dlclose(handle as *mut libc::c_void) }
                    as u64);
            }
        } else if pc == esc.tlv_atexit || pc == esc.cxa_thread_atexit {
            // `_tlv_atexit(func, obj)` and `__cxa_thread_atexit(func, obj,
            // _)` register a thread-local destructor. Registering it with the
            // shared dyld would leave a *guest* function pointer on a list
            // dyld calls natively at thread exit — which faults, since guest
            // pages are never executable here, and would run guest code
            // outside the translator if they were. Keep the registration.
            let (func, arg) = unsafe { ((*ts).regs[X0], (*ts).regs[X1]) };
            self.tlv_dtors.push((func, arg));
            ret(ts, 0);
        } else if pc == esc.cxa_atexit || pc == esc.atexit {
            // `atexit(func)` and `__cxa_atexit(func, arg, dso)` register a
            // process-exit handler. Registering it with the shared libSystem
            // would hand a *guest* function pointer to `__cxa_finalize`,
            // which calls it natively at exit — off the translator, on a page
            // that is not executable — and would keep calling it after an
            // `execve` replaced the image that owned it, which POSIX says
            // discards these entirely. Keep the registration; `atexit`'s
            // callee takes no argument, so its unused `x1` is simply not
            // passed on.
            let (func, arg) = unsafe { ((*ts).regs[X0], (*ts).regs[X1]) };
            let arg = if pc == esc.atexit { 0 } else { arg };
            self.process.push_atexit(func, arg);
            ret(ts, 0);
        } else {
            return Ok(false);
        }
        Ok(true)
    }

    /// Run the thread-local destructors this thread registered, innermost
    /// first, as the guest calls they are — dyld would call them natively at
    /// thread exit, which is why the registration is intercepted (see
    /// [`Thread::escape`]). Each runs on the guest's own stack with the
    /// object address it was registered with.
    ///
    /// A destructor that registers another (a `thread_local` constructed
    /// during teardown) is honored: the list is drained from the end, so
    /// anything pushed while draining runs before the older entries.
    /// Run the process-exit handlers the guest registered, most recent
    /// first, as the guest calls they are — the role `__cxa_finalize` plays
    /// for a native process. Shared across the thread group, since `atexit`
    /// is process-wide and any thread's `exit` runs them.
    fn run_atexit_handlers(&mut self) {
        let was_running = std::mem::replace(&mut self.running, true);
        while let Some((func, arg)) = self.process.pop_atexit() {
            if self.run_guest_call(func, arg).is_err() {
                break;
            }
        }
        self.running = was_running;
    }

    /// A guest `BRK` left the cache with `ctx.pc` still at the trapping
    /// instruction. Raise `SIGTRAP`: it enters the guest's handler, or, with
    /// none, terminates the process with a faithful `SIGTRAP` status. A
    /// handler that returns re-executes the `BRK`, which is what the guest
    /// would see natively — AArch64 takes it as a fault, not a trap.
    fn raise_trap(&mut self) {
        let restart = self.restart.take();
        let info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        self.signals
            .deliver(&mut self.state, SIGTRAP, &info, restart);
    }

    pub fn run_tlv_destructors(&mut self) {
        // The run loop has stopped, so re-arm it for the duration: a guest
        // call needs a running thread to drive it.
        let was_running = std::mem::replace(&mut self.running, true);
        while let Some((func, arg)) = self.tlv_dtors.pop() {
            if self.process.is_exiting() || self.process.exec_pending() {
                break;
            }
            if self.run_guest_call(func, arg).is_err() {
                break;
            }
        }
        self.running = was_running;
    }

    /// Run one guest function to completion from the current guest context —
    /// the runtime's analogue of dyld calling a module initializer (used for
    /// the thread-local initializers a first TLV touch owes). The register
    /// file is saved and restored around the call, honoring the TLV slow
    /// path's contract to preserve everything; the callee runs on the guest
    /// thread's own stack, exactly where dyld would call it; and a nested
    /// resolve/dispatch/syscall loop drives it until it returns through
    /// [`CALL_RETURN_SENTINEL`]. A group stop or a guest exit abandons the
    /// call — the outer run loop observes the stop at its own boundary.
    pub fn run_guest_call(&mut self, func: u64, arg: u64) -> Result<u64, Error> {
        self.run_guest_call4(func, [arg, 0, 0, 0])
    }

    /// [`Thread::run_guest_call`] with a second argument register.
    fn run_guest_call2(&mut self, func: u64, arg0: u64, arg1: u64) -> Result<u64, Error> {
        self.run_guest_call4(func, [arg0, arg1, 0, 0])
    }

    /// [`Thread::run_guest_call`] with four argument registers (a static
    /// initializer's `(argc, argv, envp, apple)`). Returns the callee's
    /// result register, captured before the caller's register file is put
    /// back.
    pub fn run_guest_call4(&mut self, func: u64, args: [u64; 4]) -> Result<u64, Error> {
        // A guest call can arrive before this thread's first `run` — a static
        // initializer does — so the per-thread plumbing translated code needs
        // must be in place here. Both steps are idempotent for a thread
        // already inside `run`.
        self.vcpu.attach(&mut self.state)?;
        if self.state.guest_tsd == 0 {
            self.state.guest_tsd = read_tsd_base();
        }
        let saved_regs = self.state.regs;
        let saved_fpstate = self.state.fpstate;
        let saved_sp = self.state.sp;
        let saved_pc = self.state.pc;
        let saved_nzcv = self.state.nzcv;

        self.state.pc = func;
        self.state.regs[X0] = args[0];
        self.state.regs[X1] = args[1];
        self.state.regs[X2] = args[2];
        self.state.regs[X3] = args[3];
        self.state.regs[30] = CALL_RETURN_SENTINEL;
        // A guest call is the outermost frame of a fresh logical call stack,
        // so terminate the frame-pointer chain the way a thread entry point
        // does. Inheriting the interrupted context's `x29` leaves a chain
        // rooted in an already-unwound frame, and a native frame walker that
        // runs during the call — `backtrace` from a guest signal handler,
        // say — follows it past the top of the stack into the guard page.
        self.state.regs[29] = 0;

        let escapes = Escapes::resolve();

        while self.running {
            if self.process.is_exiting() || self.process.exec_pending() {
                break;
            }
            // A fault taken inside this call cannot be made to go away by
            // resuming: the faulting instruction simply runs again, which is
            // how a single bad access became a run of ~200,000 of them. End
            // the call instead and let the outer run loop deliver the fault
            // at a block boundary, where the guest is on its own stack.
            // Delivering it here was tried and is wrong — a guest call runs
            // on a borrowed stack, so building a signal frame on it corrupts
            // the call in progress.
            if self.signals.has_pending_fault() {
                break;
            }
            let pc = self.state.pc;
            if pc == CALL_RETURN_SENTINEL {
                break;
            }
            // A guest handler that ran during this call has returned. The
            // outer run loop recognises the sigreturn sentinel; without the
            // same arm here the sentinel would be handed to the translator
            // as a guest address, which faults, is reflected as SIGSEGV, and
            // kills the guest at a pc of `SIGRETURN_SENTINEL` — an exit with
            // no explanation in it.
            if pc == SIGRETURN_SENTINEL {
                self.signals.restore(&mut self.state);
                continue;
            }
            // An initializer touching another image's thread-locals recurses
            // through a nested first touch; the block installed before this
            // call keeps the recursion finite.
            if self.escape(pc, &escapes)? {
                continue;
            }
            if pc == 0 || pc >> 48 != 0 {
                eprintln!("chimera: guest branched to a bad address {pc:#x}");
                crate::sys::darwin::fault::die(libc::SIGSEGV);
            }
            match self.vcpu.run(&mut self.state) {
                Exit::Syscall => self.handle_syscall(),
                // A `BRK` inside a guest call: end the call and let the outer
                // run loop raise the trap, for the same reason a fault does —
                // a guest call runs on a borrowed stack, and a handler frame
                // built there would corrupt the call in progress.
                Exit::Breakpoint => break,
                Exit::FetchFault => {
                    self.raise_fetch_fault();
                    continue;
                }
                Exit::Safepoint => {}
            }
        }

        let result = self.state.regs[X0];
        self.state.regs = saved_regs;
        self.state.fpstate = saved_fpstate;
        self.state.sp = saved_sp;
        self.state.pc = saved_pc;
        self.state.nzcv = saved_nzcv;
        Ok(result)
    }

    /// Service the SVC that just exited the cache. BSD `exit` is intercepted
    /// here (forwarding it would terminate Chimera); everything else goes
    /// through the neutral syscall driver and the embedder handler, and the
    /// result is committed to the guest register file (x0 plus the NZCV carry
    /// bit) by the host writeback.
    fn handle_syscall(&mut self) {
        let number = self.state.regs[X16];
        let args = [
            self.state.regs[X0],
            self.state.regs[X1],
            self.state.regs[X2],
            self.state.regs[X3],
            self.state.regs[X4],
            self.state.regs[X5],
            self.state.regs[X6],
            self.state.regs[X7],
        ];
        // BSD `exit` (#1) terminates the whole process on Darwin — there is no
        // separate `exit_group`. Request the process-wide stop so every sibling
        // leaves its run loop at the next boundary (interrupting any parked in a
        // forwarded syscall), then stop this thread with the same status. A
        // single-threaded guest just stops; `interrupt_others` finds no sibling.
        if number == 1 {
            let code = args[0] as i32;
            // The guest's `exit(3)` runs its registered handlers before the
            // trap only if libSystem holds them, and it does not — Chimera
            // does (see `Thread::escape`). Run them here, as the guest calls
            // they are, while this thread's stack and image are still live.
            self.run_atexit_handlers();
            // Guest data still buffered in the shared stdio at the trap is data
            // a native exit discards; drop it before Chimera's own teardown can
            // flush it. Not done on the return-sentinel path, where no guest
            // `exit(3)` ran and the teardown flush is what emits a returning
            // `main`'s buffered output.
            crate::sys::darwin::purge_guest_stdio();
            self.process.request_exit_group(code, &self.state);
            self.exit_code = code;
            self.running = false;
            return;
        }
        let mut call = SystemCall::new(number, args);
        // Clone the `Arc` so the handler borrow comes from the local handle,
        // leaving `self` free to pass mutably to the driver.
        let process = Arc::clone(&self.process);
        crate::syscall::syscall(self, &mut call, process.handler.as_ref());
        crate::sys::write_syscall_result(&mut self.state, &call);

        // Record a restart candidate for SA_RESTART: a forwarded slow syscall
        // interrupted by a signal returns EINTR, and the dispatcher must be
        // able to re-issue it if the delivered handler asked to restart.
        // `state.pc` is the instruction after the `svc`. The never-restart
        // interfaces always surface EINTR, so they are excluded. Cleared on
        // any other syscall outcome.
        self.restart = match call.result() {
            Some(crate::SyscallResult::Error(libc::EINTR)) if !never_restart(number) => {
                Some((self.state.pc, number, args[0]))
            }
            _ => None,
        };
    }
}

/// Whether a syscall interrupted by a signal must always fail with `EINTR`,
/// never restarting even under `SA_RESTART`: the signal waits (`sigsuspend` —
/// Darwin's `pause(3)` is built on it), the multiplexing calls
/// (`select`/`poll`/`kevent*`), and the sleeps (`__semwait_signal`, which backs
/// `sleep`/`usleep`/`nanosleep`), each with its `_nocancel` twin. Any other
/// interrupted slow syscall is a restart candidate.
fn never_restart(number: u64) -> bool {
    const SYS_SELECT: u64 = 93;
    const SYS_SIGSUSPEND: u64 = 111;
    const SYS_POLL: u64 = 230;
    const SYS_SEMWAIT_SIGNAL: u64 = 334;
    const SYS_KEVENT: u64 = 363;
    const SYS_KEVENT64: u64 = 369;
    const SYS_KEVENT_QOS: u64 = 374;
    const SYS_KEVENT_ID: u64 = 375;
    const SYS_SELECT_NOCANCEL: u64 = 407;
    const SYS_SIGSUSPEND_NOCANCEL: u64 = 410;
    const SYS_POLL_NOCANCEL: u64 = 417;
    const SYS_SEMWAIT_SIGNAL_NOCANCEL: u64 = 423;
    matches!(
        number,
        SYS_SELECT
            | SYS_SIGSUSPEND
            | SYS_POLL
            | SYS_SEMWAIT_SIGNAL
            | SYS_KEVENT
            | SYS_KEVENT64
            | SYS_KEVENT_QOS
            | SYS_KEVENT_ID
            | SYS_SELECT_NOCANCEL
            | SYS_SIGSUSPEND_NOCANCEL
            | SYS_POLL_NOCANCEL
            | SYS_SEMWAIT_SIGNAL_NOCANCEL
    )
}

/// Removes a thread from the shared [`Process`] thread list when its run loop
/// ends, on every exit path.
struct ThreadGuard {
    process: Arc<Process>,
    state: *const ThreadState,
}

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        self.process.unregister_thread(self.state);
    }
}
