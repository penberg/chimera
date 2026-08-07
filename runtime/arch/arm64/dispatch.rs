//! The arm64 translate-execute loop and the guest register file it drives.
//!
//! [`ThreadState`] holds the guest register file. Translated code reaches it
//! through this host thread's pthread TSD slot: each block's prologue reads
//! `TPIDRRO_EL0`, masks it to the pthread struct, and loads the [`ThreadState`]
//! pointer from the ctx key's slot (see `super::translate::emit_load_ctx`). That
//! keeps every guest GPR — `x18` included — free to hold its own value.
//! [`Thread::run`] is the loop: resolve (translating on a miss) the block at the
//! guest PC, enter the cache through the trampoline, and service whatever caused
//! the exit (a block boundary or an SVC), until the guest issues BSD `exit` or
//! its `main` returns through a null link register.
//!
//! The cache is dispatcher-only (no block linking yet). Multiple guest threads
//! run concurrently: `bsdthread_create` spawns a host thread that runs this loop
//! over the shared [`Process`] (see [`Thread::spawn_bsdthread`]). Guest signals
//! and `execve` re-entry are added in later Darwin-port phases; the safepoint
//! (`exit_requested`) and `tid` fields the shared [`Process`] reads are present
//! but inert until then.

use std::sync::{
    Arc,
    atomic::{AtomicI32, AtomicU32, Ordering},
};

use crate::{Error, SystemCall, process::Process, sys::darwin::signal::Signals};

use super::trampoline::{dispatch, exit_block, exit_syscall_no_stack, exit_trap};

// Not exposed by the `libc` crate on macOS, but a standard pthread entry: set
// the thread's stack (lowest address + size) so libpthread does not allocate one.
unsafe extern "C" {
    fn pthread_attr_setstack(
        attr: *mut libc::pthread_attr_t,
        stackaddr: *mut libc::c_void,
        stacksize: libc::size_t,
    ) -> libc::c_int;
}

/// Reserved pthread TSD slot holding each host thread's [`ThreadState`]
/// pointer. Translated code reads it inline through `TPIDRRO_EL0 + slot*8` (see
/// `super::translate::emit_load_ctx`); each thread writes it at run start with
/// [`publish_ctx`]. Per-thread — correct for concurrent guest threads, which a
/// process-wide global could not serve without racing.
///
/// This is NOT a `pthread_key_create` key, and not one of the low reserved
/// slots (`0..=9`) libpthread uses for its own per-thread state (writing those
/// — slot 6, say — corrupts the pthread struct and deadlocks its mutexes). A
/// POSIX key lives in the array `_pthread_tsd_cleanup` walks on thread exit,
/// which zeroes every allocated slot in that range even for a key with no
/// destructor — so a guest `pthread_exit`, running that cleanup as translated
/// code, would wipe Chimera's ctx mid-teardown and the next block's prologue
/// would load null and fault.
///
/// Slot 112 is `__PTK_FRAMEWORK_GC_KEY9`, an Objective-C garbage-collection TSD
/// slot. ObjC GC was removed in macOS 10.15, so the slot is permanently dead:
/// no library or guest writes it. Being a framework slot reached by direct
/// index (not `pthread_key_create`), it is outside the range the POSIX-key
/// cleanup walks, so it survives the guest's own thread lifecycle — the same
/// property mimalloc relies on for its heap pointer, which it parks in the
/// sibling GC slot 89 (`OLDGC_KEY9`); Chimera takes 112 to avoid colliding with
/// it.
pub const CHIMERA_CTX_TSD_SLOT: u64 = 112;

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
    dispatch_apply: u64,
    dispatch_apply_f: u64,
    dispatch_async: u64,
    dispatch_async_f: u64,
    dispatch_group_async: u64,
    dispatch_group_async_f: u64,
    block_copy: u64,
    block_release: u64,
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
    stackaddr: u64,
    stacksize: u64,
    atfork: u64,
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
            dispatch_apply: symbol(c"dispatch_apply"),
            dispatch_apply_f: symbol(c"dispatch_apply_f"),
            dispatch_async: symbol(c"dispatch_async"),
            dispatch_async_f: symbol(c"dispatch_async_f"),
            dispatch_group_async: symbol(c"dispatch_group_async"),
            dispatch_group_async_f: symbol(c"dispatch_group_async_f"),
            block_copy: symbol(c"_Block_copy"),
            block_release: symbol(c"_Block_release"),
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
            stackaddr: symbol(c"pthread_get_stackaddr_np"),
            stacksize: symbol(c"pthread_get_stacksize_np"),
            atfork: symbol(c"pthread_atfork"),
            analytics_send: analytics_symbol(c"AnalyticsSendEvent"),
            analytics_send_lazy: analytics_symbol(c"AnalyticsSendEventLazy"),
            analytics_send_event: analytics_symbol(c"analytics_send_event"),
            analytics_send_event_lazy: analytics_symbol(c"analytics_send_event_lazy"),
        }
    }
}

unsafe extern "C" {
    /// Declared here because `libc`'s binding is deprecated in favor of the
    /// `mach2` crate, which the runtime does not otherwise need.
    fn mach_thread_self() -> libc::mach_port_t;
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

/// Publish `ctx` for this host thread in the reserved ctx TSD slot, by writing
/// it directly through `TPIDRRO_EL0` (the pthread self-pointer, low 3 bits the
/// CPU number). Called once at run start on each thread that runs a guest.
pub fn publish_ctx(ctx: *mut ThreadState) {
    unsafe {
        let base: u64;
        core::arch::asm!("mrs {b}, tpidrro_el0", b = out(reg) base, options(nomem, nostack, preserves_flags));
        let slot = ((base & !7) as *mut u64).add(CHIMERA_CTX_TSD_SLOT as usize);
        slot.write(ctx as u64);
    }
}

/// This host thread's published ctx, read from the reserved TSD slot the same
/// way translated code does. Async-signal-safe (a single `mrs` plus a load),
/// unlike `pthread_getspecific` — so the fault handler and signal catcher use
/// it. Returns null if nothing has been published on this thread.
pub fn current_ctx() -> *mut ThreadState {
    unsafe {
        let base: u64;
        core::arch::asm!("mrs {b}, tpidrro_el0", b = out(reg) base, options(nomem, nostack, preserves_flags));
        let slot = ((base & !7) as *const u64).add(CHIMERA_CTX_TSD_SLOT as usize);
        slot.read() as *mut ThreadState
    }
}

pub const EXIT_KIND_BLOCK: u64 = 0;
pub const EXIT_KIND_SYSCALL: u64 = 1;
/// A guest `BRK` exited the cache: the run loop raises `SIGTRAP`.
pub const EXIT_KIND_TRAP: u64 = 2;

/// `SIGTRAP`, raised on a guest `BRK`.
const SIGTRAP: u32 = 5;

/// Link register seeded at guest entry. A top-level return (the C `start` glue
/// calling `exit(main(...))`, or a `main` that just returns) jumps here, and the
/// run loop treats that as a clean exit.
///
/// It must be a fixed point of `xpaci`: a PAC-signed entry such as `main` opens
/// with `pacibsp` (a no-op here, so the LR stays this plain value) and closes
/// with `retaa`/`retab`, which the translator lowers to `xpaci x30` before the
/// branch. `xpaci` rewrites bits [63:47] to the sign-extension of bit 55, so a
/// sentinel in the non-canonical range (bit 55 set, low bits clear) comes back
/// transformed and the exit check misses it. Keeping the value in the low 47-bit
/// VA with bit 55 clear leaves `xpaci` a no-op. It is also odd, so no 4-byte
/// aligned instruction can occupy it — a real branch here is impossible, keeping
/// it distinguishable from a wild branch to a null or otherwise bad pointer.
pub const RETURN_SENTINEL: u64 = 0x0000_0000_DEAD_BEEF;

/// Link register set when delivering a guest signal handler. The handler's
/// return lands here and the run loop restores the interrupted context from
/// the signal frame — Chimera's `sigreturn`. Constrained like
/// [`RETURN_SENTINEL`]: an `xpaci` fixed point, and odd.
pub const SIGRETURN_SENTINEL: u64 = 0x0000_0000_DEAD_1EE7;

/// Link register set when the runtime calls a guest function on dyld's behalf
/// (a thread-local initializer at first TLV touch). The callee's return lands
/// here and [`Thread::run_guest_call`]'s nested loop ends. Constrained like
/// [`RETURN_SENTINEL`]: an `xpaci` fixed point, and odd.
pub const CALL_RETURN_SENTINEL: u64 = 0x0000_0000_DEAD_CA11;

/// Guest register indices used by the syscall path: the Darwin arm64 ABI puts
/// the syscall number in `x16` and its arguments in `x0..x7` (Mach traps such as
/// `mach_msg2_trap` use all eight).
pub const X0: usize = 0;
pub const X1: usize = 1;
pub const X2: usize = 2;
pub const X3: usize = 3;
pub const X4: usize = 4;
pub const X5: usize = 5;
pub const X6: usize = 6;
pub const X7: usize = 7;
pub const X16: usize = 16;

/// Guest register file plus the bookkeeping slots the trampolines and exit stubs
/// read. The byte layout is load-bearing: the offsets of `regs`, `fpstate`,
/// `sp`, `nzcv`, `chimera_sp`, `host_pc_target`, and `exit_kind` are handed to
/// `trampoline.S` by name through `global_asm!` const operands, so appending
/// fields (as `exit_requested`/`tid` are) never disturbs them. `#[repr(C,
/// align(16))]` keeps `fpstate` 16-byte aligned for the paired `ldp/stp q`.
#[repr(C, align(16))]
pub struct ThreadState {
    /// Guest GPRs x0..x30, each round-tripped by the trampolines. x18 is an
    /// ordinary guest register here (the context lives in a pthread TSD slot,
    /// not a register); x16 is the sole exception — the entry path leaves it
    /// holding the host-PC target and the block prologue reloads its guest value
    /// from this slot.
    pub regs: [u64; 31],
    /// Padding that brings `fpstate` to a 16-byte boundary (offset 256).
    pub _pad: u64,
    /// FPSIMD state: 32 Q-registers (512 bytes) then FPSR and FPCR (8 bytes).
    pub fpstate: [u8; 520],
    /// Guest stack pointer; set on exit, read on entry.
    pub sp: u64,
    /// Guest program counter; set on exit, read on entry.
    pub pc: u64,
    /// Guest NZCV condition flags (bits 31..28). The syscall path also sets the
    /// carry bit here to report an error, the way Darwin's libc reads it.
    pub nzcv: u64,
    /// Chimera's stack pointer, saved on entry and restored on exit.
    pub chimera_sp: u64,
    /// Host PC for the next entry; used by `dispatch` after it has loaded the
    /// guest GPRs and has no free register to hold the target.
    pub host_pc_target: u64,
    /// Why the last exit happened. Reset to `BLOCK` before each entry.
    pub exit_kind: u64,
    /// Safepoint slot a sibling's process-wide stop arms; inert until the
    /// threads phase wires the in-cache poll and the interrupt primitive.
    pub exit_requested: AtomicU32,
    /// The host identity a sibling's interrupt targets; inert until the threads
    /// phase.
    pub tid: AtomicI32,
    /// This thread's [`crate::sys::darwin::signal::PendingSet`], published so
    /// the host signal catcher can record a caught signal on the thread that
    /// received it without TLS or locking.
    pub pending_set: u64,
    /// The guest's `TPIDRRO_EL0` view: the TSD base this guest thread believes
    /// the kernel installed. The translator rewrites every guest
    /// `mrs <r>, TPIDRRO_EL0` to load this slot, so guest `pthread_self()` and
    /// TSD accesses resolve against the guest pthread struct rather than
    /// aliasing the host thread's TSD (the real register stays Chimera's).
    /// Seeded by `spawn_bsdthread` for a spawned thread, by `run` from the real
    /// register for the main thread, and updated by the `thread_set_tsd_base`
    /// intercept.
    pub guest_tsd: u64,
}

impl ThreadState {
    /// The guest PC this thread is at, named the same way on every backend so
    /// host-neutral code (the sampling profiler) can read it. Read
    /// volatilely: the profiler samples a running thread's slot, and wants
    /// whatever value is there now rather than one the compiler cached.
    pub fn guest_pc(&self) -> u64 {
        unsafe { std::ptr::read_volatile(&self.pc) }
    }

    fn new(pc: u64, sp: u64) -> Box<Self> {
        let mut state = Box::new(Self {
            regs: [0; 31],
            _pad: 0,
            fpstate: [0; 520],
            sp: 0,
            pc: 0,
            nzcv: 0,
            chimera_sp: 0,
            host_pc_target: 0,
            exit_kind: 0,
            exit_requested: AtomicU32::new(0),
            tid: AtomicI32::new(0),
            pending_set: 0,
            guest_tsd: 0,
        });
        state.reset(pc, sp);
        state
    }

    /// Seed the guest-visible entry state (PC, SP, cleared GPRs/flags/FP). The
    /// FPCR default of zero (round-to-nearest, all exceptions masked) is the
    /// value the kernel gives a fresh process, so the zeroed `fpstate` is
    /// correct — no equivalent of x86's MXCSR seeding is needed.
    fn reset(&mut self, pc: u64, sp: u64) {
        self.regs = [0; 31];
        self.fpstate = [0; 520];
        self.sp = sp;
        self.pc = pc;
        self.nzcv = 0;
        self.exit_kind = 0;
    }
}

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
    process: Arc<Process>,
    /// Whether the run loop should keep iterating; cleared by BSD `exit`.
    pub running: bool,
    /// The status code the run returns once `running` is cleared.
    pub exit_code: i32,
    /// Whether this is the process's initial thread. A single-threaded bring-up
    /// has only the main thread; the threads phase adds `clone`-style children.
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
    /// `bsdthread_terminate` arguments for the real syscall `guest_thread_entry`
    /// issues as the host thread's last act. The policy has already performed
    /// the guest-visible death effects (stack munmap, joiner wake) in guest
    /// program order, so only the Mach port slot is populated — the deferred
    /// syscall releases the port and terminates the host thread, nothing more.
    pub terminate_args: Option<[u64; 4]>,
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
            process,
            running: false,
            exit_code: 0,
            is_main: true,
            tearing_down: false,
            signals,
            restart: None,
            tlv_dtors: Vec::new(),
            terminate_args: None,
        })
    }

    pub fn is_main(&self) -> bool {
        self.is_main
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
        crate::sys::darwin::callback::mark_fork_child();
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

    /// Spawn a guest thread for `bsdthread_create`: run `thread_start` (a shared-
    /// libpthread address) on a fresh host thread with the kernel's thread-start
    /// register convention, over this thread's shared [`Process`].
    ///
    /// The host thread is created with a stack Chimera allocates itself (via
    /// `pthread_attr_setstack`). This is load-bearing: Chimera and the guest
    /// share one libpthread, and if libpthread allocated the host thread's stack
    /// from its own cache it would reclaim the *guest* thread's stack region
    /// (allocated outside libpthread's tracking), unmapping it out from under the
    /// running guest thread.
    pub fn spawn_bsdthread(
        &self,
        func: u64,
        arg: u64,
        stack: u64,
        pthread: u64,
        flags: u64,
        thread_start: u64,
    ) -> i64 {
        let mut state = ThreadState::new(thread_start, stack);
        state.regs[X0] = pthread;
        state.regs[X2] = func;
        state.regs[X3] = arg;
        state.regs[X4] = stack;
        // The kernel ORs bit 28 into the flags when it starts the thread (the
        // value libpthread passed to `bsdthread_create` does not have it); with
        // it clear, `_pthread_start` takes an abort (`brk`) path. Match that.
        state.regs[X5] = flags | 0x1000_0000;
        // The kernel points the new thread's TSD base at the pthread struct's
        // TSD array before it runs (`_pthread_start` asserts it). The array
        // offset within the struct is libpthread's, not ours — recover it from
        // this (host) thread, where base and struct are both readable.
        state.guest_tsd = pthread.wrapping_add(host_tsd_offset());

        const HOST_STACK_SIZE: usize = 1 << 20; // 1 MiB for the Rust run loop
        let host_stack = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                HOST_STACK_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if host_stack == libc::MAP_FAILED {
            return -(libc::EAGAIN as i64);
        }

        let report = Arc::new(crate::sys::darwin::handoff::Handoff::new());
        let ctx = Box::new(GuestThreadCtx {
            process: Arc::clone(&self.process),
            state,
            report: Arc::clone(&report),
        });

        let mut attr: libc::pthread_attr_t = unsafe { std::mem::zeroed() };
        let mut tid: libc::pthread_t = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            libc::pthread_attr_init(&mut attr);
            pthread_attr_setstack(&mut attr, host_stack, HOST_STACK_SIZE);
            libc::pthread_attr_setdetachstate(&mut attr, libc::PTHREAD_CREATE_DETACHED);
            let rc = libc::pthread_create(
                &mut tid,
                &attr,
                guest_thread_entry,
                Box::into_raw(ctx) as *mut libc::c_void,
            );
            libc::pthread_attr_destroy(&mut attr);
            rc
        };
        if rc != 0 {
            unsafe { libc::munmap(host_stack, HOST_STACK_SIZE) };
            return -(libc::EAGAIN as i64);
        }
        report.recv() as i64
    }

    /// Build a non-main thread from a prepared register state, sharing `process`.
    fn from_state(process: Arc<Process>, mut state: Box<ThreadState>) -> Self {
        let signals = Signals::new(Arc::clone(&process.sig_table));
        state.pending_set = signals.pending_set_ptr() as u64;
        Self {
            state,
            process,
            running: false,
            exit_code: 0,
            is_main: false,
            tearing_down: false,
            signals,
            restart: None,
            tlv_dtors: Vec::new(),
            terminate_args: None,
        }
    }

    /// Lock and return the shared guest address space (the code cache and guest
    /// mappings). Held only across a resolve/translate, never across `dispatch`.
    pub fn addr_space(&self) -> crate::process::AddrSpaceGuard<'_> {
        self.process.lock_addr_space()
    }

    /// Set the entry registers for a freshly mapped image without touching the
    /// address space. For re-entry after an `execve` once the new image is
    /// mapped (the exec path arrives in a later phase).
    pub fn enter(&mut self, pc: u64, sp: u64) {
        self.state.reset(pc, sp);
    }

    /// Run the guest from its current entry state. Returns when it issues BSD
    /// `exit` or its `main` returns to a null link register — neither is
    /// forwarded to the host kernel.
    pub fn run(&mut self) -> Result<ExitReason, Error> {
        let ts_ptr: *mut ThreadState = &mut *self.state;
        // Publish this thread's ctx in its reserved TSD slot, so translated code
        // reaches the right ThreadState per host thread (via TPIDRRO_EL0; see
        // translate::emit_load_ctx). Per-thread, so concurrent guest threads do
        // not race the way they would on the global above.
        publish_ctx(ts_ptr);
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
        // syscall (see crate::sys::thread::interrupt). Must be set before
        // register_thread exposes this thread to interrupt_others.
        unsafe {
            let port = libc::pthread_mach_thread_np(libc::pthread_self());
            self.state.tid.store(port as i32, Ordering::Release);
        }
        self.process.register_thread(&self.state);
        let _guard = ThreadGuard {
            process: Arc::clone(&self.process),
            state: ts_ptr as *const ThreadState,
        };

        let block_exit = exit_block as *const () as u64;
        // SVC sites exit through the no-stack tail, which honors the kernel's
        // "syscalls don't touch the user stack" contract.
        let syscall_exit = exit_syscall_no_stack as *const () as u64;
        // No BRK/undefined-instruction terminator yet; the translator ignores
        // this trampoline, so any value is harmless.
        let trap_exit = exit_trap as *const () as u64;

        // `CHIMERA_TRACE` logs the guest PC entering each block — the last line
        // before a crash localizes the faulting block during bring-up. Sampled
        // once at startup (crate::trace) so this never calls getenv mid-run.
        let trace = crate::trace::trace();

        // The guest's thread-local-variable descriptors call Chimera's native
        // `chimera_tlv_get_addr` to fetch a slot (`blr desc->thunk`). That thunk
        // is Chimera's own Rust and must run natively — translating it as guest
        // code returns a bogus slot pointer. Recognize a branch to it here.
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
                let state = unsafe { &mut *ts_ptr };
                self.signals.deliver(state, signo, &info, restart);
            }
            self.refresh_exit_requested();
            let pc = unsafe { (*ts_ptr).pc };
            // A top-level `ret` jumps to the sentinel LR we seeded at entry;
            // treat it as a clean exit with x0 as the status, the way crt's
            // start glue would have called `exit(retval)`.
            if pc == RETURN_SENTINEL {
                // A `main` that returned: its status is the process's, and
                // the handlers registered along the way still owe a run.
                self.exit_code = unsafe { (*ts_ptr).regs[X0] } as i32;
                self.run_atexit_handlers();
                break;
            }
            // A guest signal handler returned: restore the interrupted
            // context from its frame — Chimera's `sigreturn`.
            if pc == SIGRETURN_SENTINEL {
                let state = unsafe { &mut *ts_ptr };
                self.signals.restore(state);
                continue;
            }
            // A branch to a null or non-canonical pointer (bits 48+ set — e.g. a
            // PAC-poisoned value) is a wild jump. Natively it faults; until
            // Darwin delivers guest signals (Phase 4), reflect it as a segfault
            // instead of letting the translator read unmapped memory and crash
            // the runtime itself — a sandbox must contain a guest's bad branch.
            if pc == 0 || pc >> 48 != 0 {
                eprintln!("chimera: guest branched to a bad address {pc:#x}");
                crate::sys::darwin::fault::die(libc::SIGSEGV);
            }
            // Native TLV thunk escape: run the slot lookup natively for the
            // guest and return through its link register with the slot in x0,
            // instead of translating Chimera's own code as a guest block. A
            // first touch also hands back the image's thread-local
            // initializers (C++ `thread_local` constructors) — guest code the
            // runtime calls before the slot is handed over, the way dyld's
            // `tlv_allocate_and_initialize` calls them after installing the
            // block.
            if self.escape(pc, &escapes)? {
                continue;
            }
            if trace {
                let r = unsafe { &(*ts_ptr).regs };
                eprintln!(
                    "chimera: pc={:#x} x0={:#x} x8={:#x} x30={:#x} sp={:#x}",
                    pc,
                    r[X0],
                    r[8],
                    r[30],
                    unsafe { (*ts_ptr).sp }
                );
            }
            break_dump(ts_ptr, pc);
            let resolved = self
                .addr_space()
                .resolve(pc, block_exit, syscall_exit, trap_exit);
            let host_pc = match resolved {
                Ok(host_pc) => host_pc,
                // The guest branched into unmapped memory — a wild indirect
                // branch through a corrupted pointer, say. Natively the fetch
                // faults; raise the same SIGSEGV, which enters the guest's
                // handler or (default action) terminates the process
                // faithfully. Reflecting it here rather than letting the
                // translator's own load fault also keeps the report clean:
                // that fault would arrive while this thread holds the
                // address-space lock.
                Err(Error::BadAccess(_)) => {
                    let restart = self.restart.take();
                    let state = unsafe { &mut *ts_ptr };
                    let info: libc::siginfo_t = unsafe { std::mem::zeroed() };
                    self.signals
                        .deliver(state, libc::SIGSEGV as u32, &info, restart);
                    continue;
                }
                Err(e) => return Err(e),
            };
            unsafe {
                (*ts_ptr).exit_kind = EXIT_KIND_BLOCK;
                dispatch(ts_ptr, host_pc);
            }
            if unsafe { (*ts_ptr).exit_kind } == EXIT_KIND_SYSCALL {
                self.handle_syscall();
            }
            if unsafe { (*ts_ptr).exit_kind } == EXIT_KIND_TRAP {
                self.raise_trap();
            }
        }
        // The main thread's stack outlives its run, so this is the right
        // moment for it; a worker ran its own at `bsdthread_terminate`,
        // before the guest stack it needs was unmapped, and leaves nothing.
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
            // so far that a joiner has been released. Run them here — the
            // `bsdthread_terminate` arm keeps a backstop for a thread whose
            // cleanup never reaches this key.
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
        } else if pc == esc.stackaddr {
            // See `sys::darwin::guest_stackaddr`: libpthread's answer for the
            // main thread names the host stack, not the one the guest runs on.
            ret(
                ts,
                crate::sys::darwin::guest_stackaddr(unsafe { (*ts).regs[X0] }),
            );
        } else if pc == esc.stacksize {
            ret(
                ts,
                crate::sys::darwin::guest_stacksize(unsafe { (*ts).regs[X0] }),
            );
        } else if pc == esc.atfork {
            // Divert the handlers to the runtime's own list (see
            // `sys::darwin::GUEST_ATFORK`); `spawn::forked` runs them
            // translated around a guest fork.
            let (prepare, parent, child) =
                unsafe { ((*ts).regs[X0], (*ts).regs[X1], (*ts).regs[X2]) };
            crate::sys::darwin::guest_atfork_register(prepare, parent, child);
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
        } else if pc == esc.dispatch_apply || pc == esc.dispatch_apply_f {
            // `dispatch_apply` runs the guest's callback on libdispatch's own
            // worker threads — host threads with no translator context, which
            // would execute guest code natively off a page that is not
            // executable (and, were it executable, outside the sandbox). Run
            // the iterations on runtime-owned threads instead, through the
            // callback machinery, with this thread taking iterations too.
            // The iterations run here, in order, on this thread: the API
            // promises only that they all complete before it returns, not
            // that they run in parallel or on any particular thread. Running
            // them on runtime-owned threads instead was tried and reverted —
            // such a thread is one the *guest* never created, so its
            // thread-local storage is whatever an uninitialized pthread has,
            // and guest code that reads a thread-local got garbage from it.
            //
            // `dispatch_apply(n, queue, block)` calls `block->invoke(block,
            // i)` — the invoke pointer sits at offset 16 of a block object —
            // while `dispatch_apply_f(n, queue, context, work)` calls
            // `work(context, i)`.
            let (n, work, ctx) = unsafe {
                let (n, arg2, arg3) = ((*ts).regs[X0], (*ts).regs[X2], (*ts).regs[X3]);
                if pc == esc.dispatch_apply {
                    let mut invoke = [0u8; 8];
                    if !crate::sys::mmap::copy_from_guest(arg2 + 16, &mut invoke) {
                        return Err(Error::BadAccess(arg2 + 16));
                    }
                    (n, u64::from_ne_bytes(invoke), arg2)
                } else {
                    (n, arg3, arg2)
                }
            };
            for i in 0..n {
                self.run_guest_call2(work, ctx, i)?;
                if !self.running {
                    break;
                }
            }
            ret(ts, 0);
        } else if pc == esc.dispatch_async
            || pc == esc.dispatch_async_f
            || pc == esc.dispatch_group_async
            || pc == esc.dispatch_group_async_f
        {
            // The queue keeps the work and the thread it runs on; only the
            // pointer it calls changes. Hand libdispatch the runtime's shim
            // and a context naming the guest's function, so the callback
            // arrives on the library's worker thread and is run there through
            // the translator (see `crate::sys::darwin::callback`). Forwarding
            // the real call — rather than running the block here — is what
            // keeps the library's own bookkeeping intact, group counts and
            // queue ownership included.
            //
            // The group forms take the group ahead of the queue; the block
            // forms pass `block` where the `_f` forms pass `(context, work)`,
            // and call `block->invoke(block)`. A block may still be on the
            // caller's stack, so it is copied to the heap first — as a guest
            // call, since a block's copy helper is guest code too — and
            // released after the callback runs.
            let group = if pc == esc.dispatch_group_async || pc == esc.dispatch_group_async_f {
                unsafe { (*ts).regs[X0] }
            } else {
                0
            };
            let base = if group != 0 { 1 } else { 0 };
            let (queue, a, b) =
                unsafe { ((*ts).regs[base], (*ts).regs[base + 1], (*ts).regs[base + 2]) };
            let is_block = pc == esc.dispatch_async || pc == esc.dispatch_group_async;
            let (work, arg, release) = if is_block {
                let block = self.run_guest_call(esc.block_copy, a)?;
                let mut invoke = [0u8; 8];
                if !crate::sys::mmap::copy_from_guest(block + 16, &mut invoke) {
                    return Err(Error::BadAccess(block + 16));
                }
                (
                    u64::from_ne_bytes(invoke),
                    block,
                    Some((esc.block_release, block)),
                )
            } else {
                (b, a, None)
            };
            let context = crate::sys::darwin::callback::wrap(work, arg, release);
            unsafe extern "C" {
                fn dispatch_async_f(
                    queue: *mut libc::c_void,
                    context: *mut libc::c_void,
                    work: extern "C" fn(*mut libc::c_void),
                );
                fn dispatch_group_async_f(
                    group: *mut libc::c_void,
                    queue: *mut libc::c_void,
                    context: *mut libc::c_void,
                    work: extern "C" fn(*mut libc::c_void),
                );
            }
            let shim = crate::sys::darwin::callback::shim;
            if crate::sys::darwin::callback::is_fork_child() {
                crate::sys::darwin::callback::enqueue(context, group);
            } else {
                unsafe {
                    if group != 0 {
                        dispatch_group_async_f(
                            group as *mut libc::c_void,
                            queue as *mut libc::c_void,
                            context as *mut libc::c_void,
                            shim,
                        );
                    } else {
                        dispatch_async_f(
                            queue as *mut libc::c_void,
                            context as *mut libc::c_void,
                            shim,
                        );
                    }
                }
            }
            ret(ts, 0);
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
        let state = &mut *self.state;
        let info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        self.signals.deliver(state, SIGTRAP, &info, restart);
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
        publish_ctx(&mut *self.state);
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

        let ts_ptr: *mut ThreadState = &mut *self.state;
        let block_exit = exit_block as *const () as u64;
        let syscall_exit = exit_syscall_no_stack as *const () as u64;
        let trap_exit = exit_trap as *const () as u64;
        let escapes = Escapes::resolve();
        let trace = crate::trace::trace();

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
            let pc = unsafe { (*ts_ptr).pc };
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
                let state = unsafe { &mut *ts_ptr };
                self.signals.restore(state);
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
            if trace {
                let r = unsafe { &(*ts_ptr).regs };
                eprintln!(
                    "chimera: [call] pc={pc:#x} x0={:#x} x8={:#x} x30={:#x} sp={:#x}",
                    r[X0],
                    r[8],
                    r[30],
                    unsafe { (*ts_ptr).sp }
                );
            }
            break_dump(ts_ptr, pc);
            let resolved = self
                .addr_space()
                .resolve(pc, block_exit, syscall_exit, trap_exit);
            let host_pc = match resolved {
                Ok(host_pc) => host_pc,
                // The guest branched into unmapped memory — a wild indirect
                // branch through a corrupted pointer, say. Natively the fetch
                // faults; raise the same SIGSEGV, which enters the guest's
                // handler or (default action) terminates the process
                // faithfully. Reflecting it here rather than letting the
                // translator's own load fault also keeps the report clean:
                // that fault would arrive while this thread holds the
                // address-space lock.
                Err(Error::BadAccess(_)) => {
                    let restart = self.restart.take();
                    let state = unsafe { &mut *ts_ptr };
                    let info: libc::siginfo_t = unsafe { std::mem::zeroed() };
                    self.signals
                        .deliver(state, libc::SIGSEGV as u32, &info, restart);
                    continue;
                }
                Err(e) => return Err(e),
            };
            unsafe {
                (*ts_ptr).exit_kind = EXIT_KIND_BLOCK;
                dispatch(ts_ptr, host_pc);
            }
            if unsafe { (*ts_ptr).exit_kind } == EXIT_KIND_SYSCALL {
                self.handle_syscall();
            }
            // A `BRK` inside a guest call: end the call and let the outer
            // run loop raise the trap, for the same reason a fault does —
            // a guest call runs on a borrowed stack, and a handler frame
            // built there would corrupt the call in progress.
            if unsafe { (*ts_ptr).exit_kind } == EXIT_KIND_TRAP {
                break;
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

/// The calling host thread's TSD base: `TPIDRRO_EL0` with the CPU-number bits
/// masked off.
fn read_tsd_base() -> u64 {
    let v: u64;
    unsafe { std::arch::asm!("mrs {v}, tpidrro_el0", v = out(reg) v, options(nomem, nostack)) };
    v & !7
}

/// Byte offset of the TSD array inside libpthread's pthread struct, recovered
/// from the calling thread (whose TSD base the kernel set and whose struct
/// `pthread_self` names). A libpthread layout constant, so any thread's pair
/// yields the same value.
fn host_tsd_offset() -> u64 {
    read_tsd_base().wrapping_sub(unsafe { libc::pthread_self() } as u64)
}

/// Run one guest function on a host thread the runtime did not create — a
/// library's worker thread delivering a callback the guest registered (see
/// [`crate::sys::darwin::callback`]).
///
/// The thread has no translator context, so one is built here and torn down
/// afterwards: a register file, this thread's ctx published in its TSD slot so
/// translated code can find it, an alternate signal stack, and registration
/// with the shared [`Process`] so a group-wide stop can reach the thread while
/// it runs guest code. The guest runs on the host thread's own stack, below
/// the frame this function occupies — the library gave the thread a stack for
/// running work on, and this is that work.
pub fn run_guest_callback(process: &Arc<Process>, func: u64, arg: u64) {
    // The guest gets a stack of its own rather than a slice of this thread's.
    // Carving the host stack — handing the guest everything below a fixed
    // headroom — puts the guest's frames and Chimera's own on one stack
    // growing towards each other, so the runtime's native depth becomes a
    // silent correctness budget: exceed it and the dispatch loop's frames
    // overwrite the guest's. Measured, by adding unrelated code to `escape`
    // and watching every `cargo build` under Chimera die.
    const GUEST_STACK_SIZE: usize = 1 << 20;
    let stack = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            GUEST_STACK_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if stack == libc::MAP_FAILED {
        eprintln!("chimera: guest callback: no stack");
        return;
    }
    let guest_sp = (stack as u64 + GUEST_STACK_SIZE as u64) & !15;

    let mut state = ThreadState::new(func, guest_sp);
    state.guest_tsd = read_tsd_base();
    let mut thread = Thread::from_state(Arc::clone(process), state);

    let ts_ptr: *mut ThreadState = &mut *thread.state;
    publish_ctx(ts_ptr);
    crate::sys::darwin::fault::install_altstack();
    unsafe {
        let port = mach_thread_self();
        (*ts_ptr).tid.store(port as i32, Ordering::Release);
    }
    thread.process().register_thread(&thread.state);
    let _guard = ThreadGuard {
        process: Arc::clone(process),
        state: ts_ptr as *const ThreadState,
    };

    thread.running = true;
    if let Err(err) = thread.run_guest_call(func, arg) {
        eprintln!("chimera: guest callback failed: {err}");
    }
    unsafe { libc::munmap(stack, GUEST_STACK_SIZE) };
    // The ctx is this frame's and dies with it; leave nothing for the next
    // native code on this thread to find.
    publish_ctx(std::ptr::null_mut());
}

/// Dump the full register file when `pc` is the `CHIMERA_BREAK` target (see
/// `crate::trace`). Serviced in both dispatch loops — a spin inside a nested
/// guest call never reaches the outer one.
fn break_dump(ts: *const ThreadState, pc: u64) {
    if !crate::trace::break_hit(pc, 0x1_0000_0000 + crate::sys::darwin::image_slide()) {
        return;
    }
    let state = unsafe { &*ts };
    eprintln!("chimera: break at {pc:#x}");
    for (i, chunk) in state.regs.chunks(4).enumerate() {
        let cols: Vec<String> = chunk
            .iter()
            .enumerate()
            .map(|(j, v)| format!("x{:<2}={v:#018x}", i * 4 + j))
            .collect();
        eprintln!("  {}", cols.join(" "));
    }
    eprintln!("  sp ={:#018x} nzcv={:#010x}", state.sp, state.nzcv);
    // The first few registers dereferenced, for loops whose state lives in
    // memory the registers only point at; x20 twice, for a this-pointer whose
    // first field is itself an object.
    let deref = |label: &str, addr: u64| {
        let mut data = [0u8; 32];
        if addr > 0x1000 && crate::sys::mmap::copy_from_guest(addr, &mut data) {
            let words: Vec<String> = data
                .chunks(4)
                .map(|w| format!("{:08x}", u32::from_ne_bytes(w.try_into().unwrap())))
                .collect();
            eprintln!("  [{label}]: {}", words.join(" "));
        }
    };
    for (i, &reg) in state.regs.iter().take(4).enumerate() {
        deref(&format!("x{i}"), reg);
    }
    deref("x20", state.regs[20]);
    let mut ptr = [0u8; 8];
    if state.regs[20] > 0x1000 && crate::sys::mmap::copy_from_guest(state.regs[20], &mut ptr) {
        deref("[x20]", u64::from_ne_bytes(ptr));
    }
}

/// State handed to a spawned guest thread's host entry.
struct GuestThreadCtx {
    process: Arc<Process>,
    state: Box<ThreadState>,
    report: Arc<crate::sys::darwin::handoff::Handoff>,
}

/// Host `pthread` entry for a guest thread from `bsdthread_create`. Reports the
/// child's mach port to the waiting parent, then runs its dispatch loop.
extern "C" fn guest_thread_entry(arg: *mut libc::c_void) -> *mut libc::c_void {
    let ctx = unsafe { Box::from_raw(arg as *mut GuestThreadCtx) };
    let GuestThreadCtx {
        process,
        mut state,
        report,
    } = *ctx;
    let port = unsafe { mach_thread_self() } as u64;
    state.regs[X1] = port;
    // The kernel writes the new thread's mach port into the pthread struct when
    // it creates a bsdthread; `_pthread_start` aborts ("Unable to allocate thread
    // port") if that slot is unset. Replicate the write (self-port field at
    // pthread + 0xf8 on the current libpthread).
    unsafe { ((state.regs[X0] + 0xf8) as *mut u32).write(port as u32) };
    report.send(port);
    let mut child = Thread::from_state(process, state);
    let reason = child.run();
    // A fork in this thread made it the main (and only) thread of a new
    // process; the parent's other host threads did not survive the fork. Keep
    // driving it here, the way `execv` drives the initial thread — installing
    // any committed execve image and re-entering — and end the process with
    // the guest's status.
    if child.is_main()
        && let Ok(reason) = reason
    {
        let code = crate::sys::darwin::exec::drive(&mut child, reason).unwrap_or_else(|err| {
            eprintln!("chimera: fork child failed: {err}");
            127
        });
        std::process::exit(code);
    }
    // The guest ended this thread with `bsdthread_terminate`: the policy already
    // freed the guest stack and woke the joiner in guest program order, stashed
    // the Mach port, and stopped the run loop cleanly (the run guard already
    // unregistered), so every Rust object can drop here. Now issue the real
    // syscall as the host thread's last act — the kernel releases the port and
    // terminates this host thread. It never returns. (Chimera's own mmap'd host
    // stack for this thread leaks, as on any `pthread_attr_setstack` exit path.)
    let terminate = child.terminate_args;
    drop(child);
    if let Some(a) = terminate {
        let call = SystemCall::new(361, [a[0], a[1], a[2], a[3], 0, 0, 0, 0]);
        crate::sys::host_syscall(&call);
    }
    std::ptr::null_mut()
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
