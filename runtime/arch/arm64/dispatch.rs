//! The arm64 translator: a [`Vcpu`] that runs the guest by translating it, and
//! the guest register file it operates on.
//!
//! [`ThreadState`] holds the guest register file. Translated code reaches it
//! through this host thread's pthread TSD slot: each block's prologue reads
//! `TPIDRRO_EL0`, masks it to the pthread struct, and loads the [`ThreadState`]
//! pointer from the ctx key's slot (see `super::translate::emit_load_ctx`). That
//! keeps every guest GPR — `x18` included — free to hold its own value.
//!
//! [`Translator::run`] resolves (translating on a miss) the block at the guest
//! PC, enters the cache through [`dispatch`], and reports why the cache exited
//! — a block boundary, an SVC, or a BRK. What the guest's kernel does with
//! each exit is the Darwin thread's business (`crate::sys::darwin::thread`).
//! The boundary-crossing assembly lives in [`super::trampoline`].

use std::sync::{
    Arc,
    atomic::{AtomicI32, AtomicU32},
};

use crate::{
    Error,
    process::Process,
    vcpu::{Exit, Vcpu},
};

use super::trampoline::{dispatch, exit_block, exit_syscall_no_stack, exit_trap};

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
/// here and the nested loop in `crate::sys::darwin::thread::Thread::
/// run_guest_call` ends. Constrained like [`RETURN_SENTINEL`]: an `xpaci`
/// fixed point, and odd.
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
    /// Safepoint slot a sibling's process-wide stop arms; the translated
    /// loop-closing polls read it (see [`crate::process::Process`]).
    pub exit_requested: AtomicU32,
    /// The host identity a sibling's interrupt targets: this thread's Mach
    /// port.
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

    pub fn new(pc: u64, sp: u64) -> Box<Self> {
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
    pub fn reset(&mut self, pc: u64, sp: u64) {
        self.regs = [0; 31];
        self.fpstate = [0; 520];
        self.sp = sp;
        self.pc = pc;
        self.nzcv = 0;
        self.exit_kind = 0;
    }
}

/// A [`Vcpu`] that runs guest code by translating it into the process's
/// shared `MAP_JIT` code cache. Each [`Vcpu::run`] runs the cache until it
/// exits, which it does at an SVC, a BRK, or a block whose successor is not
/// yet linked; the last is an [`Exit::Safepoint`].
pub struct Translator {
    process: Arc<Process>,
}

impl Translator {
    pub fn new(process: Arc<Process>) -> Self {
        Self { process }
    }
}

impl Vcpu for Translator {
    fn attach(&mut self, state: &mut ThreadState) -> Result<(), Error> {
        // Publish this thread's ctx in its reserved TSD slot, so translated
        // code reaches the right ThreadState per host thread (via
        // TPIDRRO_EL0; see translate::emit_load_ctx). Per-thread, so
        // concurrent guest threads never race on it.
        publish_ctx(state);
        Ok(())
    }

    fn run(&mut self, state: &mut ThreadState) -> Exit {
        let block_exit = exit_block as *const () as u64;
        // SVC sites exit through the no-stack tail, which honors the kernel's
        // "syscalls don't touch the user stack" contract.
        let syscall_exit = exit_syscall_no_stack as *const () as u64;
        let trap_exit = exit_trap as *const () as u64;

        let pc = state.pc;
        // `CHIMERA_TRACE` logs the guest PC entering each block — the last
        // line before a crash localizes the faulting block during bring-up.
        // Sampled once at startup (crate::trace) so this never calls getenv
        // mid-run.
        if crate::trace::trace() {
            let r = &state.regs;
            eprintln!(
                "chimera: pc={pc:#x} x0={:#x} x8={:#x} x30={:#x} sp={:#x}",
                r[X0], r[8], r[30], state.sp
            );
        }
        break_dump(state, pc);
        // Hold the address-space lock only long enough to resolve (and, on a
        // miss, translate) the next block; the guard drops at the end of this
        // statement, before `dispatch()` runs the block.
        let host_pc =
            match self
                .process
                .lock_addr_space()
                .resolve(pc, block_exit, syscall_exit, trap_exit)
            {
                Ok(host_pc) => host_pc,
                Err(Error::BadAccess(_)) => return Exit::FetchFault,
                Err(e) => panic!("translate failed at {pc:#x}: {e}"),
            };
        state.exit_kind = EXIT_KIND_BLOCK;
        unsafe {
            dispatch(state, host_pc);
        }
        match state.exit_kind {
            EXIT_KIND_SYSCALL => Exit::Syscall,
            EXIT_KIND_TRAP => Exit::Breakpoint,
            _ => Exit::Safepoint,
        }
    }
}

/// The calling host thread's TSD base: `TPIDRRO_EL0` with the CPU-number bits
/// masked off.
pub fn read_tsd_base() -> u64 {
    let v: u64;
    unsafe { std::arch::asm!("mrs {v}, tpidrro_el0", v = out(reg) v, options(nomem, nostack)) };
    v & !7
}

/// Dump the full register file when `pc` is the `CHIMERA_BREAK` target (see
/// `crate::trace`). Checked on every cache entry, so a spin inside a nested
/// guest call is caught as well as one in the main run.
fn break_dump(state: &ThreadState, pc: u64) {
    if !crate::trace::break_hit(pc, 0x1_0000_0000 + crate::sys::darwin::image_slide()) {
        return;
    }
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
