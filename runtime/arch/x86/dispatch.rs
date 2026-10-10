//! The translator: a [`Vcpu`] that runs the guest by translating it, and the
//! guest register file it operates on.
//!
//! `ThreadState` holds the guest register file. The GS segment base is set to
//! point at it via `arch_prctl`, so translated code can reach any field with a
//! plain `gs:[disp]` access — no reserved GPR required.
//!
//! [`Translator::run`] translates the next block if it isn't already cached,
//! enters the cache through [`dispatch`], and reports why the cache exited.
//! The boundary-crossing assembly lives in [`super::trampoline`].

use std::{
    arch::asm,
    sync::{
        Arc,
        atomic::{AtomicI32, AtomicU32, Ordering},
    },
};

use crate::{
    Error,
    process::Process,
    vcpu::{Exit, Vcpu},
};

use super::trampoline::{dispatch, exit_block, exit_syscall, exit_trap};

/// Linux x86-64 `arch_prctl` subfunction code: set the GS base.
const ARCH_SET_GS: libc::c_int = 0x1001;

pub const EXIT_KIND_BLOCK: u64 = 0;
pub const EXIT_KIND_SYSCALL: u64 = 1;
/// A guest breakpoint (`int3`) exited the cache.
pub const EXIT_KIND_TRAP: u64 = 2;

/// A [`Vcpu`] that runs guest code by translating it into the process's
/// shared code cache. Each [`Vcpu::run`] runs the cache until it exits, which
/// it does at a syscall, a breakpoint, or a block whose successor is not yet
/// linked; the last is a [`Exit::Safepoint`].
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
        // Emit (once per cache) the shared inline indirect-branch lookup routine
        // and record its address so translated indirect branches can reach it.
        state.ib_lookup = self
            .process
            .addr_space
            .lock()
            .unwrap()
            .code
            .ensure_ib_lookup(exit_block as *const () as usize as u64)?;
        Ok(())
    }

    fn run(&mut self, state: &mut ThreadState) -> Exit {
        let block_exit = exit_block as *const () as usize as u64;
        let syscall_exit = exit_syscall as *const () as usize as u64;
        let trap_exit = exit_trap as *const () as usize as u64;

        // Hold the address-space lock only long enough to resolve (and, on a
        // miss, translate) the next block; the guard drops at the end of this
        // statement, before `dispatch()` runs the block.
        let host_pc = match self.process.addr_space.lock().unwrap().resolve(
            state.rip,
            block_exit,
            syscall_exit,
            trap_exit,
        ) {
            Ok(host_pc) => host_pc,
            Err(Error::BadAccess(_)) => return Exit::FetchFault,
            Err(e) => panic!("translate failed at {:#x}: {}", state.rip, e),
        };
        state.exit_kind = EXIT_KIND_BLOCK;
        self.process.addr_space.lock().unwrap().code_deny_writes();
        unsafe {
            dispatch(state, host_pc);
        }
        self.process.addr_space.lock().unwrap().code_allow_writes();
        match state.exit_kind {
            EXIT_KIND_SYSCALL => Exit::Syscall,
            EXIT_KIND_TRAP => Exit::Breakpoint,
            _ => Exit::Safepoint,
        }
    }
}

/// Size of the XSAVE area in [`ThreadState::fpstate`]. The standard (non-
/// compacted) XSAVE layout for x87+SSE+AVX+AVX-512 ends at architecturally
/// fixed offsets totaling ~2688 bytes; 4096 leaves comfortable margin. The
/// trampoline saves/restores with the `0xe7` component mask, which never
/// selects AMX, so the area size is bounded regardless of the host's XCR0.
const XSAVE_AREA_SIZE: usize = 4096;

/// Guest register file plus a few bookkeeping slots. The exact byte layout is
/// load-bearing: the offsets are consumed by `trampoline.S` (via `offset_of!`
/// in [`super::trampoline`]) and by the per-block exit stubs emitted by the
/// translator. The struct is 64-byte aligned, and the fields are arranged so
/// `fpstate` falls on a 64-byte boundary (offset 256) — XSAVE/XRSTOR `#GP`
/// on a misaligned save area.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct ThreadState {
    /// Guest GPRs: rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8..r15.
    pub regs: [u64; 16],
    /// Guest program counter; set on exit, read on entry.
    pub rip: u64,
    /// Guest rflags; set on exit, read on entry.
    pub rflags: u64,
    /// Chimera's stack pointer, saved on entry and restored on exit.
    pub chimera_rsp: u64,
    /// Host PC for the next entry, used by `dispatch` after it has already
    /// clobbered `rsi`.
    pub host_pc_target: u64,
    /// Why the last exit happened. Read by the run loop after every entry,
    /// reset to `BLOCK` before each entry.
    pub exit_kind: u64,
    /// Guest's FS base. Loaded into the FS MSR on every entry, restored
    /// from the FS MSR on every exit. Updated by `syscall` when it
    /// intercepts `arch_prctl(ARCH_SET_FS, ...)`.
    pub guest_fs_base: u64,
    /// Chimera's FS base, captured on the host thread immediately before
    /// guest execution starts. Restored on every exit so the runtime's own
    /// TLS works after the guest has changed FS.
    pub chimera_fs_base: u64,
    /// Host address of the shared inline indirect-branch lookup routine in the
    /// code cache (`CodeCache::ensure_ib_lookup`). Each translated indirect
    /// branch ends in `jmp gs:[ib_lookup]`; set once per run before the loop.
    pub ib_lookup: u64,
    /// Scratch slots used only by the inline indirect-branch lookup routine,
    /// which has no free registers of its own: the guest's flags (via
    /// `lahf`/`seto`), the branch target, the borrowed rcx/rdx, and the
    /// resolved host PC. Live only for the duration of one lookup.
    pub ib_flags: u64,
    pub ib_target: u64,
    pub ib_rcx: u64,
    pub ib_rdx: u64,
    pub ib_host: u64,
    /// Asynchronous-exit flag polled by translated code at loop-closing edges.
    /// The host signal catcher sets it; the run loop recomputes it each iteration
    /// as "a deliverable signal is pending" so it self-clears. When set, a fully
    /// linked, syscall-free guest loop is dragged back to the run loop within one
    /// iteration, where a pending signal is delivered at a real block boundary.
    /// Reached from translated code as `gs:[]`.
    ///
    /// `AtomicU32` because the signal catcher writes it asynchronously (from the
    /// handler, via `gs:[]`) while the run loop reads and writes it; plain accesses
    /// would be a data race the compiler could miscompile (e.g. drop the clear in
    /// [`Thread::refresh_exit_requested`] as a dead store). The in-memory layout is
    /// an identical 32-bit word at the same offset, so the `gs:[]` poll is
    /// unaffected.
    pub exit_requested: AtomicU32,
    /// Whether the physical FP/SIMD registers currently hold this thread's
    /// guest state. `dispatch` clears it on every cache entry (the Rust code
    /// that ran since the last exit has clobbered the vector registers); the
    /// first translated block that actually touches FP restores `fpstate` and
    /// sets it (see the translator's block prologue). The exit trampolines save
    /// `fpstate` only when it is set, so a residency that never touches FP pays
    /// neither XRSTOR nor XSAVE. Read and written by both `trampoline.S` and the
    /// emitted prologue, so it sits in the gs-reachable region.
    /// 32-bit so it packs against `exit_requested` and `fpstate` stays at
    /// offset 256; only 0 and 1 are ever stored.
    pub fp_in_regs: u32,
    /// Scratch slot the block prologue uses to park the guest's status flags
    /// (via `lahf`/`seto`) across its checks, for the blocks whose first
    /// instructions read flags set by a predecessor. Live only for the few
    /// instructions of one prologue.
    pub fp_flags: u64,
    /// Scratch slot the block prologue uses to park the guest's rdx across the
    /// `xrstor64` (whose `edx` mask half clobbers it). Live only for the
    /// duration of one restore.
    pub fp_scratch: u64,
    /// XSAVE area for the guest's extended FP/SIMD state (x87, SSE, AVX,
    /// AVX-512). The canonical copy whenever no translated code is running;
    /// saved and restored only around blocks that touch FP (see `fp_in_regs`).
    /// Must be 64-byte aligned; the field layout above guarantees offset 256.
    pub fpstate: [u8; XSAVE_AREA_SIZE],
    /// Whether the FS base register currently holds the guest's base (rather
    /// than Chimera's, used by Rust TLS). Mirrors `fp_in_regs` for the FS-base
    /// swap: `dispatch` clears it and leaves FS holding Chimera's base, the
    /// first block that reads guest TLS (`fs:`) installs `guest_fs_base` and
    /// sets it, and the exit trampolines restore Chimera's base only when it is
    /// set. A residency that never touches `fs:` keeps both `wrfsbase`s off the
    /// path. Placed after `fpstate` so that field stays 64-byte aligned.
    pub fs_is_guest: u64,
    /// Address of this thread's `PendingSet` (owned by its `Signals`), read by
    /// the host signal catcher via `gs:[]` so a caught signal is recorded on
    /// the thread that caught it — pending state is per-thread, and this
    /// pointer is how the async-signal-safe catcher finds the right set with
    /// no TLS. Placed after `fpstate` so every offset above is unchanged.
    pub pending_set: u64,
    /// The kernel TID of the host thread backing this guest thread. Written by
    /// the owning thread at run entry — and rewritten in a fork child, where
    /// the copied value names a thread that no longer exists — and read by
    /// siblings through the thread list as the target for the reserved interrupt
    /// signal. Atomic because those reads are cross-thread; placed after
    /// `fpstate` so every `gs:[]` offset above is unchanged.
    pub tid: AtomicI32,
    /// Save slot for the register a far-RIP-relative rewrite borrows to
    /// materialize an absolute guest address (see `rewrite_far_rip_operand` in
    /// the translator). Live only across the few instructions of one rewritten
    /// guest instruction, always within a block body; placed after `fpstate`
    /// so every `gs:[]` offset above is unchanged.
    pub riprel_scratch: u64,
    /// This thread's syscall user dispatch selector: the byte the kernel reads
    /// on every system call to decide whether to let it through (`ALLOW`) or
    /// raise SIGSYS (`BLOCK`). Only the native vcpu arms it
    /// ([`super::sud`]); it is `ALLOW` whenever Chimera's own code runs.
    pub sud_selector: u8,
    /// Whether this thread is executing guest code natively: set by the native
    /// vcpu's entry stub, cleared when a signal handler takes the thread back
    /// to Chimera. A runtime signal handler that finds it set has interrupted
    /// guest code and must leave through the native vcpu's exit path.
    pub sud_in_guest: u8,
}

// XSAVE/XRSTOR #GP unless the save area is 64-byte aligned. The struct's
// align(64) handles the allocation; this guards the field offset against a
// future reordering.
const _: () = assert!(
    core::mem::offset_of!(ThreadState, fpstate) % 64 == 0,
    "ThreadState::fpstate must be 64-byte aligned for XSAVE/XRSTOR"
);

impl ThreadState {
    /// A zeroed register file; [`ThreadState::reset`] gives it an entry point.
    pub fn new() -> Box<Self> {
        Box::new(Self {
            regs: [0; 16],
            rip: 0,
            rflags: 0,
            chimera_rsp: 0,
            host_pc_target: 0,
            exit_kind: 0,
            guest_fs_base: 0,
            chimera_fs_base: 0,
            ib_lookup: 0,
            ib_flags: 0,
            ib_target: 0,
            ib_rcx: 0,
            ib_rdx: 0,
            ib_host: 0,
            exit_requested: AtomicU32::new(0),
            fp_in_regs: 0,
            fp_flags: 0,
            fp_scratch: 0,
            fpstate: [0; XSAVE_AREA_SIZE],
            fs_is_guest: 0,
            pending_set: 0,
            tid: AtomicI32::new(0),
            riprel_scratch: 0,
            sud_selector: 0,
            sud_in_guest: 0,
        })
    }

    /// Point the calling host thread's GS base at this register file, so
    /// translated code and the host signal catcher reach it as `gs:[disp]`.
    /// GS is host-thread-local, so this runs on the OS thread that is about
    /// to execute the guest.
    pub fn install_gs(&self) -> Result<(), Error> {
        let state_addr = self as *const ThreadState as usize;
        let ret = unsafe { libc::syscall(libc::SYS_arch_prctl, ARCH_SET_GS, state_addr) };
        if ret != 0 {
            return Err(Error::last_os_error("arch_prctl(ARCH_SET_GS)"));
        }
        Ok(())
    }

    pub fn reset(&mut self, rip: u64, rsp: u64, guest_fs_base: u64) {
        self.regs = [0; 16];
        self.rip = 0;
        self.rflags = 0;
        self.chimera_rsp = 0;
        self.host_pc_target = 0;
        self.exit_kind = 0;
        self.guest_fs_base = guest_fs_base;
        self.chimera_fs_base = 0;
        self.ib_lookup = 0;
        self.ib_flags = 0;
        self.ib_target = 0;
        self.ib_rcx = 0;
        self.ib_rdx = 0;
        self.ib_host = 0;
        self.riprel_scratch = 0;
        self.sud_selector = 0;
        self.sud_in_guest = 0;
        self.exit_requested.store(0, Ordering::Relaxed);
        self.fp_in_regs = 0;
        self.fp_flags = 0;
        self.fp_scratch = 0;
        self.fs_is_guest = 0;
        self.fpstate.fill(0);

        // XRSTOR loads MXCSR from the legacy region (bytes 24..28) of the save
        // area on every entry, regardless of XSTATE_BV. A zeroed area would load
        // MXCSR = 0, which unmasks every SSE floating-point exception and turns
        // ordinary float math into a SIGFPE. Seed it with the ABI default
        // 0x1f80 (all exceptions masked) — the value the Linux kernel gives a
        // fresh process — for the first entry. After that the guest's own MXCSR
        // round-trips through XSAVE/XRSTOR. The x87 control word and all vector
        // registers initialize correctly from the zeroed XSTATE_BV.
        self.fpstate[24..28].copy_from_slice(&0x0000_1f80u32.to_le_bytes());
        self.regs[RSP] = rsp;
        self.rip = rip;
    }

    pub fn capture_chimera_fs(&mut self) {
        self.chimera_fs_base = current_fs_base();
    }

    /// Build the register state for a `clone(CLONE_VM)` child. The guest-visible
    /// registers, flags, and FP/SIMD state are copied from the parent, so the
    /// child resumes exactly where the parent's `clone` returns — except `rax`,
    /// which is 0 (the child's `clone` return value), and `rsp`, which is the
    /// child's own stack. `tls` is the child's thread pointer (FS base) when the
    /// clone requested `CLONE_SETTLS`, else the child inherits the parent's. The
    /// per-run bookkeeping slots (the Chimera/FS scratch and the indirect-branch
    /// lookup fields) start cleared; `run` repopulates them on the child's own
    /// host thread.
    pub fn clone_for_child(&self, child_stack: u64, tls: Option<u64>) -> Box<ThreadState> {
        let mut child = Box::new(ThreadState {
            regs: self.regs,
            rip: self.rip,
            rflags: self.rflags,
            chimera_rsp: 0,
            host_pc_target: 0,
            exit_kind: 0,
            guest_fs_base: tls.unwrap_or(self.guest_fs_base),
            chimera_fs_base: 0,
            ib_lookup: 0,
            ib_flags: 0,
            ib_target: 0,
            ib_rcx: 0,
            ib_rdx: 0,
            ib_host: 0,
            exit_requested: AtomicU32::new(0),
            fp_in_regs: 0,
            fp_flags: 0,
            fp_scratch: 0,
            fpstate: self.fpstate,
            fs_is_guest: 0,
            // Cleared, not inherited: the child's own `PendingSet` address is
            // published by `Thread::from_state`. Copying the parent's pointer
            // would let the catcher record the child's signals on the parent.
            pending_set: 0,
            // Cleared for the same reason as the run-entry store: the child
            // publishes its own TID before it runs any guest code.
            tid: AtomicI32::new(0),
            riprel_scratch: 0,
            sud_selector: 0,
            sud_in_guest: 0,
        });
        child.regs[RAX] = 0;
        child.regs[RSP] = child_stack;
        child
    }
}

pub fn current_fs_base() -> u64 {
    let fs_base: u64;
    unsafe {
        asm!(
            "rdfsbase {0}",
            out(reg) fs_base,
            options(nomem, nostack, preserves_flags),
        );
    }
    fs_base
}

pub const RAX: usize = 0;
pub const RDX: usize = 3;
pub const RSI: usize = 4;
pub const RDI: usize = 5;
pub const RSP: usize = 7;
pub const R8: usize = 8;
pub const R9: usize = 9;
pub const R10: usize = 10;
