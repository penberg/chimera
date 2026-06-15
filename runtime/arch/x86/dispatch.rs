//! The translate-execute loop and the guest register file it operates on.
//!
//! `ThreadState` holds the guest register file. The GS segment base is set to
//! point at it via `arch_prctl`, so translated code can reach any field with a
//! plain `gs:[disp]` access — no reserved GPR required.
//!
//! [`Thread::run`] is the loop: deliver any pending guest signal, translate the
//! next block if it isn't already cached, enter the cache through [`dispatch`],
//! handle whatever caused the cache to exit (a block boundary or a syscall), and
//! repeat until the guest issues `exit_group` or `exit`. The boundary-crossing
//! assembly lives in [`super::trampoline`].

use std::arch::asm;

use crate::{
    Error, SystemCall, SystemCalls,
    sys::{linux::signal::Signals, mmap::AddressSpace},
};

use super::trampoline::{dispatch, exit_block, exit_syscall};

/// Linux x86-64 `arch_prctl` subfunction code: set the GS base.
const ARCH_SET_GS: libc::c_int = 0x1001;

const EXIT_KIND_BLOCK: u64 = 0;
pub const EXIT_KIND_SYSCALL: u64 = 1;

/// Why [`Thread::run`] returned: the guest terminated, or it `execve`d and the
/// caller must load the new image and re-enter the thread on it.
pub enum ExitReason {
    /// The guest issued `exit`/`exit_group`. Carries the exit code.
    Exited(i32),
    /// The guest issued an `execve`/`execveat` the handler allowed. `number`
    /// distinguishes the two; `args` are the raw guest syscall arguments
    /// (pathname, argv, and envp pointers among them).
    Execve { number: u64, args: [u64; 6] },
}

/// Size of the XSAVE area in [`ThreadState::fpstate`]. The standard (non-
/// compacted) XSAVE layout for x87+SSE+AVX+AVX-512 ends at architecturally
/// fixed offsets totaling ~2688 bytes; 4096 leaves comfortable margin. The
/// trampoline saves/restores with the `0xe7` component mask, which never
/// selects AMX, so the area size is bounded regardless of the host's XCR0.
const XSAVE_AREA_SIZE: usize = 4096;

/// The `Thread` struct represents a guest thread: the register state and the
/// address space it runs in. The purpose of this struct is similar to the
/// `task_struct` in the Linux kernel. `running` and `exit_code` mirror the
/// kernel's task state: a syscall implementation can mark a thread done by
/// clearing `running` and recording an `exit_code`, and the run loop
/// terminates on its next iteration.
pub struct Thread {
    pub state: Box<ThreadState>,
    addr_space: AddressSpace,
    /// Per-process guest signal state: handler table, blocked mask, alt stack.
    signals: Signals,
    /// Whether the run loop should keep iterating. Set true on entry to
    /// [`Thread::run`]; cleared by the `exit`/`exit_group` syscall
    /// implementation in [`crate::syscall::syscall`].
    pub running: bool,
    /// The status code the run loop returns once `running` is cleared.
    pub exit_code: i32,
    /// Set when the most recently forwarded syscall returned `EINTR` and is
    /// restartable: `(resume rip after the syscall, original syscall number)`.
    /// Consumed at the next signal delivery to honor `SA_RESTART`. Cleared
    /// after every syscall, so it reflects only the immediately preceding one.
    restart: Option<(u64, u64)>,
}

impl Thread {
    /// Create a new guest thread.
    pub fn new(rip: u64, rsp: u64) -> Result<Self, Error> {
        let guest_fs_base = current_fs_base();
        let mut thread = Self {
            state: Box::new(ThreadState {
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
                _align_fpstate: [0; 3],
                fpstate: [0; XSAVE_AREA_SIZE],
            }),
            addr_space: AddressSpace::new()?,
            signals: Signals::new(),
            running: false,
            exit_code: 0,
            restart: None,
        };
        thread.reset(rip, rsp, guest_fs_base);
        Ok(thread)
    }

    /// Reset the thread to a new entry point and a stack.
    pub fn reset(&mut self, rip: u64, rsp: u64, guest_fs_base: u64) {
        self.addr_space.reset();
        self.state.reset(rip, rsp, guest_fs_base);
        self.running = false;
        self.exit_code = 0;
    }

    pub fn addr_space(&mut self) -> &mut AddressSpace {
        &mut self.addr_space
    }

    pub fn signals_mut(&mut self) -> &mut Signals {
        &mut self.signals
    }

    /// Restore the pre-signal guest context on a guest `rt_sigreturn`.
    pub fn sigreturn(&mut self) {
        let state = &mut *self.state;
        self.signals.restore(state);
    }

    /// Deliver one pending, unblocked guest signal at a safe point (a block
    /// boundary), building its frame and redirecting the guest to the handler.
    /// Entry into the handler then happens through the normal `dispatch` path.
    fn deliver_pending_signals(&mut self) {
        if let Some(signo) = crate::sys::linux::signal::pending_take_one(self.signals.blocked) {
            let restart = self.restart.take();
            let state = &mut *self.state;
            self.signals.deliver(state, signo, restart);
        }
    }

    /// Set the thread's entry registers for a freshly mapped image, without
    /// touching its address space. Used to re-enter after an `execve` once the
    /// caller has torn down the old image and mapped the new one.
    pub fn enter(&mut self, rip: u64, rsp: u64) {
        self.state.reset(rip, rsp, current_fs_base());
    }

    /// Run the guest using the thread's current entry state. Returns when the
    /// guest issues `exit`/`exit_group` (with the code) or an allowed
    /// `execve`/`execveat` (for the caller to act on); neither syscall is
    /// forwarded to the host kernel. The handler observes the call first.
    pub fn run(&mut self, handler: &mut dyn SystemCalls) -> Result<ExitReason, Error> {
        // GS is host-thread-local, so bind it on the OS thread that is
        // actually about to execute the translated guest.
        self.setup_gs()?;
        self.state.capture_chimera_fs();
        self.running = true;

        let block_exit = exit_block as *const () as usize as u64;
        let syscall_exit = exit_syscall as *const () as usize as u64;

        // Emit (once per cache) the shared inline indirect-branch lookup routine
        // and record its address so translated indirect branches can reach it.
        self.state.ib_lookup = self.addr_space.code.ensure_ib_lookup(block_exit)?;

        while self.running {
            self.deliver_pending_signals();

            let ts_ptr: *mut ThreadState = &mut *self.state;

            let rip = unsafe { (*ts_ptr).rip };
            let host_pc = self
                .addr_space
                .code
                .resolve(rip, block_exit, syscall_exit)
                .unwrap_or_else(|e| panic!("translate failed at {:#x}: {}", rip, e));
            unsafe {
                (*ts_ptr).exit_kind = EXIT_KIND_BLOCK;
                dispatch(ts_ptr, host_pc);
            }
            if unsafe { (*ts_ptr).exit_kind } == EXIT_KIND_SYSCALL
                && let Some(reason) = self.handle_syscall(handler)
            {
                return Ok(reason);
            }
        }
        Ok(ExitReason::Exited(self.exit_code))
    }

    /// Service the syscall that just exited the cache. Returns `Some` only when
    /// the guest issued an allowed `execve`/`execveat`, in which case the run
    /// loop hands the request back to its caller to re-enter on the new image.
    fn handle_syscall(&mut self, handler: &mut dyn SystemCalls) -> Option<ExitReason> {
        let number = self.state.regs[RAX];
        let args = [
            self.state.regs[RDI],
            self.state.regs[RSI],
            self.state.regs[RDX],
            self.state.regs[R10],
            self.state.regs[R8],
            self.state.regs[R9],
        ];
        let mut call = SystemCall::new(number, args);
        crate::syscall::syscall(self, &mut call, handler);
        let result = call.return_value();
        self.state.regs[RAX] = result as u64;

        // Record a restart candidate for SA_RESTART: a forwarded slow syscall
        // interrupted by a signal returns EINTR, and the dispatcher must be able
        // to re-issue it if the delivered handler asked to restart. `state.rip`
        // is the instruction after the `syscall` (the syscall is 2 bytes wide).
        // The never-restart interfaces always surface EINTR, so they are
        // excluded here. Cleared on any other syscall outcome.
        self.restart = if result == -(libc::EINTR as i64) && !never_restart(number) {
            Some((self.state.rip, number))
        } else {
            None
        };

        if number == libc::SYS_execve as u64 || number == libc::SYS_execveat as u64 {
            return Some(ExitReason::Execve { number, args });
        }
        None
    }

    fn setup_gs(&self) -> Result<(), Error> {
        let state_addr = &*self.state as *const ThreadState as usize;
        let ret = unsafe { libc::syscall(libc::SYS_arch_prctl, ARCH_SET_GS, state_addr) };
        if ret != 0 {
            return Err(Error::last_os_error("arch_prctl(ARCH_SET_GS)"));
        }
        Ok(())
    }
}

/// Whether a syscall interrupted by a signal must always fail with `EINTR`,
/// never restarting even under `SA_RESTART`. These are the interfaces the kernel
/// documents as non-restartable (`signal(7)`): the signal/event waits, the
/// multiplexing calls, sleeps, and System V IPC. Any other interrupted slow
/// syscall is a restart candidate.
fn never_restart(number: u64) -> bool {
    matches!(
        number as i64,
        libc::SYS_pause
            | libc::SYS_rt_sigsuspend
            | libc::SYS_rt_sigtimedwait
            | libc::SYS_poll
            | libc::SYS_ppoll
            | libc::SYS_select
            | libc::SYS_pselect6
            | libc::SYS_epoll_wait
            | libc::SYS_epoll_pwait
            | libc::SYS_nanosleep
            | libc::SYS_clock_nanosleep
            | libc::SYS_msgrcv
            | libc::SYS_msgsnd
            | libc::SYS_semop
            | libc::SYS_semtimedop
            | libc::SYS_io_getevents
    )
}

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
    /// Padding so `fpstate` lands at offset 256 (a 64-byte boundary).
    _align_fpstate: [u64; 3],
    /// XSAVE area for the guest's extended FP/SIMD state (x87, SSE, AVX,
    /// AVX-512). Saved on every exit, restored on every entry. Must be
    /// 64-byte aligned; the field layout above guarantees offset 256.
    pub fpstate: [u8; XSAVE_AREA_SIZE],
}

// XSAVE/XRSTOR #GP unless the save area is 64-byte aligned. The struct's
// align(64) handles the allocation; this guards the field offset against a
// future reordering.
const _: () = assert!(
    core::mem::offset_of!(ThreadState, fpstate) % 64 == 0,
    "ThreadState::fpstate must be 64-byte aligned for XSAVE/XRSTOR"
);

impl ThreadState {
    fn reset(&mut self, rip: u64, rsp: u64, guest_fs_base: u64) {
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
        self._align_fpstate = [0; 3];
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

    fn capture_chimera_fs(&mut self) {
        self.chimera_fs_base = current_fs_base();
    }
}

fn current_fs_base() -> u64 {
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
