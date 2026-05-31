//! The translate-execute loop and the guest register file it operates on.
//!
//! `ThreadState` holds the guest register file. The GS segment base is set to
//! point at it via `arch_prctl`, so translated code can reach any field with a
//! plain `gs:[disp]` access — no reserved GPR required.
//!
//! [`Thread::run`] is the loop: translate the next block if it isn't already
//! cached, enter the cache through [`dispatch`], handle whatever caused the
//! cache to exit (a block boundary or a syscall), and repeat until the guest
//! issues `exit_group` or `exit`. The boundary-crossing assembly lives in
//! [`super::trampoline`].

use std::arch::asm;

use crate::{Error, SystemCall, SystemCalls, sys::mmap::AddressSpace};

use super::{
    trampoline::{dispatch, exit_block, exit_syscall},
    translate::translate,
};

/// Linux x86-64 `arch_prctl` subfunction code: set the GS base.
const ARCH_SET_GS: libc::c_int = 0x1001;

const EXIT_KIND_BLOCK: u64 = 0;
pub const EXIT_KIND_SYSCALL: u64 = 1;

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
    /// Whether the run loop should keep iterating. Set true on entry to
    /// [`Thread::run`]; cleared by the `exit`/`exit_group` syscall
    /// implementation in [`crate::syscall::syscall`].
    pub running: bool,
    /// The status code the run loop returns once `running` is cleared.
    pub exit_code: i32,
}

impl Thread {
    /// Create a new guest thread.
    pub fn new(rip: u64, rsp: u64) -> Result<Self, Error> {
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
                _align_fpstate: [0; 1],
                fpstate: [0; XSAVE_AREA_SIZE],
            }),
            addr_space: AddressSpace::new()?,
            running: false,
            exit_code: 0,
        };
        thread.reset(rip, rsp);
        Ok(thread)
    }

    /// Reset the thread to a new entry point and a stack.
    pub fn reset(&mut self, rip: u64, rsp: u64) {
        self.addr_space.reset();
        self.state.reset(rip, rsp);
        self.running = false;
        self.exit_code = 0;
    }

    /// Run the guest using the thread's current entry state. Returns the
    /// guest's exit code once the `exit`/`exit_group` syscall implementation
    /// has cleared `running`; the syscall itself is never forwarded to the
    /// host kernel (that would terminate Chimera).
    pub fn run(&mut self, mut handler: Box<dyn SystemCalls>) -> Result<i32, Error> {
        self.setup_gs()?;
        self.state.setup_fs();
        self.running = true;

        let block_exit = exit_block as *const () as usize as u64;
        let syscall_exit = exit_syscall as *const () as usize as u64;

        while self.running {
            let ts_ptr: *mut ThreadState = &mut *self.state;

            let rip = unsafe { (*ts_ptr).rip };
            let host_pc = match self.addr_space.map.get(&rip) {
                Some(&hpc) => hpc,
                None => {
                    let hpc = translate(&mut self.addr_space.cache, rip, block_exit, syscall_exit)
                        .unwrap_or_else(|e| panic!("translate failed at {:#x}: {}", rip, e));
                    self.addr_space.map.insert(rip, hpc);
                    hpc
                }
            };
            unsafe {
                (*ts_ptr).exit_kind = EXIT_KIND_BLOCK;
                dispatch(ts_ptr, host_pc);
            }
            if unsafe { (*ts_ptr).exit_kind } == EXIT_KIND_SYSCALL {
                self.handle_syscall(handler.as_mut());
            }
        }
        Ok(self.exit_code)
    }

    fn handle_syscall(&mut self, handler: &mut dyn SystemCalls) {
        let mut call = SystemCall::new(
            self.state.regs[RAX],
            [
                self.state.regs[RDI],
                self.state.regs[RSI],
                self.state.regs[RDX],
                self.state.regs[R10],
                self.state.regs[R8],
                self.state.regs[R9],
            ],
        );
        crate::syscall::syscall(self, &mut call, handler);
        self.state.regs[RAX] = call.return_value() as u64;
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

/// Guest register file plus a few bookkeeping slots. The exact byte layout is
/// load-bearing: the offsets are consumed by `trampoline.S` (via `offset_of!`
/// in [`super::trampoline`]) and by the per-block exit stubs emitted by the
/// translator. The struct is 64-byte aligned, and the fields are arranged so
/// `fpstate` falls on a 64-byte boundary (offset 192) — XSAVE/XRSTOR `#GP`
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
    /// Padding so `fpstate` lands at offset 192 (a 64-byte boundary).
    _align_fpstate: [u64; 1],
    /// XSAVE area for the guest's extended FP/SIMD state (x87, SSE, AVX,
    /// AVX-512). Saved on every exit, restored on every entry. Must be
    /// 64-byte aligned; the field layout above guarantees offset 192.
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
    fn reset(&mut self, rip: u64, rsp: u64) {
        self.regs = [0; 16];
        self.rip = 0;
        self.rflags = 0;
        self.chimera_rsp = 0;
        self.host_pc_target = 0;
        self.exit_kind = 0;
        self.guest_fs_base = 0;
        self.chimera_fs_base = 0;
        self._align_fpstate = [0; 1];
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

    fn setup_fs(&mut self) {
        // Capture the executing host thread's FS base immediately before the
        // guest starts. A fresh guest inherits that value until it issues its
        // own `arch_prctl(ARCH_SET_FS, ...)`.
        let chimera_fs: u64;
        unsafe {
            asm!(
                "rdfsbase {0}",
                out(reg) chimera_fs,
                options(nomem, nostack, preserves_flags),
            );
        }
        self.chimera_fs_base = chimera_fs;
        self.guest_fs_base = chimera_fs;
    }
}

pub const RAX: usize = 0;
pub const RDX: usize = 3;
pub const RSI: usize = 4;
pub const RDI: usize = 5;
pub const RSP: usize = 7;
pub const R8: usize = 8;
pub const R9: usize = 9;
pub const R10: usize = 10;
