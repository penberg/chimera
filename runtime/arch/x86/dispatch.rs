//! The translate-execute loop and the guest register file it operates on.
//!
//! `ThreadState` holds the guest register file. The GS segment base is set to
//! point at it via `arch_prctl`, so translated code can reach any field with a
//! plain `gs:[disp]` access — no reserved GPR required.
//!
//! [`start_thread`] is the loop: translate the next block if it isn't already cached,
//! enter the cache through [`dispatch`], handle whatever caused the cache to
//! exit (a block boundary or a syscall), and repeat until the guest issues
//! `exit_group` or `exit`. The boundary-crossing assembly lives in
//! [`super::trampoline`].

use std::{arch::asm, collections::HashMap};

use crate::{Error, SystemCall, SystemCalls};

use super::{
    trampoline::{dispatch, exit_block, exit_syscall},
    translate::{CodeCache, translate},
};

/// Linux x86-64 `arch_prctl` subfunction code: set the GS base.
const ARCH_SET_GS: libc::c_int = 0x1001;

const EXIT_KIND_BLOCK: u64 = 0;
pub const EXIT_KIND_SYSCALL: u64 = 1;

/// Guest register file plus a few bookkeeping slots. The exact byte layout is
/// load-bearing: the offsets are consumed by `trampoline.S` (via `offset_of!`
/// in [`super::trampoline`]) and by the per-block exit stubs emitted by the
/// translator. The struct is 16-byte aligned so the `fpstate` slot is
/// suitable for FXSAVE/FXRSTOR.
#[repr(C, align(16))]
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
    /// from the FS MSR on every exit. Updated by `host_syscall` when it
    /// intercepts `arch_prctl(ARCH_SET_FS, ...)`.
    pub guest_fs_base: u64,
    /// 512-byte buffer for FXSAVE/FXRSTOR (x87 + SSE state). Aligned at
    /// offset 176 inside the 16-byte-aligned struct.
    pub fpstate: [u8; 512],
    /// Chimera's FS base, captured once at the start of [`start_thread`]. Restored on
    /// every exit so the runtime's own TLS works after the guest has
    /// changed FS.
    pub chimera_fs_base: u64,
}

impl Default for ThreadState {
    fn default() -> Self {
        Self {
            regs: [0; 16],
            rip: 0,
            rflags: 0,
            chimera_rsp: 0,
            host_pc_target: 0,
            exit_kind: 0,
            guest_fs_base: 0,
            fpstate: [0; 512],
            chimera_fs_base: 0,
        }
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

/// Run the guest, starting with `rsp` and `rip` as given. Returns the guest's
/// exit code when it issues `exit_group` or `exit`; the syscall itself is not
/// forwarded to the host kernel (that would terminate Chimera). The handler
/// still observes the call before the run ends.
pub fn start_thread(rip: u64, rsp: u64, mut handler: Box<dyn SystemCalls>) -> Result<i32, Error> {
    let mut cache = CodeCache::new()?;
    let mut map: HashMap<u64, u64> = HashMap::new();
    let mut ts = Box::new(ThreadState::default());

    // Install the ThreadState pointer as this thread's GS base.
    let ts_addr = &*ts as *const ThreadState as usize;
    let ret = unsafe { libc::syscall(libc::SYS_arch_prctl, ARCH_SET_GS, ts_addr) };
    if ret != 0 {
        return Err(Error::last_os_error("arch_prctl(ARCH_SET_GS)"));
    }

    // Capture Chimera's FS base. The trampolines restore it on every exit,
    // and a fresh guest starts with the same value (until its first
    // `arch_prctl(ARCH_SET_FS, ...)`, which `host_syscall` intercepts).
    let chimera_fs: u64;
    unsafe {
        asm!(
            "rdfsbase {0}",
            out(reg) chimera_fs,
            options(nomem, nostack, preserves_flags),
        );
    }
    ts.chimera_fs_base = chimera_fs;
    ts.guest_fs_base = chimera_fs;

    let ts_ptr: *mut ThreadState = &mut *ts as *mut ThreadState;
    unsafe {
        (*ts_ptr).regs[RSP] = rsp;
        (*ts_ptr).rip = rip;
    }
    let block_exit = exit_block as *const () as usize as u64;
    let syscall_exit = exit_syscall as *const () as usize as u64;

    loop {
        let rip = unsafe { (*ts_ptr).rip };
        let host_pc = match map.get(&rip) {
            Some(&hpc) => hpc,
            None => {
                let hpc = translate(&mut cache, rip, block_exit, syscall_exit)
                    .unwrap_or_else(|e| panic!("translate failed at {:#x}: {}", rip, e));
                map.insert(rip, hpc);
                hpc
            }
        };
        unsafe {
            (*ts_ptr).exit_kind = EXIT_KIND_BLOCK;
            dispatch(ts_ptr, host_pc);
        }
        if unsafe { (*ts_ptr).exit_kind } == EXIT_KIND_SYSCALL
            && let Some(code) = handle_syscall(ts_ptr, handler.as_mut())
        {
            return Ok(code);
        }
    }
}

fn handle_syscall(ts_ptr: *mut ThreadState, handler: &mut dyn SystemCalls) -> Option<i32> {
    let regs = unsafe { &(*ts_ptr).regs };
    let number = regs[RAX];
    let args = [
        regs[RDI], regs[RSI], regs[RDX], regs[R10], regs[R8], regs[R9],
    ];
    let exit_code = if number == libc::SYS_exit_group as u64 || number == libc::SYS_exit as u64 {
        Some(args[0] as i32)
    } else {
        None
    };
    let mut call = SystemCall::new(number, args);
    handler.handle(&mut call);
    unsafe {
        (*ts_ptr).regs[RAX] = call.return_value() as u64;
    }
    exit_code
}
