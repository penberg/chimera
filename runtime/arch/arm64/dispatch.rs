//! The translate-execute loop and the guest register file it operates on.
//!
//! `ThreadState` holds the guest register file. The pointer to it is held in
//! `x18` while translated code is running — on Darwin, `x18` is reserved for
//! the platform, so the guest never touches it. Translated code reaches any
//! field with a plain `ldr/str x?, [x18, #disp]`.
//!
//! [`start_thread`] is the loop: translate the next block if it isn't already
//! cached, enter the cache through [`dispatch`], handle whatever caused the
//! cache to exit (a block boundary or a syscall), and repeat until the guest
//! issues `exit`. The boundary-crossing assembly lives in [`super::trampoline`].

use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{Error, SystemCall, SystemCalls};

/// Global slot the trampolines and translated-code exit stubs use to find
/// the active `ThreadState`. Set once per `start_thread` invocation.
#[unsafe(no_mangle)]
pub static CHIMERA_CTX_PTR: AtomicU64 = AtomicU64::new(0);

use super::{
    trampoline::{dispatch, exit_block, exit_syscall, exit_syscall_no_stack},
    translate::{CodeCache, translate},
};

const EXIT_KIND_BLOCK: u64 = 0;
pub const EXIT_KIND_SYSCALL: u64 = 1;

/// Guest register file plus a few bookkeeping slots. The exact byte layout is
/// load-bearing: the offsets are consumed by `trampoline.S` (via `offset_of!`
/// in [`super::trampoline`]) and by the per-block exit stubs emitted by the
/// translator. The struct is 16-byte aligned so the `fpstate` slot is
/// suitable for `ldp q?, q?` / `stp q?, q?`.
///
/// Field ordering is chosen so `regs[0..31]` lands at offset 0, which gives
/// the dispatch/exit trampolines the tightest immediate range when loading
/// pairs of GPRs (`ldp x0, x1, [x18, #0]`, etc.).
#[repr(C, align(16))]
#[derive(Debug)]
pub struct ThreadState {
    /// Guest GPRs: x0..x30. x18 is reserved for Chimera's context pointer
    /// and its slot here is never read or written by the trampolines or by
    /// translated code — Darwin's platform ABI keeps guest code out of x18.
    pub regs: [u64; 31],
    /// Pad to bring `fpstate` to a 16-byte alignment.
    pub _pad: u64,
    /// FPSIMD register file: 32 Q-registers (512 bytes) followed by FPSR
    /// and FPCR (8 bytes). Reached as a single base plus 16-byte-aligned
    /// pair offsets in the trampolines.
    pub fpstate: [u8; 520],
    /// Guest stack pointer; set on exit, read on entry.
    pub sp: u64,
    /// Guest program counter; set on exit, read on entry.
    pub pc: u64,
    /// Guest NZCV (the four condition flags occupying bits 31..28 of the
    /// NZCV system register). Saved/restored by the trampolines via
    /// `mrs/msr nzcv`.
    pub nzcv: u64,
    /// Chimera's stack pointer, saved on entry and restored on exit.
    pub chimera_sp: u64,
    /// Host PC for the next entry; used by `dispatch` after it has already
    /// loaded the guest GPRs and has no free register to hold the target.
    pub host_pc_target: u64,
    /// Why the last exit happened. Read by the run loop after every entry,
    /// reset to `BLOCK` before each entry.
    pub exit_kind: u64,
}

impl Default for ThreadState {
    fn default() -> Self {
        Self {
            regs: [0; 31],
            _pad: 0,
            fpstate: [0; 520],
            sp: 0,
            pc: 0,
            nzcv: 0,
            chimera_sp: 0,
            host_pc_target: 0,
            exit_kind: 0,
        }
    }
}

/// Common guest register indices.
pub const X0: usize = 0;
pub const X1: usize = 1;
pub const X2: usize = 2;
pub const X3: usize = 3;
pub const X4: usize = 4;
pub const X5: usize = 5;
pub const X16: usize = 16;

/// Initial guest register state at dispatch time. `pc` and `sp` are the
/// only required values; `regs` lets the caller seed argument registers
/// (x0..x7), the link register (x30), or anything else the entry point
/// expects to find. Unused slots stay zero.
pub struct InitialState {
    pub pc: u64,
    pub sp: u64,
    pub regs: [u64; 31],
}

impl InitialState {
    pub fn new(pc: u64, sp: u64) -> Self {
        Self {
            pc,
            sp,
            regs: [0; 31],
        }
    }
}

/// Run the guest, starting from `state.pc` with the register file primed
/// from `state`. Returns the guest's exit code:
///   - when the guest issues Darwin's `exit` (BSD #1) — the syscall isn't
///     forwarded to the host kernel (that would terminate Chimera);
///   - when the guest's entry function returns through `pc == 0`, with
///     `x0` taken as the exit code (the typical path for a `main`
///     returning to a NULL link register).
pub fn start_thread(state: InitialState, mut handler: Box<dyn SystemCalls>) -> Result<i32, Error> {
    let mut cache = CodeCache::new()?;
    let mut map: HashMap<u64, u64> = HashMap::new();
    let mut ts = Box::new(ThreadState::default());

    let ts_ptr: *mut ThreadState = &mut *ts as *mut ThreadState;
    unsafe {
        (*ts_ptr).sp = state.sp;
        (*ts_ptr).pc = state.pc;
        (*ts_ptr).regs = state.regs;
    }
    CHIMERA_CTX_PTR.store(ts_ptr as u64, Ordering::Relaxed);
    let block_exit = exit_block as *const () as u64;
    // SVC sites jump through `exit_syscall_no_stack`, which expects the
    // per-block stub to have saved x16 into ctx directly and not touched
    // the guest stack. The original `exit_syscall` (which pops x16/x17
    // off the guest stack) is unused at translation time but kept around
    // for symmetry with `exit_block`.
    let _ = exit_syscall as *const () as u64;
    let syscall_exit = exit_syscall_no_stack as *const () as u64;

    let trace = std::env::var("CHIMERA_TRACE").is_ok();
    let trace_each = std::env::var("CHIMERA_TRACE_EACH").is_ok();
    loop {
        let pc = unsafe { (*ts_ptr).pc };
        // The guest's `main` returns through `ret`, which jumps to the link
        // register (x30). When we set up the initial state with x30 = 0,
        // main's return lands here. Treat that as a clean exit with x0 as
        // the status code, the way crt's start glue would have called
        // `exit(retval)` natively.
        if pc == 0 {
            let code = unsafe { (*ts_ptr).regs[X0] } as i32;
            return Ok(code);
        }
        let host_pc = match map.get(&pc) {
            Some(&hpc) => hpc,
            None => {
                if trace {
                    eprintln!("chimera: translate guest_pc={:#x}", pc);
                    use std::io::Write;
                    let _ = std::io::stderr().flush();
                }
                let hpc = translate(&mut cache, pc, block_exit, syscall_exit)
                    .unwrap_or_else(|e| panic!("translate failed at {:#x}: {}", pc, e));
                map.insert(pc, hpc);
                hpc
            }
        };
        if trace_each {
            eprintln!(
                "chimera: enter guest_pc={:#x} host_pc={:#x} x0={:#x} sp={:#x}",
                pc,
                host_pc,
                unsafe { (*ts_ptr).regs[X0] },
                unsafe { (*ts_ptr).sp },
            );
            use std::io::Write;
            let _ = std::io::stderr().flush();
        }
        unsafe {
            (*ts_ptr).exit_kind = EXIT_KIND_BLOCK;
            dispatch(ts_ptr, host_pc);
        }
        if unsafe { (*ts_ptr).exit_kind } == EXIT_KIND_SYSCALL {
            if trace {
                let regs = unsafe { &(*ts_ptr).regs };
                eprintln!(
                    "chimera: syscall #{:#x} x0={:#x} x1={:#x} x2={:#x} pc={:#x}",
                    regs[X16],
                    regs[X0],
                    regs[X1],
                    regs[X2],
                    unsafe { (*ts_ptr).pc }
                );
            }
            if let Some(code) = handle_syscall(ts_ptr, handler.as_mut()) {
                return Ok(code);
            }
            if trace {
                let regs = unsafe { &(*ts_ptr).regs };
                eprintln!(
                    "chimera: returned from syscall, x0={:#x} nzcv={:#x}",
                    regs[X0],
                    unsafe { (*ts_ptr).nzcv }
                );
            }
        }
    }
}

/// Bit position of the carry flag inside NZCV (bit 29).
const NZCV_CARRY: u64 = 1 << 29;

fn handle_syscall(ts_ptr: *mut ThreadState, handler: &mut dyn SystemCalls) -> Option<i32> {
    let regs = unsafe { &(*ts_ptr).regs };
    // Darwin/arm64: syscall number in x16, args in x0..x5, return in x0.
    let number = regs[X16];
    let args = [regs[X0], regs[X1], regs[X2], regs[X3], regs[X4], regs[X5]];
    // BSD `exit` is syscall #1.
    let exit_code = if number == 1 {
        Some(args[0] as i32)
    } else {
        None
    };
    let mut call = SystemCall::new(number, args);
    handler.handle(&mut call);
    unsafe {
        (*ts_ptr).regs[X0] = call.return_value() as u64;
        // Darwin signals errors via the NZCV carry flag rather than a
        // negative return value. The handler reports its intent through
        // `SystemCall::set_error`; translate that to the bit the guest
        // expects to see when it resumes.
        if call.is_error() {
            (*ts_ptr).nzcv |= NZCV_CARRY;
        } else {
            (*ts_ptr).nzcv &= !NZCV_CARRY;
        }
    }
    exit_code
}
