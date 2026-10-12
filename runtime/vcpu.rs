//! The boundary between Chimera's kernel and the mechanism that executes guest
//! instructions.
//!
//! The kernel — the host's process semantics: system calls, signals, threads,
//! `execve` — owns each guest thread's register file and drives it the way a
//! kernel drives a CPU: it calls [`Vcpu::run`], which executes guest code until
//! the guest needs the kernel, and handles the [`Exit`] it returns before
//! running it again. The kernel reads and writes guest state only through the
//! register file, so it is written once for every `Vcpu`. There is one kernel
//! per host (`sys::linux::thread`, `sys::darwin::thread`) and one translating
//! `Vcpu` per backend (`arch::x86::dispatch`, `arch::arm64::dispatch`).

use crate::{Error, arch::dispatch::ThreadState};

/// Why [`Vcpu::run`] returned. The register file describes the guest at the
/// point of the exit.
pub enum Exit {
    /// The guest executed its system-call instruction (`syscall`, `svc`). The
    /// program counter is past the instruction, and the number and arguments
    /// are in the guest registers.
    Syscall,
    /// The guest executed a breakpoint (`int3`, `brk`). On x86-64 the program
    /// counter is past the instruction; on arm64 it is at it, which is where
    /// the hardware leaves it.
    Breakpoint,
    /// The guest branched to an address that cannot be fetched. The program
    /// counter is that address.
    FetchFault,
    /// The guest stopped at an instruction boundary with nothing to report.
    /// The kernel uses it to deliver pending signals and observe process-wide
    /// stops before running the guest again.
    Safepoint,
}

/// A virtual CPU executing one guest thread.
pub trait Vcpu: Send {
    /// Prepare to run on the calling host thread. Called each time a host
    /// thread starts running the guest, before the first [`Vcpu::run`].
    fn attach(&mut self, state: &mut ThreadState) -> Result<(), Error>;

    /// Run guest code from `state` until the guest needs the kernel.
    fn run(&mut self, state: &mut ThreadState) -> Exit;
}
