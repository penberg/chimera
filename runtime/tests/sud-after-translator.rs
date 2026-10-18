//! Syscall user dispatch takes the interrupt signal's host slot back from a
//! guest the translator ran before it. That guest owned the slot and set it
//! to `SIG_IGN`; a dispatch guest run afterwards with the slot left that way
//! could never be interrupted, and an `execve` waiting on a sibling that
//! spins without system calls would hang.
//!
//! The test runs without the libtest harness, so no other threads share the
//! process's signal dispositions.

use std::{mem, process::ExitCode, ptr};

use chimera::{Backend, Sandbox};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn run(backend: Backend, args: &[&str]) -> bool {
    let mut sandbox = Sandbox::new(args[0]).expect("sandbox");
    sandbox.backend(backend);
    sandbox.args(&args[1..]);
    sandbox.run().expect("run").code() == 0
}

fn interrupt_ignored() -> bool {
    let mut sa: libc::sigaction = unsafe { mem::zeroed() };
    unsafe { libc::sigaction(libc::SIGRTMAX(), ptr::null(), &mut sa) };
    sa.sa_sigaction == libc::SIG_IGN
}

fn main() -> ExitCode {
    if !run(Backend::SyscallUserDispatch, &["/bin/true"]) {
        eprintln!("first dispatch run failed");
        return ExitCode::FAILURE;
    }
    if !run(Backend::Translator, &["/bin/bash", "-c", "trap '' RTMAX"]) {
        eprintln!("translator run failed");
        return ExitCode::FAILURE;
    }
    if !interrupt_ignored() {
        eprintln!("the translated guest's SIG_IGN did not reach the host slot");
        return ExitCode::FAILURE;
    }
    if !run(Backend::SyscallUserDispatch, &["/bin/true"]) {
        eprintln!("second dispatch run failed");
        return ExitCode::FAILURE;
    }
    if interrupt_ignored() {
        eprintln!("the interrupt signal is still ignored under dispatch");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
