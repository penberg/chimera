//! A program that runs a guest under syscall user dispatch gets its own
//! SIGSYS and SIGTRAP handlers back once the sandbox has finished. The
//! backend takes over those host slots while guest threads run; left
//! installed afterwards, a SIGTRAP the program raises itself reaches
//! Chimera's handler, which reads the freed state of a thread that is gone.
//!
//! The test runs without the libtest harness, so the sandbox runs on the
//! process's main thread with no other threads around.

use std::{
    process::ExitCode,
    sync::atomic::{AtomicI32, Ordering},
};

use chimera::{Backend, Sandbox};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

static CAUGHT: AtomicI32 = AtomicI32::new(0);

extern "C" fn catch(signo: libc::c_int) {
    CAUGHT.store(signo, Ordering::Relaxed);
}

fn handled_by_program(signo: libc::c_int) -> bool {
    CAUGHT.store(0, Ordering::Relaxed);
    unsafe { libc::raise(signo) };
    CAUGHT.load(Ordering::Relaxed) == signo
}

fn main() -> ExitCode {
    for signo in [libc::SIGSYS, libc::SIGTRAP] {
        unsafe { libc::signal(signo, catch as *const () as libc::sighandler_t) };
    }
    for run in 1..=2 {
        let mut sandbox = Sandbox::new("/bin/true").expect("sandbox");
        sandbox.backend(Backend::SyscallUserDispatch);
        let status = sandbox.run().expect("run");
        if status.code() != 0 {
            eprintln!("run {run}: /bin/true exited {}", status.code());
            return ExitCode::FAILURE;
        }
        for signo in [libc::SIGSYS, libc::SIGTRAP] {
            if !handled_by_program(signo) {
                eprintln!("run {run}: signal {signo} did not reach the program's handler");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
