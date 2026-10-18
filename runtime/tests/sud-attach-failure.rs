//! A sandbox whose syscall user dispatch cannot be armed fails without
//! leaving the program's thread on Chimera's alternate signal stack. A
//! seccomp filter makes `prctl(PR_SET_SYSCALL_USER_DISPATCH)` fail with
//! `EINVAL`, as it does on a kernel older than 5.11.
//!
//! The test runs without the libtest harness, so the sandbox runs on the
//! process's main thread, whose alternate stack is the one checked.

use std::{mem, process::ExitCode, ptr};

use chimera::{Backend, Sandbox};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const PR_SET_SYSCALL_USER_DISPATCH: u32 = 59;

/// Make `prctl(PR_SET_SYSCALL_USER_DISPATCH, ...)` fail with `EINVAL`.
fn refuse_dispatch() {
    let stmt = |code: u32, k: u32| libc::sock_filter {
        code: code as u16,
        jt: 0,
        jf: 0,
        k,
    };
    let jeq = |k: u32, jf: u8| libc::sock_filter {
        code: (libc::BPF_JMP | libc::BPF_JEQ | libc::BPF_K) as u16,
        jt: 0,
        jf,
        k,
    };
    let ld = libc::BPF_LD | libc::BPF_W | libc::BPF_ABS;
    let ret = libc::BPF_RET | libc::BPF_K;
    let mut filter = [
        stmt(ld, 0), // seccomp_data.nr
        jeq(libc::SYS_prctl as u32, 3),
        stmt(ld, 16), // seccomp_data.args[0], low half
        jeq(PR_SET_SYSCALL_USER_DISPATCH, 1),
        stmt(ret, libc::SECCOMP_RET_ERRNO | libc::EINVAL as u32),
        stmt(ret, libc::SECCOMP_RET_ALLOW),
    ];
    let prog = libc::sock_fprog {
        len: filter.len() as u16,
        filter: filter.as_mut_ptr(),
    };
    unsafe {
        assert_eq!(libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), 0);
        assert_eq!(
            libc::prctl(
                libc::PR_SET_SECCOMP,
                libc::SECCOMP_MODE_FILTER,
                &prog as *const libc::sock_fprog
            ),
            0
        );
    }
}

fn main() -> ExitCode {
    let mut stack = vec![0u8; libc::SIGSTKSZ];
    let ss = libc::stack_t {
        ss_sp: stack.as_mut_ptr().cast(),
        ss_flags: 0,
        ss_size: stack.len(),
    };
    unsafe { libc::sigaltstack(&ss, ptr::null_mut()) };
    refuse_dispatch();

    let mut sandbox = Sandbox::new("/bin/true").expect("sandbox");
    sandbox.backend(Backend::SyscallUserDispatch);
    if sandbox.run().is_ok() {
        eprintln!("the sandbox ran although dispatch could not be armed");
        return ExitCode::FAILURE;
    }
    let mut now: libc::stack_t = unsafe { mem::zeroed() };
    unsafe { libc::sigaltstack(ptr::null(), &mut now) };
    if now.ss_sp != ss.ss_sp || now.ss_size != ss.ss_size {
        eprintln!(
            "alternate stack is {:p}+{:#x}, not the program's {:p}+{:#x}",
            now.ss_sp, now.ss_size, ss.ss_sp, ss.ss_size
        );
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
