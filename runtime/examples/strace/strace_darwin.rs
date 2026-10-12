//! The Darwin tracer: arm64 BSD syscall numbers and Mach traps. It prints
//! the syscall name and its first three argument registers in hex — enough
//! to scan a trace by eye without re-implementing every BSD/Mach struct
//! decoder — and decodes the one family whose pointer says nothing on its
//! own, the `getattrlist` attribute bitmaps.

use std::{env, fmt::Write, process::ExitCode};

use chimera::{Sandbox, SyscallResult, SystemCall, SystemCalls};

pub fn main() -> ExitCode {
    let mut argv = env::args_os().skip(1);
    let program = match argv.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: strace <program> [args...]");
            return ExitCode::from(2);
        }
    };
    let prog_args: Vec<_> = argv.collect();

    let mut sandbox = match Sandbox::new(&program) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("strace: {e}");
            return ExitCode::FAILURE;
        }
    };
    sandbox.args(prog_args).system_calls(Strace);

    match sandbox.run() {
        Ok(status) => ExitCode::from(status.code() as u8),
        Err(e) => {
            eprintln!("strace: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Strace;

// The Darwin path prints the syscall name and its first three argument
// registers in hex — enough to scan a trace by eye without re-implementing
// every BSD/Mach struct decoder. The Linux path keeps the rich decoders.
impl SystemCalls for Strace {
    fn post_syscall(&self, call: &SystemCall) {
        let a = &call.args;
        let mut line = String::with_capacity(48);
        match syscall_name(call.number) {
            "syscall" => {
                let _ = write!(line, "syscall#{}(", call.number as i64);
            }
            name => {
                let _ = write!(line, "{name}(");
            }
        }
        let _ = write!(line, "{:#x}, {:#x}, {:#x})", a[0], a[1], a[2]);
        // The attrlist family: show the five requested bitmaps, which
        // say what a caller wants far better than a pointer does.
        if let Some(list) = match call.number {
            220 | 221 | 228 | 229 | 461 => Some(a[1]),
            476 | 524 => Some(a[2]),
            _ => None,
        } && list != 0
        {
            let words = unsafe { std::ptr::read_unaligned(list as *const [u32; 6]) };
            let _ = write!(
                line,
                " attrs[cmn={:#x} vol={:#x} dir={:#x} file={:#x} fork={:#x}] opts={:#x}",
                words[1], words[2], words[3], words[4], words[5], a[4]
            );
        }
        match call.result() {
            Some(SyscallResult::Ok(v)) => eprintln!("{line:<48} = {v:#x}"),
            Some(SyscallResult::Error(e)) => eprintln!("{line:<48} = -{e}"),
            None => eprintln!("{line:<48} = ?"),
        }
    }
}

/// Darwin/arm64 syscall names. Positive numbers are BSD syscalls; values
/// with bit 31 set are Mach traps (the kernel reads x16 as a signed 32-bit
/// value, so they appear here as `0xFFFF_FFFF_FFFF_FFxx`). A common subset;
/// anything unrecognized prints as `syscall`.
fn syscall_name(n: u64) -> &'static str {
    match n {
        1 => "exit",
        2 => "fork",
        3 => "read",
        4 => "write",
        5 => "open",
        6 => "close",
        7 => "wait4",
        9 => "link",
        10 => "unlink",
        12 => "chdir",
        13 => "fchdir",
        14 => "mknod",
        15 => "chmod",
        16 => "chown",
        20 => "getpid",
        24 => "getuid",
        33 => "access",
        34 => "chflags",
        35 => "fchflags",
        37 => "kill",
        39 => "getppid",
        41 => "dup",
        42 => "pipe",
        46 => "sigaction",
        48 => "sigprocmask",
        53 => "sigaltstack",
        54 => "ioctl",
        57 => "symlink",
        58 => "readlink",
        59 => "execve",
        73 => "munmap",
        74 => "mprotect",
        75 => "madvise",
        90 => "dup2",
        92 => "fcntl",
        95 => "fsync",
        120 => "readv",
        121 => "writev",
        123 => "fchown",
        124 => "fchmod",
        128 => "rename",
        131 => "flock",
        132 => "mkfifo",
        136 => "mkdir",
        137 => "rmdir",
        138 => "utimes",
        139 => "futimes",
        153 => "pread",
        154 => "pwrite",
        169 => "csops",
        170 => "csops_audittoken",
        187 => "fdatasync",
        188 => "stat",
        189 => "fstat",
        190 => "lstat",
        191 => "pathconf",
        192 => "fpathconf",
        196 => "getdirentries",
        197 => "mmap",
        199 => "lseek",
        200 => "truncate",
        201 => "ftruncate",
        202 => "sysctl",
        220 => "getattrlist",
        221 => "setattrlist",
        228 => "fgetattrlist",
        229 => "fsetattrlist",
        234 => "getxattr",
        235 => "fgetxattr",
        236 => "setxattr",
        237 => "fsetxattr",
        238 => "removexattr",
        239 => "fremovexattr",
        240 => "listxattr",
        241 => "flistxattr",
        244 => "posix_spawn",
        336 => "proc_info",
        338 => "stat64",
        339 => "fstat64",
        340 => "lstat64",
        344 => "getdirentries64",
        345 => "statfs64",
        346 => "fstatfs64",
        347 => "getfsstat64",
        360 => "bsdthread_create",
        361 => "bsdthread_terminate",
        364 => "lchown",
        366 => "bsdthread_register",
        372 => "thread_selfid",
        396 => "read_nocancel",
        397 => "write_nocancel",
        398 => "open_nocancel",
        399 => "close_nocancel",
        406 => "fcntl_nocancel",
        408 => "fsync_nocancel",
        411 => "readv_nocancel",
        412 => "writev_nocancel",
        414 => "pread_nocancel",
        415 => "pwrite_nocancel",
        427 => "fsgetpath",
        461 => "getattrlistbulk",
        462 => "clonefileat",
        463 => "openat",
        464 => "openat_nocancel",
        465 => "renameat",
        466 => "faccessat",
        467 => "fchmodat",
        468 => "fchownat",
        469 => "fstatat",
        470 => "fstatat64",
        471 => "linkat",
        472 => "unlinkat",
        473 => "readlinkat",
        474 => "symlinkat",
        475 => "mkdirat",
        476 => "getattrlistat",
        488 => "renameatx_np",
        515 => "ulock_wait",
        516 => "ulock_wake",
        517 => "fclonefileat",
        520 => "terminate_with_payload",
        521 => "abort_with_payload",
        524 => "setattrlistat",
        540 => "preadv",
        541 => "pwritev",
        551 => "freadlink",
        553 => "mkfifoat",
        554 => "mknodat",
        0x8000_0000 => "thread_set_tsd_base",
        0xFFFF_FFFF_FFFF_FFE4 => "task_self_trap",
        0xFFFF_FFFF_FFFF_FFE3 => "host_self_trap",
        0xFFFF_FFFF_FFFF_FFE6 => "mach_reply_port",
        0xFFFF_FFFF_FFFF_FFE1 => "mach_msg_trap",
        0xFFFF_FFFF_FFFF_FFF1 => "_kernelrpc_mach_vm_protect_trap",
        0xFFFF_FFFF_FFFF_FFF2 => "_kernelrpc_mach_vm_deallocate_trap",
        0xFFFF_FFFF_FFFF_FFF4 => "_kernelrpc_mach_vm_allocate_trap",
        0xFFFF_FFFF_FFFF_FFF6 => "_kernelrpc_mach_vm_map_trap",
        0xFFFF_FFFF_FFFF_FFED => "_kernelrpc_mach_port_deallocate_trap",
        _ => "syscall",
    }
}
