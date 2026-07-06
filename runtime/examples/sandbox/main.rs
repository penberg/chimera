//! examples/sandbox: an allowlist-based system-call sandbox.
//!
//! Usage:
//!   cargo run --example sandbox -- --allow <regex> [--allow <regex>...] <program> [args...]
//!
//! Every guest syscall is matched against the supplied `--allow` regex
//! patterns. Matches are forwarded to the host kernel; anything else
//! returns `-EPERM` to the guest. The first denial of each distinct
//! syscall is logged to stderr so the policy can be built up
//! incrementally — repeated denials of the same call are silent.
//!
//! Example:
//!   cargo run --example sandbox -- --allow '^(read|write|brk|mprotect)$' /bin/echo hi
//!
//! This is the smallest non-trivial use of Chimera's `SystemCalls`
//! trait: classify by name and either forward or refuse. The `strace`
//! example shows the more elaborate side, with per-syscall argument
//! decoding.

use std::{collections::HashSet, path::PathBuf, process::ExitCode, sync::Mutex};

use argh::FromArgs;
use mimalloc::MiMalloc;
use regex::Regex;

use chimera::{Sandbox, SyscallResult, SystemCall, SystemCalls, host_syscall};

/// Route this embedder's allocations through mimalloc's `mmap`-backed
/// segments, keeping Chimera's heap off the guest libc's shared `brk`.
/// A `#[global_allocator]` is per-binary, so every embedder needs its own.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// A demo sandbox: allow only syscalls whose name matches one of the
/// supplied regex patterns; deny everything else with EPERM.
#[derive(FromArgs)]
struct Opts {
    /// system-call name regex to allow (repeatable)
    #[argh(option)]
    allow: Vec<String>,

    /// program to run
    #[argh(positional)]
    program: PathBuf,

    /// arguments to pass to the guest
    #[argh(positional)]
    args: Vec<String>,
}

fn main() -> ExitCode {
    let opts: Opts = argh::from_env();

    let allowed: Vec<Regex> = match opts.allow.iter().map(|p| Regex::new(p)).collect() {
        Ok(rs) => rs,
        Err(e) => {
            eprintln!("sandbox: invalid --allow regex: {e}");
            return ExitCode::from(2);
        }
    };

    let mut sandbox = match Sandbox::new(&opts.program) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("sandbox: {e}");
            return ExitCode::FAILURE;
        }
    };
    sandbox
        .args(opts.args)
        .system_calls(Allowlist::new(allowed));

    match sandbox.run() {
        Ok(status) => ExitCode::from(status.code() as u8),
        Err(e) => {
            eprintln!("sandbox: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Allowlist {
    allowed: Vec<Regex>,
    // A single handler serves every guest thread (`SystemCalls` is `&self`), so
    // the "first denial seen" set needs interior mutability.
    denied_seen: Mutex<HashSet<u64>>,
}

impl Allowlist {
    fn new(allowed: Vec<Regex>) -> Self {
        Self {
            allowed,
            denied_seen: Mutex::new(HashSet::new()),
        }
    }
}

impl SystemCalls for Allowlist {
    fn do_syscall(&self, call: &mut SystemCall) {
        let name = syscall_name(call.number);
        // Unknown numbers serialize as `syscall_<n>` so a user's regex
        // can never let one through by accident.
        let display = if name.is_empty() {
            format!("syscall_{}", call.number)
        } else {
            name.to_string()
        };

        if self.allowed.iter().any(|r| r.is_match(&display)) {
            call.set_result(host_syscall(call));
        } else {
            if self.denied_seen.lock().unwrap().insert(call.number) {
                eprintln!("sandbox: denied {}", display);
            }
            call.set_result(SyscallResult::Error(libc::EPERM));
        }
    }
}

fn syscall_name(n: u64) -> &'static str {
    match n {
        0 => "read",
        1 => "write",
        2 => "open",
        3 => "close",
        4 => "stat",
        5 => "fstat",
        6 => "lstat",
        7 => "poll",
        8 => "lseek",
        9 => "mmap",
        10 => "mprotect",
        11 => "munmap",
        12 => "brk",
        13 => "rt_sigaction",
        14 => "rt_sigprocmask",
        15 => "rt_sigreturn",
        16 => "ioctl",
        17 => "pread64",
        18 => "pwrite64",
        19 => "readv",
        20 => "writev",
        21 => "access",
        22 => "pipe",
        23 => "select",
        24 => "sched_yield",
        25 => "mremap",
        28 => "madvise",
        32 => "dup",
        33 => "dup2",
        39 => "getpid",
        56 => "clone",
        57 => "fork",
        58 => "vfork",
        59 => "execve",
        60 => "exit",
        61 => "wait4",
        62 => "kill",
        63 => "uname",
        72 => "fcntl",
        78 => "getdents",
        79 => "getcwd",
        80 => "chdir",
        81 => "fchdir",
        82 => "rename",
        83 => "mkdir",
        84 => "rmdir",
        85 => "creat",
        86 => "link",
        87 => "unlink",
        89 => "readlink",
        102 => "getuid",
        104 => "getgid",
        107 => "geteuid",
        108 => "getegid",
        110 => "getppid",
        158 => "arch_prctl",
        186 => "gettid",
        202 => "futex",
        217 => "getdents64",
        218 => "set_tid_address",
        228 => "clock_gettime",
        231 => "exit_group",
        257 => "openat",
        262 => "newfstatat",
        273 => "set_robust_list",
        302 => "prlimit64",
        318 => "getrandom",
        334 => "rseq",
        _ => "",
    }
}
