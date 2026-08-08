//! examples/strace: print every system call a guest program makes.
//!
//! Usage: `cargo run --example strace -- <program> [args...]`
//!
//! The handler runs in Chimera's address space, so guest pointers can be
//! dereferenced directly when formatting arguments. Output is written to
//! stderr, with the syscall-name-and-args column padded so return values
//! line up the way `strace(1)` does.
//!
//! The handler implements [`SystemCalls`] by logging in
//! `post_syscall(&SystemCall)`. The `SystemCall` value carries the number,
//! argument registers, and the final result when one exists.
//!
//! The argument decoders are the rich ones: signal names, open flags,
//! struct stat, and so on.

use mimalloc::MiMalloc;

/// Route this embedder's allocations through mimalloc's `mmap`-backed
/// segments, keeping the embedder's heap traffic out of the process's `brk`
/// segment. A `#[global_allocator]` is per-binary, so every embedder needs
/// its own.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use linux::main;

mod linux {

    use std::{env, fmt::Write, process::ExitCode, ptr};

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

    impl SystemCalls for Strace {
        fn post_syscall(&self, call: &SystemCall) {
            let name = syscall_name(call.number);
            match call.result() {
                Some(result) => {
                    let ret = match result {
                        SyscallResult::Ok(v) => v,
                        SyscallResult::Error(e) => -(e as i64),
                    };
                    eprintln!(
                        "{:<40} = {}",
                        format_call(name, call, ret),
                        format_ret(name, result),
                    );
                }
                None => {
                    eprintln!("{:<40} = ?", format_call(name, call, 0));
                }
            }
        }
    }

    fn format_call(name: &str, c: &SystemCall, ret: i64) -> String {
        let mut line = String::with_capacity(64);
        let _ = write!(line, "{}(", name);
        format_args(&mut line, name, c, ret);
        line.push(')');
        line
    }

    // === Argument formatting ===

    fn format_args(out: &mut String, name: &str, c: &SystemCall, ret: i64) {
        let a = &c.args;
        match name {
            "read" => {
                // The kernel fills the buffer; decode it on success.
                let buf = if ret > 0 {
                    quoted_buf(a[1], ret as usize)
                } else {
                    ptr(a[1])
                };
                let _ = write!(out, "{}, {}, {}", a[0] as i32, buf, a[2]);
            }
            "pread64" => {
                let buf = if ret > 0 {
                    quoted_buf(a[1], ret as usize)
                } else {
                    ptr(a[1])
                };
                let _ = write!(out, "{}, {}, {}, {}", a[0] as i32, buf, a[2], a[3] as i64);
            }
            "write" => {
                let _ = write!(
                    out,
                    "{}, {}, {}",
                    a[0] as i32,
                    quoted_buf(a[1], a[2] as usize),
                    a[2],
                );
            }
            "open" => {
                let flags = a[1] as i32;
                if flags & (libc::O_CREAT | libc::O_TMPFILE) != 0 {
                    let _ = write!(
                        out,
                        "{}, {}, {:#o}",
                        cstr(a[0]),
                        open_flags(flags),
                        a[2] as u32,
                    );
                } else {
                    let _ = write!(out, "{}, {}", cstr(a[0]), open_flags(flags));
                }
            }
            "openat" => {
                let flags = a[2] as i32;
                if flags & (libc::O_CREAT | libc::O_TMPFILE) != 0 {
                    let _ = write!(
                        out,
                        "{}, {}, {}, {:#o}",
                        dirfd(a[0] as i32),
                        cstr(a[1]),
                        open_flags(flags),
                        a[3] as u32,
                    );
                } else {
                    let _ = write!(
                        out,
                        "{}, {}, {}",
                        dirfd(a[0] as i32),
                        cstr(a[1]),
                        open_flags(flags),
                    );
                }
            }
            "close" | "dup" | "fchdir" | "fsync" | "fdatasync" => {
                let _ = write!(out, "{}", a[0] as i32);
            }
            "access" => {
                let _ = write!(out, "{}, {}", cstr(a[0]), access_mode(a[1] as i32));
            }
            "stat" | "lstat" => {
                let buf = if ret == 0 { stat_buf(a[1]) } else { ptr(a[1]) };
                let _ = write!(out, "{}, {}", cstr(a[0]), buf);
            }
            "unlink" | "chdir" | "rmdir" => {
                let _ = write!(out, "{}", cstr(a[0]));
            }
            "mkdir" => {
                let _ = write!(out, "{}, {:#o}", cstr(a[0]), a[1] as u32);
            }
            "readlink" => {
                let buf = if ret > 0 {
                    quoted_buf(a[1], ret as usize)
                } else {
                    ptr(a[1])
                };
                let _ = write!(out, "{}, {}, {}", cstr(a[0]), buf, a[2]);
            }
            "execve" => {
                let _ = write!(out, "{}, {}, {}", cstr(a[0]), ptr(a[1]), ptr(a[2]));
            }
            "fstat" => {
                let buf = if ret == 0 { stat_buf(a[1]) } else { ptr(a[1]) };
                let _ = write!(out, "{}, {}", a[0] as i32, buf);
            }
            "newfstatat" => {
                let buf = if ret == 0 { stat_buf(a[2]) } else { ptr(a[2]) };
                let _ = write!(
                    out,
                    "{}, {}, {}, {}",
                    dirfd(a[0] as i32),
                    cstr(a[1]),
                    buf,
                    at_flags(a[3] as i32),
                );
            }
            "lseek" => {
                let _ = write!(out, "{}, {}, {}", a[0] as i32, a[1] as i64, a[2] as i32);
            }
            "mmap" => {
                let offset = a[5];
                let offset_str = if offset == 0 {
                    "0".to_string()
                } else {
                    format!("{:#x}", offset)
                };
                let _ = write!(
                    out,
                    "{}, {}, {}, {}, {}, {}",
                    ptr(a[0]),
                    a[1],
                    mmap_prot(a[2] as i32),
                    mmap_flags(a[3] as i32),
                    a[4] as i32,
                    offset_str,
                );
            }
            "mprotect" => {
                let _ = write!(out, "{}, {}, {}", ptr(a[0]), a[1], mmap_prot(a[2] as i32));
            }
            "munmap" => {
                let _ = write!(out, "{}, {}", ptr(a[0]), a[1]);
            }
            "brk" => {
                let _ = write!(out, "{}", ptr(a[0]));
            }
            "ioctl" | "fcntl" => {
                let _ = write!(out, "{}, {:#x}, {:#x}", a[0] as i32, a[1], a[2]);
            }
            "arch_prctl" => {
                let _ = write!(out, "{}, {}", arch_prctl_op(a[0] as i32), ptr(a[1]));
            }
            "rt_sigaction" => {
                let _ = write!(
                    out,
                    "{}, {}, {}, {}",
                    signal_name(a[0] as i32),
                    ptr(a[1]),
                    ptr(a[2]),
                    a[3],
                );
            }
            "rt_sigprocmask" => {
                let _ = write!(
                    out,
                    "{}, {}, {}, {}",
                    a[0] as i32,
                    ptr(a[1]),
                    ptr(a[2]),
                    a[3],
                );
            }
            "exit" | "exit_group" => {
                let _ = write!(out, "{}", a[0] as i32);
            }
            "set_tid_address" => {
                let _ = write!(out, "{}", ptr(a[0]));
            }
            "set_robust_list" => {
                let _ = write!(out, "{}, {}", ptr(a[0]), a[1]);
            }
            "rseq" => {
                let _ = write!(out, "{}, {:#x}, {}, {:#x}", ptr(a[0]), a[1], a[2], a[3],);
            }
            "getpid" | "getppid" | "getuid" | "geteuid" | "getgid" | "getegid" | "gettid"
            | "sched_yield" => {
                // No arguments.
            }
            "prlimit64" => {
                let new_lim = ptr(a[2]);
                let old_lim = if a[3] != 0 && ret == 0 {
                    rlimit_buf(a[3])
                } else {
                    ptr(a[3])
                };
                let _ = write!(
                    out,
                    "{}, {}, {}, {}",
                    a[0] as i32,
                    rlimit_name(a[1] as i32),
                    new_lim,
                    old_lim,
                );
            }
            "getrandom" => {
                let _ = write!(out, "{}, {}, {:#x}", ptr(a[0]), a[1], a[2]);
            }
            _ => {
                for (i, &x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{:#x}", x);
                }
            }
        }
    }

    fn format_ret(name: &str, result: SyscallResult) -> String {
        match result {
            SyscallResult::Error(errno) => {
                format!("-1 {} ({})", errno_name(errno), errno_text(errno))
            }
            SyscallResult::Ok(v) => match name {
                "mmap" | "brk" => format!("{:#x}", v as u64),
                _ => v.to_string(),
            },
        }
    }

    // === Guest-memory helpers ===

    /// Format a guest pointer as `NULL` if zero, otherwise as hex.
    fn ptr(addr: u64) -> String {
        if addr == 0 {
            "NULL".to_string()
        } else {
            format!("{:#x}", addr)
        }
    }

    /// Decode a kernel `struct stat` at `ptr` into `{st_mode=..., st_size=...}`.
    /// The layout is the x86_64 kernel ABI (asm/stat.h): st_mode at offset 24,
    /// st_size at offset 48.
    fn stat_buf(p: u64) -> String {
        if p == 0 {
            return "NULL".to_string();
        }
        let mode = unsafe { ptr::read_volatile((p as usize + 24) as *const u32) };
        let size = unsafe { ptr::read_volatile((p as usize + 48) as *const i64) };
        let kind = match mode & 0o170000 {
            0o100000 => "S_IFREG",
            0o040000 => "S_IFDIR",
            0o120000 => "S_IFLNK",
            0o020000 => "S_IFCHR",
            0o060000 => "S_IFBLK",
            0o010000 => "S_IFIFO",
            0o140000 => "S_IFSOCK",
            _ => "S_IFUNK",
        };
        format!(
            "{{st_mode={}|{:04o}, st_size={}, ...}}",
            kind,
            mode & 0o7777,
            size,
        )
    }

    /// Decode a `struct rlimit64` at `ptr` into `{rlim_cur=..., rlim_max=...}`.
    fn rlimit_buf(p: u64) -> String {
        if p == 0 {
            return "NULL".to_string();
        }
        let cur = unsafe { ptr::read_volatile(p as *const u64) };
        let max = unsafe { ptr::read_volatile((p as usize + 8) as *const u64) };
        format!(
            "{{rlim_cur={}, rlim_max={}}}",
            rlim_value(cur),
            rlim_value(max),
        )
    }

    fn rlim_value(v: u64) -> String {
        if v == u64::MAX {
            "RLIM64_INFINITY".to_string()
        } else {
            v.to_string()
        }
    }

    /// Read up to `max` bytes from guest memory at `ptr`, stopping at NUL.
    /// Returns a strace-style quoted string. Chimera and the guest share an
    /// address space, so the read is a plain dereference.
    fn cstr(ptr: u64) -> String {
        if ptr == 0 {
            return "NULL".to_string();
        }
        let mut bytes = Vec::new();
        for i in 0..256 {
            let b = unsafe { ptr::read_volatile((ptr as usize + i) as *const u8) };
            if b == 0 {
                break;
            }
            bytes.push(b);
        }
        let mut out = String::with_capacity(bytes.len() + 2);
        out.push('"');
        push_quoted(&mut out, &bytes);
        out.push('"');
        if bytes.len() == 256 {
            out.push_str("...");
        }
        out
    }

    fn quoted_buf(ptr: u64, len: usize) -> String {
        if ptr == 0 {
            return "NULL".to_string();
        }
        let limit = len.min(32);
        let mut out = String::new();
        out.push('"');
        for i in 0..limit {
            let b = unsafe { ptr::read_volatile((ptr as usize + i) as *const u8) };
            push_quoted_byte(&mut out, b);
        }
        out.push('"');
        if len > limit {
            out.push_str("...");
        }
        out
    }

    fn push_quoted(out: &mut String, bytes: &[u8]) {
        for &b in bytes {
            push_quoted_byte(out, b);
        }
    }

    fn push_quoted_byte(out: &mut String, b: u8) {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0 => out.push_str("\\0"),
            0x20..=0x7e => out.push(b as char),
            _ => {
                let _ = write!(out, "\\x{:02x}", b);
            }
        }
    }

    // === Flag and constant decoding ===

    fn dirfd(fd: i32) -> String {
        if fd == libc::AT_FDCWD {
            "AT_FDCWD".to_string()
        } else {
            fd.to_string()
        }
    }

    fn access_mode(mode: i32) -> String {
        if mode == libc::F_OK {
            return "F_OK".to_string();
        }
        let mut parts = Vec::new();
        if mode & libc::R_OK != 0 {
            parts.push("R_OK");
        }
        if mode & libc::W_OK != 0 {
            parts.push("W_OK");
        }
        if mode & libc::X_OK != 0 {
            parts.push("X_OK");
        }
        if parts.is_empty() {
            format!("{:#x}", mode)
        } else {
            parts.join("|")
        }
    }

    fn arch_prctl_op(op: i32) -> String {
        match op {
            0x1001 => "ARCH_SET_GS".to_string(),
            0x1002 => "ARCH_SET_FS".to_string(),
            0x1003 => "ARCH_GET_FS".to_string(),
            0x1004 => "ARCH_GET_GS".to_string(),
            _ => format!("{:#x}", op),
        }
    }

    fn rlimit_name(r: i32) -> String {
        match r {
            0 => "RLIMIT_CPU",
            1 => "RLIMIT_FSIZE",
            2 => "RLIMIT_DATA",
            3 => "RLIMIT_STACK",
            4 => "RLIMIT_CORE",
            5 => "RLIMIT_RSS",
            6 => "RLIMIT_NPROC",
            7 => "RLIMIT_NOFILE",
            8 => "RLIMIT_MEMLOCK",
            9 => "RLIMIT_AS",
            10 => "RLIMIT_LOCKS",
            11 => "RLIMIT_SIGPENDING",
            12 => "RLIMIT_MSGQUEUE",
            13 => "RLIMIT_NICE",
            14 => "RLIMIT_RTPRIO",
            15 => "RLIMIT_RTTIME",
            _ => return r.to_string(),
        }
        .to_string()
    }

    fn at_flags(flags: i32) -> String {
        if flags == 0 {
            return "0".to_string();
        }
        let mut parts = Vec::new();
        for (bit, name) in &[
            (0x100, "AT_SYMLINK_NOFOLLOW"),
            (0x200, "AT_REMOVEDIR"),
            (0x400, "AT_SYMLINK_FOLLOW"),
            (0x800, "AT_NO_AUTOMOUNT"),
            (0x1000, "AT_EMPTY_PATH"),
        ] {
            if flags & bit != 0 {
                parts.push(*name);
            }
        }
        if parts.is_empty() {
            format!("{:#x}", flags)
        } else {
            parts.join("|")
        }
    }

    fn open_flags(flags: i32) -> String {
        let acc = match flags & 0b11 {
            0 => "O_RDONLY",
            1 => "O_WRONLY",
            2 => "O_RDWR",
            _ => "O_ACCMODE",
        };
        let mut parts = vec![acc.to_string()];
        for (bit, name) in &[
            (libc::O_CREAT, "O_CREAT"),
            (libc::O_EXCL, "O_EXCL"),
            (libc::O_NOCTTY, "O_NOCTTY"),
            (libc::O_TRUNC, "O_TRUNC"),
            (libc::O_APPEND, "O_APPEND"),
            (libc::O_NONBLOCK, "O_NONBLOCK"),
            (libc::O_DIRECTORY, "O_DIRECTORY"),
            (libc::O_CLOEXEC, "O_CLOEXEC"),
        ] {
            if flags & bit != 0 {
                parts.push((*name).to_string());
            }
        }
        parts.join("|")
    }

    fn mmap_prot(prot: i32) -> String {
        if prot == 0 {
            return "PROT_NONE".to_string();
        }
        let mut parts = Vec::new();
        if prot & libc::PROT_READ != 0 {
            parts.push("PROT_READ");
        }
        if prot & libc::PROT_WRITE != 0 {
            parts.push("PROT_WRITE");
        }
        if prot & libc::PROT_EXEC != 0 {
            parts.push("PROT_EXEC");
        }
        parts.join("|")
    }

    fn mmap_flags(flags: i32) -> String {
        let mut parts = Vec::new();
        if flags & libc::MAP_SHARED != 0 {
            parts.push("MAP_SHARED");
        }
        if flags & libc::MAP_PRIVATE != 0 {
            parts.push("MAP_PRIVATE");
        }
        if flags & libc::MAP_FIXED != 0 {
            parts.push("MAP_FIXED");
        }
        if flags & libc::MAP_ANONYMOUS != 0 {
            parts.push("MAP_ANONYMOUS");
        }
        if flags & libc::MAP_DENYWRITE != 0 {
            parts.push("MAP_DENYWRITE");
        }
        if flags & libc::MAP_STACK != 0 {
            parts.push("MAP_STACK");
        }
        if parts.is_empty() {
            format!("{:#x}", flags)
        } else {
            parts.join("|")
        }
    }

    fn signal_name(n: i32) -> String {
        let name = match n {
            1 => "SIGHUP",
            2 => "SIGINT",
            3 => "SIGQUIT",
            4 => "SIGILL",
            6 => "SIGABRT",
            8 => "SIGFPE",
            9 => "SIGKILL",
            10 => "SIGUSR1",
            11 => "SIGSEGV",
            12 => "SIGUSR2",
            13 => "SIGPIPE",
            14 => "SIGALRM",
            15 => "SIGTERM",
            17 => "SIGCHLD",
            18 => "SIGCONT",
            19 => "SIGSTOP",
            _ => return n.to_string(),
        };
        name.to_string()
    }

    fn errno_name(e: i32) -> &'static str {
        match e {
            libc::EPERM => "EPERM",
            libc::ENOENT => "ENOENT",
            libc::ESRCH => "ESRCH",
            libc::EINTR => "EINTR",
            libc::EIO => "EIO",
            libc::ENXIO => "ENXIO",
            libc::E2BIG => "E2BIG",
            libc::ENOEXEC => "ENOEXEC",
            libc::EBADF => "EBADF",
            libc::ECHILD => "ECHILD",
            libc::EAGAIN => "EAGAIN",
            libc::ENOMEM => "ENOMEM",
            libc::EACCES => "EACCES",
            libc::EFAULT => "EFAULT",
            libc::EBUSY => "EBUSY",
            libc::EEXIST => "EEXIST",
            libc::EXDEV => "EXDEV",
            libc::ENODEV => "ENODEV",
            libc::ENOTDIR => "ENOTDIR",
            libc::EISDIR => "EISDIR",
            libc::EINVAL => "EINVAL",
            libc::ENFILE => "ENFILE",
            libc::EMFILE => "EMFILE",
            libc::ENOTTY => "ENOTTY",
            libc::EFBIG => "EFBIG",
            libc::ENOSPC => "ENOSPC",
            libc::ESPIPE => "ESPIPE",
            libc::EROFS => "EROFS",
            libc::EMLINK => "EMLINK",
            libc::EPIPE => "EPIPE",
            libc::ERANGE => "ERANGE",
            libc::ENOSYS => "ENOSYS",
            _ => "E???",
        }
    }

    fn errno_text(e: i32) -> String {
        let s = unsafe { libc::strerror(e) };
        if s.is_null() {
            return String::from("Unknown error");
        }
        let cs = unsafe { std::ffi::CStr::from_ptr(s) };
        cs.to_string_lossy().into_owned()
    }

    // === Syscall number → name ===

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
            _ => "syscall",
        }
    }
} // mod linux
