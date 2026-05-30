//! Guest system-call interception: the `SystemCall` value handed to embedder
//! handlers, the `SystemCalls` trait, the free [`syscall`] function, and the
//! default [`Passthrough`] handler.

/// A single guest system call, presented to a [`SystemCalls`] handler.
///
/// `number` is the syscall number from the guest's syscall-number register
/// (`rax` on x86-64, `x16` on Darwin/arm64). `args` contains the six argument
/// registers in the guest ABI's syscall order. The handler decides what the
/// call should do — forward it to the host kernel via [`syscall`],
/// synthesize an answer with [`SystemCall::set_result`] or
/// [`SystemCall::set_return`], or both.
pub struct SystemCall {
    /// The syscall number.
    pub number: u64,
    /// The six argument registers.
    pub args: [u64; 6],
    return_value: i64,
    is_error: bool,
}

impl SystemCall {
    /// Write `result` into this `SystemCall` in the host's syscall-return ABI.
    /// On Linux, `Error(errno)` is encoded as `-errno` in the return slot with
    /// the error flag clear; on Darwin, it's encoded as positive `errno` with
    /// the error flag set so the dispatcher can drive the NZCV carry bit. Use
    /// this in preference to a separate `set_return`/`set_error` pair when the
    /// handler is forwarding a [`SyscallResult`] verbatim.
    pub fn set_result(&mut self, result: SyscallResult) {
        match result {
            SyscallResult::Ok(value) => {
                self.set_return(value);
                self.set_error(false);
            }
            SyscallResult::Error(errno) => {
                #[cfg(target_os = "macos")]
                {
                    self.set_return(errno as i64);
                    self.set_error(true);
                }
                #[cfg(not(target_os = "macos"))]
                {
                    self.set_return(-(errno as i64));
                    self.set_error(false);
                }
            }
        }
    }

    /// Set the value the guest will see in its return register after this
    /// syscall. On hosts that use a negative-errno convention (Linux), the
    /// caller writes `-errno` here; on hosts that use a separate error flag
    /// (Darwin's NZCV carry bit), the caller pairs this with
    /// [`SystemCall::set_error`]. Most handlers should use
    /// [`SystemCall::set_result`] instead, which encodes the host's
    /// convention for you.
    pub fn set_return(&mut self, value: i64) {
        self.return_value = value;
    }

    /// Mark this syscall as an error. On Darwin/arm64, the dispatcher
    /// translates this into the NZCV carry flag the guest's libc expects.
    /// On Linux/x86-64 the bit is ignored — errors are conveyed by setting
    /// a negative return value.
    pub fn set_error(&mut self, is_error: bool) {
        self.is_error = is_error;
    }

    pub(crate) fn return_value(&self) -> i64 {
        self.return_value
    }

    #[allow(dead_code)] // Read only by the Darwin/arm64 dispatcher.
    pub(crate) fn is_error(&self) -> bool {
        self.is_error
    }

    pub(crate) fn new(number: u64, args: [u64; 6]) -> Self {
        Self {
            number,
            args,
            return_value: 0,
            is_error: false,
        }
    }
}

/// Guest system-call implementation supplied by the embedder.
///
/// Chimera does not implement system-call semantics itself: every guest
/// syscall instruction is intercepted and handed to [`SystemCalls::handle`].
/// Implementors can forward to the host kernel, return errors, log,
/// re-route, or emulate the call entirely.
pub trait SystemCalls {
    /// Invoked for every guest syscall, before the guest resumes.
    fn handle(&mut self, call: &mut SystemCall);
}

/// The default system-call handler: forwards every guest syscall to the
/// host kernel verbatim.
pub struct Passthrough;

impl SystemCalls for Passthrough {
    fn handle(&mut self, call: &mut SystemCall) {
        call.set_result(syscall(call));
    }
}

/// The outcome [`syscall`] reports.
///
/// `Ok(value)` is what the kernel returned on success; `Error(errno)` carries
/// the positive errno on failure. The ABI difference between Linux's "errors
/// are encoded as `-errno` in the return value" and Darwin's "errors are
/// flagged by the NZCV carry bit and `x0` carries the positive errno" lives
/// inside [`syscall`] (which produces the right variant) and
/// [`SystemCall::set_result`] (which writes it back in the host's
/// convention); handlers see one portable shape either way.
#[derive(Copy, Clone)]
pub enum SyscallResult {
    /// The kernel reported success and produced this value.
    Ok(i64),
    /// The kernel reported failure with this errno.
    Error(i32),
}

// === Host-specific passthrough implementation ===
//
// `syscall` is the bridge from a `SystemCall` to the host kernel. The shape
// of that bridge depends on both the host ISA (which registers carry which
// arguments) and the host kernel (which numbers mean what, which syscalls the
// runtime must intercept rather than forward). It therefore lives in per-host
// modules selected by `cfg`.

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod host {
    use std::arch::asm;

    use super::{SyscallResult, SystemCall};
    use crate::sys::linux::syscall::host_syscall;

    /// Forward `call` to the host kernel and return the result.
    ///
    /// `arch_prctl` is intercepted: the guest's FS base is virtualized
    /// (Chimera owns the real FS for its own TLS), and GS is reserved for
    /// Chimera. `ARCH_SET_FS` records the requested value into the per-thread
    /// state and returns success without touching the kernel; `ARCH_GET_FS`
    /// reads it back; `ARCH_SET_GS` and `ARCH_GET_GS` return `EINVAL`.
    ///
    /// `exit` and `exit_group` are intercepted: forwarding them to the host
    /// kernel would terminate Chimera itself, so they no-op and return 0. The
    /// runtime captures the requested exit code from the syscall's first
    /// argument and ends the run cleanly after the handler returns.
    ///
    /// `execve` and `execveat` are intercepted and refused with `EPERM`.
    /// Forwarding either to the host kernel would replace the whole process
    /// image — Chimera's runtime, code cache, and translation map included —
    /// with an untranslated program that then runs natively, outside the
    /// sandbox entirely. That is a sandbox escape, not a feature, so the
    /// stop-gap is to deny it and log the attempt. (A real implementation
    /// would re-enter Chimera on the new image; see `ARCHITECTURE.md`.)
    pub fn syscall(call: &SystemCall) -> SyscallResult {
        if call.number == libc::SYS_exit_group as u64 || call.number == libc::SYS_exit as u64 {
            return SyscallResult::Ok(0);
        }

        if call.number == libc::SYS_execve as u64 || call.number == libc::SYS_execveat as u64 {
            let name = if call.number == libc::SYS_execve as u64 {
                "execve"
            } else {
                "execveat"
            };
            eprintln!("chimera: blocked {name} (would escape the sandbox); returning EPERM");
            return SyscallResult::Error(libc::EPERM);
        }

        if call.number == libc::SYS_arch_prctl as u64 {
            const ARCH_SET_GS: u64 = 0x1001;
            const ARCH_SET_FS: u64 = 0x1002;
            const ARCH_GET_FS: u64 = 0x1003;
            const ARCH_GET_GS: u64 = 0x1004;
            // Offset of `guest_fs_base` in `ThreadState`, addressed via GS.
            const GUEST_FS_OFF: usize = 168;
            match call.args[0] {
                ARCH_SET_FS => {
                    unsafe {
                        asm!(
                            "mov gs:[{off}], {val}",
                            off = const GUEST_FS_OFF,
                            val = in(reg) call.args[1],
                            options(nostack, preserves_flags),
                        );
                    }
                    return SyscallResult::Ok(0);
                }
                ARCH_GET_FS => {
                    let fs: u64;
                    unsafe {
                        asm!(
                            "mov {val}, gs:[{off}]",
                            off = const GUEST_FS_OFF,
                            val = out(reg) fs,
                            options(nostack, preserves_flags),
                        );
                    }
                    if call.args[1] != 0 {
                        unsafe {
                            (call.args[1] as *mut u64).write(fs);
                        }
                    }
                    return SyscallResult::Ok(0);
                }
                ARCH_SET_GS | ARCH_GET_GS => return SyscallResult::Error(libc::EINVAL),
                _ => {}
            }
        }

        // Linux's kernel signals errors as `-errno` in the return value, in
        // the closed range `[-4095, -1]`. Anything outside that range is a
        // successful result (including legitimate "negative-looking" values
        // like the high user-space addresses `mmap` can hand back).
        let ret = host_syscall(call);
        if (-4095..0).contains(&ret) {
            SyscallResult::Error(-ret as i32)
        } else {
            SyscallResult::Ok(ret)
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod host {
    use std::arch::asm;

    use super::{SyscallResult, SystemCall};

    /// Forward `call` to the host kernel and return the result. Darwin's
    /// kernel signals errors via the NZCV carry flag rather than as a
    /// negative return value, so the carry bit becomes [`SyscallResult::Error`]
    /// here and the value field carries the positive errno.
    ///
    /// Darwin's `exit` (BSD syscall #1) is intercepted: forwarding it to
    /// the host kernel would terminate Chimera itself, so it no-ops and
    /// returns 0. The run loop captures the requested exit code from the
    /// syscall's first argument and ends the run cleanly after the handler
    /// returns.
    pub fn syscall(call: &SystemCall) -> SyscallResult {
        if call.number == 1 {
            return SyscallResult::Ok(0);
        }
        // `__abort_with_payload` (BSD #521): if dyld asserts during the
        // bring-up we are still bringing online, forwarding the call would
        // terminate Chimera too. Pretend it succeeded so the caller's
        // following `exit` runs through our intercept instead, giving us a
        // chance to see post-assertion behavior. (Long-term, the
        // assertion should not fire at all; this is a temporary survival
        // valve while we shake the translator out.)
        if call.number == 521 {
            return SyscallResult::Ok(0);
        }
        // `thread_set_tsd_base` (syscall #0x80000000 on arm64 Darwin):
        // the guest is trying to install its own value into the kernel's
        // per-thread TPIDRRO_EL0 register. Forwarding that would clobber
        // the same register Chimera's Rust runtime uses to find its own
        // pthread state — every subsequent libc call from the runtime
        // (including the next eprintln) would deadlock or crash. We pretend
        // the call succeeded; the guest's TSD will live in its in-process
        // memory layout but the kernel-level TPIDRRO_EL0 stays bound to
        // Chimera. A future revision can virtualize this properly.
        if call.number == 0x80000000 {
            return SyscallResult::Ok(0);
        }

        // Darwin's userspace syscall ABI: number in x16, args in x0..x5,
        // `svc #0x80`. Return value lands in x0; the carry flag indicates
        // an error, in which case x0 holds the positive errno.
        let ret: i64;
        let cflag: u64;
        unsafe {
            asm!(
                "svc #0x80",
                "cset {cflag}, cs",
                in("x16") call.number,
                inout("x0") call.args[0] => ret,
                in("x1") call.args[1],
                in("x2") call.args[2],
                in("x3") call.args[3],
                in("x4") call.args[4],
                in("x5") call.args[5],
                cflag = lateout(reg) cflag,
                options(nostack, preserves_flags),
            );
        }
        if cflag != 0 {
            SyscallResult::Error(ret as i32)
        } else {
            SyscallResult::Ok(ret)
        }
    }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_os = "linux"),
    all(target_arch = "aarch64", target_os = "macos"),
)))]
mod host {
    use super::{SyscallResult, SystemCall};

    /// Stub used on hosts Chimera has not been ported to. Always reports
    /// `ENOSYS`.
    pub fn syscall(_call: &SystemCall) -> SyscallResult {
        SyscallResult::Error(libc::ENOSYS)
    }
}

pub use host::syscall;
