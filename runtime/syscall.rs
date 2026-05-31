//! Guest system-call interception: the [`SystemCall`] value handed to embedder
//! handlers, the [`SystemCalls`] trait, the runtime's syscall driver
//! [`syscall`], and the default [`Passthrough`] handler.

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
use crate::sys::linux::syscall::host_syscall as host_syscall_;

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
use host::host_syscall as host_syscall_;

/// A single guest system call, presented to a [`SystemCalls`] handler.
///
/// `number` is the syscall number from the guest's syscall-number register
/// (`rax` on x86-64, `x16` on Darwin/arm64). `args` contains the six argument
/// registers in the guest ABI's syscall order. The handler decides what the
/// call should do — forward it to the host kernel via
/// [`crate::host_syscall`], synthesize an answer with
/// [`SystemCall::set_result`] or [`SystemCall::set_return`], or both.
pub struct SystemCall {
    /// The syscall number.
    pub number: u64,
    /// The six argument registers.
    pub args: [u64; 6],
    return_value: i64,
    is_error: bool,
    has_result: bool,
}

impl SystemCall {
    /// Write `result` into this `SystemCall` in the host's syscall-return ABI.
    /// On Linux, `Error(errno)` is encoded as `-errno` in the return slot with
    /// the error flag clear; on Darwin, it's encoded as positive `errno` with
    /// the error flag set so the dispatcher can drive the NZCV carry bit.
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
    /// syscall. Most handlers should use [`SystemCall::set_result`] instead,
    /// which encodes the host's convention for you.
    pub fn set_return(&mut self, value: i64) {
        self.return_value = value;
        self.has_result = true;
    }

    /// Mark this syscall as an error. On Darwin/arm64, the dispatcher
    /// translates this into the NZCV carry flag the guest's libc expects.
    /// On Linux/x86-64 the bit is ignored — errors are conveyed by setting
    /// a negative return value.
    pub fn set_error(&mut self, is_error: bool) {
        self.is_error = is_error;
    }

    /// Return the syscall outcome currently stored in this value.
    ///
    /// Returns `None` when no result exists yet, which is the case in
    /// `pre_syscall` and for syscalls like `exit`/`exit_group` that never
    /// resume in the guest.
    pub fn result(&self) -> Option<SyscallResult> {
        if !self.has_result {
            return None;
        }
        #[cfg(target_os = "macos")]
        {
            if self.is_error {
                Some(SyscallResult::Error(self.return_value as i32))
            } else {
                Some(SyscallResult::Ok(self.return_value))
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            if (-4095..0).contains(&self.return_value) {
                Some(SyscallResult::Error((-self.return_value) as i32))
            } else {
                Some(SyscallResult::Ok(self.return_value))
            }
        }
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
            has_result: false,
        }
    }
}

/// Guest system-call implementation supplied by the embedder.
///
/// Chimera does not implement system-call policy itself: delegated guest
/// syscalls are handed to [`SystemCalls::do_syscall`], while
/// [`SystemCalls::pre_syscall`] and [`SystemCalls::post_syscall`] can observe
/// every guest syscall, including the few Chimera intercepts for its own
/// correctness (`exit`, `execve`, `arch_prctl`, `mmap`, `munmap`, `mremap`).
pub trait SystemCalls {
    /// Observe a guest syscall before Chimera or the embedder services it.
    fn pre_syscall(&mut self, _call: &SystemCall) {}

    /// Service a guest syscall that Chimera delegated to the embedder.
    ///
    /// The default implementation forwards the call to the host kernel.
    fn do_syscall(&mut self, call: &mut SystemCall) {
        call.set_result(host_syscall_(call));
    }

    /// Observe a guest syscall after its final result is known, if any.
    fn post_syscall(&mut self, _call: &SystemCall) {}
}

/// The default system-call handler: forwards every delegated guest syscall to
/// the host kernel verbatim.
pub struct Passthrough;

impl SystemCalls for Passthrough {}

/// The outcome the host kernel reported for a forwarded syscall, or that a
/// runtime intercept synthesized in lieu of forwarding. `Ok(value)` is the
/// kernel's success value; `Error(errno)` carries the positive errno. The
/// ABI difference between Linux ("errno in `-rax`") and Darwin ("errno in
/// `x0` with the NZCV carry bit set") is hidden inside
/// [`crate::host_syscall`] and [`SystemCall::set_result`]; handlers see one
/// portable shape either way.
#[derive(Copy, Clone)]
pub enum SyscallResult {
    /// The kernel reported success and produced this value.
    Ok(i64),
    /// The kernel reported failure with this errno.
    Error(i32),
}

// === Host-specific syscall driver ===
//
// `syscall` (on Linux) is the runtime entry the arch dispatcher calls once per
// guest syscall instruction: it intercepts the calls Chimera must service
// itself (`exit`/`exit_group`, `execve`/`execveat`, `arch_prctl`, the
// `mmap` family) and hands the rest to the embedder hooks. On Darwin and
// unsupported hosts the file only exposes `host_syscall` — the unified
// driver is Linux-only for now.

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod host {
    use super::{SyscallResult, SystemCall, SystemCalls, host_syscall_};
    use crate::arch::dispatch::Thread;

    /// Drive one guest syscall. Runtime-owned syscalls are serviced inline
    /// here, the way a kernel's syscall table services its own; delegated
    /// syscalls go to the embedder hook.
    ///
    /// `exit`/`exit_group` mark the thread done (the dispatch loop notices
    /// `running == false` on its next iteration and returns `exit_code`).
    /// Forwarding either to the host kernel would terminate Chimera itself.
    ///
    /// `execve`/`execveat` are refused with `EPERM`. Forwarding either would
    /// have the host kernel replace the whole process image — Chimera's
    /// runtime, code cache, and translation map included — with an
    /// untranslated program that then runs natively, outside the sandbox
    /// entirely. The stop-gap is to deny it and log the attempt; a complete
    /// implementation would re-enter Chimera on the new image (see
    /// `ARCHITECTURE.md`).
    ///
    /// `arch_prctl` is virtualized: Chimera owns the real FS base for its own
    /// TLS and reserves GS for the thread-context pointer, so the guest's view
    /// of both is kept in the thread context rather than in the CPU's MSRs.
    /// `ARCH_SET_FS` records the requested base into `thread.state` and
    /// `ARCH_GET_FS` reads it back, both without touching the kernel.
    /// `ARCH_SET_GS`/`ARCH_GET_GS` return `EINVAL`. Unknown subfunctions fall
    /// through to the embedder.
    ///
    /// `mmap`/`munmap`/`mremap` are also runtime-owned: Chimera forwards them
    /// to the host kernel itself so its guest-mapping bookkeeping stays
    /// authoritative. Embedders can observe them in `pre_syscall()` and
    /// `post_syscall()`, but they do not reach `do_syscall()`.
    pub fn syscall(thread: &mut Thread, call: &mut SystemCall, handler: &mut dyn SystemCalls) {
        handler.pre_syscall(call);

        if call.number == libc::SYS_exit_group as u64 || call.number == libc::SYS_exit as u64 {
            thread.exit_code = call.args[0] as i32;
            thread.running = false;
            handler.post_syscall(call);
            return;
        }

        if call.number == libc::SYS_execve as u64 || call.number == libc::SYS_execveat as u64 {
            let name = if call.number == libc::SYS_execve as u64 {
                "execve"
            } else {
                "execveat"
            };
            eprintln!("chimera: blocked {name} (would escape the sandbox); returning EPERM");
            call.set_result(SyscallResult::Error(libc::EPERM));
            handler.post_syscall(call);
            return;
        }

        if call.number == libc::SYS_arch_prctl as u64 {
            const ARCH_SET_GS: u64 = 0x1001;
            const ARCH_SET_FS: u64 = 0x1002;
            const ARCH_GET_FS: u64 = 0x1003;
            const ARCH_GET_GS: u64 = 0x1004;
            match call.args[0] {
                ARCH_SET_FS => {
                    thread.state.guest_fs_base = call.args[1];
                    call.set_result(SyscallResult::Ok(0));
                    handler.post_syscall(call);
                    return;
                }
                ARCH_GET_FS => {
                    let fs = thread.state.guest_fs_base;
                    if call.args[1] != 0 {
                        unsafe {
                            (call.args[1] as *mut u64).write(fs);
                        }
                    }
                    call.set_result(SyscallResult::Ok(0));
                    handler.post_syscall(call);
                    return;
                }
                ARCH_SET_GS | ARCH_GET_GS => {
                    call.set_result(SyscallResult::Error(libc::EINVAL));
                    handler.post_syscall(call);
                    return;
                }
                _ => {} // unknown subfunction: delegate to the embedder
            }
        }

        if call.number == libc::SYS_mmap as u64 {
            let result = host_syscall_(call);
            if let SyscallResult::Ok(addr) = result {
                thread
                    .addr_space()
                    .add_region(addr as usize, call.args[1] as usize);
            }
            call.set_result(result);
            handler.post_syscall(call);
            return;
        }

        if call.number == libc::SYS_munmap as u64 {
            let result = host_syscall_(call);
            if matches!(result, SyscallResult::Ok(_)) {
                thread
                    .addr_space()
                    .remove_region(call.args[0] as usize, call.args[1] as usize);
            }
            call.set_result(result);
            handler.post_syscall(call);
            return;
        }

        if call.number == libc::SYS_mremap as u64 {
            let result = host_syscall_(call);
            if let SyscallResult::Ok(new_start) = result {
                let flags = call.args[3] as libc::c_int;
                let dontunmap = (flags & libc::MREMAP_DONTUNMAP) != 0;
                thread.addr_space().remap_region(
                    call.args[0] as usize,
                    call.args[1] as usize,
                    new_start as usize,
                    call.args[2] as usize,
                    dontunmap,
                );
            }
            call.set_result(result);
            handler.post_syscall(call);
            return;
        }

        handler.do_syscall(call);
        handler.post_syscall(call);
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod host {
    use std::arch::asm;

    use super::{SyscallResult, SystemCall};

    /// Forward `call` to the host kernel and return the result. Darwin's
    /// kernel signals errors via the NZCV carry flag rather than as a
    /// negative return value, so the carry bit becomes [`SyscallResult::Error`]
    /// and the value field carries the positive errno.
    ///
    /// Darwin's `exit` (BSD syscall #1) is intercepted: forwarding it to
    /// the host kernel would terminate Chimera itself, so it no-ops and
    /// returns 0.
    pub fn host_syscall(call: &SystemCall) -> SyscallResult {
        if call.number == 1 {
            return SyscallResult::Ok(0);
        }
        // `__abort_with_payload` (BSD #521): if dyld asserts during the
        // bring-up we are still bringing online, forwarding the call would
        // terminate Chimera too. Pretend it succeeded.
        if call.number == 521 {
            return SyscallResult::Ok(0);
        }
        // `thread_set_tsd_base` (syscall #0x80000000 on arm64 Darwin):
        // forwarding would clobber the same register Chimera's Rust runtime
        // uses to find its own pthread state.
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
    pub fn host_syscall(_call: &SystemCall) -> SyscallResult {
        SyscallResult::Error(libc::ENOSYS)
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub use host::syscall;

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
pub use host::host_syscall;
