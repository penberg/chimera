//! Guest system-call interception: the [`SystemCall`] value handed to embedder
//! handlers, the [`SystemCalls`] trait, the runtime's syscall driver
//! [`syscall`], and the default [`Passthrough`] handler.

use crate::sys::linux::syscall::host_syscall;

/// A single guest system call, presented to a [`SystemCalls`] handler.
///
/// `number` is the syscall number from the guest's syscall-number register
/// (`rax` on x86-64). `args` contains the six argument
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
    has_result: bool,
}

impl SystemCall {
    /// Write `result` into this `SystemCall` in the host's syscall-return ABI:
    /// `Error(errno)` is encoded as `-errno` in the return slot, the way the
    /// Linux/x86-64 kernel reports a failed syscall.
    pub fn set_result(&mut self, result: SyscallResult) {
        match result {
            SyscallResult::Ok(value) => self.set_return(value),
            SyscallResult::Error(errno) => self.set_return(-(errno as i64)),
        }
    }

    /// Set the value the guest will see in its return register after this
    /// syscall. Most handlers should use [`SystemCall::set_result`] instead,
    /// which encodes the host's convention for you.
    pub fn set_return(&mut self, value: i64) {
        self.return_value = value;
        self.has_result = true;
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
        if (-4095..0).contains(&self.return_value) {
            Some(SyscallResult::Error((-self.return_value) as i32))
        } else {
            Some(SyscallResult::Ok(self.return_value))
        }
    }

    pub(crate) fn return_value(&self) -> i64 {
        self.return_value
    }

    pub(crate) fn new(number: u64, args: [u64; 6]) -> Self {
        Self {
            number,
            args,
            return_value: 0,
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
///
/// Chimera also rewrites the `prot` argument of `mmap`, `mprotect`, and
/// `pkey_mprotect` before servicing them, clearing `PROT_EXEC` (see
/// [`crate::syscall::syscall`]). `pre_syscall` sees the guest's original
/// request; every later stage, including `do_syscall`, sees the rewritten one.
pub trait SystemCalls {
    /// Observe a guest syscall before Chimera or the embedder services it.
    fn pre_syscall(&mut self, _call: &SystemCall) {}

    /// Service a guest syscall that Chimera delegated to the embedder.
    ///
    /// The default implementation forwards the call to the host kernel.
    fn do_syscall(&mut self, call: &mut SystemCall) {
        call.set_result(host_syscall(call));
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
/// kernel's "errno in `-rax`" convention is hidden inside
/// [`crate::host_syscall`] and [`SystemCall::set_result`]; handlers see one
/// portable shape either way.
#[derive(Copy, Clone)]
pub enum SyscallResult {
    /// The kernel reported success and produced this value.
    Ok(i64),
    /// The kernel reported failure with this errno.
    Error(i32),
}

// === Syscall driver ===
//
// `syscall` is the runtime entry the arch dispatcher calls once per guest
// syscall instruction: it intercepts the calls Chimera must service itself
// (`exit`/`exit_group`, `execve`/`execveat`, `arch_prctl`, the `mmap` family)
// and hands the rest to the embedder hooks.

mod host {
    use super::{SyscallResult, SystemCall, SystemCalls, host_syscall};
    use crate::arch::dispatch::Thread;

    /// Drive one guest syscall. Runtime-owned syscalls are serviced inline
    /// here, the way a kernel's syscall table services its own; delegated
    /// syscalls go to the embedder hook.
    ///
    /// `exit`/`exit_group` mark the thread done (the dispatch loop notices
    /// `running == false` on its next iteration and returns `exit_code`).
    /// Forwarding either to the host kernel would terminate Chimera itself.
    ///
    /// `execve`/`execveat` are intercepted and never forwarded to the host
    /// kernel: forwarding either would have the host replace the whole process
    /// image — Chimera's runtime, code cache, and translation map included —
    /// with an untranslated program that then runs natively, outside the
    /// sandbox entirely. Instead the dispatch loop tears down the old image and
    /// re-enters the translator on the new one (see `crate::sys::linux::exec`).
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
    ///
    /// Finally, Chimera enforces a W^X invariant on the guest: the guest never
    /// executes its own pages natively — the dispatcher reads them and runs
    /// translated blocks from the code cache — so freshly mapped or re-protected
    /// guest code must never be executable in the host page tables. Before any
    /// dispatch, the `prot` argument of `mmap`, `mprotect`, and `pkey_mprotect`
    /// (always `args[2]`) has `PROT_EXEC` replaced with `PROT_READ`, so a stray
    /// native jump into guest code faults instead of running untranslated while
    /// the translator can still read the bytes it will translate (an
    /// execute-only mapping would otherwise become `PROT_NONE`). Unlike the
    /// `mmap` family, `mprotect`/`pkey_mprotect` are not otherwise
    /// runtime-owned: they still reach `do_syscall()`, only with `PROT_EXEC`
    /// already cleared from their argument.
    ///
    /// `clone`/`clone3` are refused with `EPERM` when they request `CLONE_VM`, a
    /// new thread sharing the address space: Chimera is single-threaded, and the
    /// new host thread would execute guest code with no translator context,
    /// natively. Without `CLONE_VM` the call is an ordinary `fork`, whose
    /// copy-on-write child carries Chimera and resumes in translated code, so it
    /// is forwarded. (`clone3` carries its flags in a `clone_args` struct rather
    /// than a register; the rest is identical.) `vfork` always shares the
    /// address space and has no `fork`-shaped variant, so it is refused outright.
    ///
    /// The io_uring interface (`io_uring_setup`/`io_uring_enter`/
    /// `io_uring_register`) is refused with `EPERM`: it would let the guest
    /// queue system calls the kernel runs asynchronously, never passing them
    /// back through this driver, bypassing every interception here. `shmat`/
    /// `shmdt` are refused for a related reason: `shmat` maps shared memory
    /// outside the runtime's `mmap` bookkeeping (and `SHM_EXEC` would dodge
    /// W^X), so the segment is never tracked. `remap_file_pages` is refused
    /// because it rebinds the pages under an existing mapping, which can change
    /// the bytes at an already-translated guest PC behind the translator's back.
    /// `ptrace` is refused because it reads and writes a process's memory and
    /// registers out of band, ignoring page protection and the translator both.
    pub fn syscall(thread: &mut Thread, call: &mut SystemCall, handler: &mut dyn SystemCalls) {
        handler.pre_syscall(call);

        let nr = call.number as i64;
        match nr {
            libc::SYS_mmap => {
                let prot = call.args[2] as libc::c_int;
                if prot & libc::PROT_EXEC != 0 {
                    call.args[2] = ((prot & !libc::PROT_EXEC) | libc::PROT_READ) as u64;
                }
                let result = host_syscall(call);
                if let SyscallResult::Ok(addr) = result {
                    thread
                        .addr_space()
                        .add_region(addr as usize, call.args[1] as usize);
                }
                call.set_result(result);
            }
            libc::SYS_mprotect | libc::SYS_pkey_mprotect => {
                // Not runtime-owned: strip PROT_EXEC from the requested
                // protection, then hand off to the embedder unchanged.
                let prot = call.args[2] as libc::c_int;
                if prot & libc::PROT_EXEC != 0 {
                    call.args[2] = ((prot & !libc::PROT_EXEC) | libc::PROT_READ) as u64;
                }
                handler.do_syscall(call);
            }
            libc::SYS_munmap => {
                let result = host_syscall(call);
                if matches!(result, SyscallResult::Ok(_)) {
                    thread
                        .addr_space()
                        .remove_region(call.args[0] as usize, call.args[1] as usize);
                }
                call.set_result(result);
            }
            libc::SYS_mremap => {
                let result = host_syscall(call);
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
            }
            libc::SYS_exit | libc::SYS_exit_group => {
                thread.exit_code = call.args[0] as i32;
                thread.running = false;
            }
            libc::SYS_execve | libc::SYS_execveat => {
                // Intercepted, never forwarded to the host kernel: forwarding would
                // have the host replace Chimera's whole process image — runtime,
                // code cache, and translation map included — with an untranslated
                // program running natively, outside the sandbox. The dispatch loop
                // tears down the old image and re-enters the translator on the new
                // one (see `crate::sys::linux::exec`); report success so observers
                // see the allowed call.
                call.set_result(SyscallResult::Ok(0));
            }
            // A `clone` that creates a new thread sharing this address space
            // (`CLONE_VM`) would have the host kernel run guest code on the new
            // thread with no Chimera context — natively, never through the
            // translator — a sandbox escape (and, today, a crash). Refuse it with
            // `EPERM`. A `clone` without `CLONE_VM` is an ordinary `fork`: a
            // copy-on-write duplicate of the whole process, Chimera included,
            // that resumes in translated code, so it falls through to the default
            // arm and is forwarded.
            libc::SYS_clone if call.args[0] & libc::CLONE_VM as u64 != 0 => {
                call.set_result(SyscallResult::Error(libc::EPERM));
            }
            // `clone3` is the same escape as `clone` above, but carries its
            // flags in the `clone_args` struct `args[0]` points at rather than
            // in a register. Refuse it when those flags request `CLONE_VM`; a
            // `clone3` without it (the `fork`-shaped case) falls through to the
            // default arm and is forwarded.
            libc::SYS_clone3 if clone3_requests_vm(call.args[0]) => {
                call.set_result(SyscallResult::Error(libc::EPERM));
            }
            // `vfork` always shares the address space (`CLONE_VM`) and, worse,
            // suspends the parent until the child execs or exits while both run
            // on the same stack — there is no `fork`-shaped variant to allow, so
            // refuse it outright.
            libc::SYS_vfork => {
                call.set_result(SyscallResult::Error(libc::EPERM));
            }
            // io_uring lets the guest queue system calls that the kernel then
            // executes asynchronously, on its own, without ever passing them
            // back through Chimera's syscall path — a direct way around every
            // interception above. Refuse the whole interface: denying
            // `io_uring_setup` stops a ring from ever existing, and denying
            // `enter`/`register` covers any fd that slipped through.
            libc::SYS_io_uring_setup | libc::SYS_io_uring_enter | libc::SYS_io_uring_register => {
                call.set_result(SyscallResult::Error(libc::EPERM));
            }
            // `shmat` maps a System V shared-memory segment into the address
            // space behind the `mmap` family's back — so the runtime never
            // records it, and `SHM_EXEC` would make it executable, dodging W^X —
            // and `shmdt` tears such a mapping back down. Refuse both with
            // `EPERM`; `shmget` alone only allocates an id and maps nothing.
            libc::SYS_shmat | libc::SYS_shmdt => {
                call.set_result(SyscallResult::Error(libc::EPERM));
            }
            // `remap_file_pages` rebinds the file pages backing an existing
            // mapping without changing its address, so the bytes at a guest PC
            // the translator has already cached can change underneath it (a
            // stale-translation hole), and the resulting nonlinear mapping
            // escapes the runtime's region bookkeeping. It is deprecated and
            // emulated by the kernel anyway; refuse it with `EPERM`.
            libc::SYS_remap_file_pages => {
                call.set_result(SyscallResult::Error(libc::EPERM));
            }
            // `ptrace` is an out-of-band channel into another process: it reads
            // and writes registers and memory regardless of page protection
            // (`PTRACE_POKETEXT`/`POKEDATA`) and redirects control flow, none of
            // which passes through the translator. A traced peer — or, with a
            // future multi-process model, Chimera itself — could be driven
            // straight out of the sandbox. Refuse it with `EPERM`.
            libc::SYS_ptrace => {
                call.set_result(SyscallResult::Error(libc::EPERM));
            }
            libc::SYS_arch_prctl => {
                const ARCH_SET_GS: u64 = 0x1001;
                const ARCH_SET_FS: u64 = 0x1002;
                const ARCH_GET_FS: u64 = 0x1003;
                const ARCH_GET_GS: u64 = 0x1004;
                match call.args[0] {
                    ARCH_SET_FS => {
                        thread.state.guest_fs_base = call.args[1];
                        call.set_result(SyscallResult::Ok(0));
                    }
                    ARCH_GET_FS => {
                        let fs = thread.state.guest_fs_base;
                        if call.args[1] != 0 {
                            unsafe {
                                (call.args[1] as *mut u64).write(fs);
                            }
                        }
                        call.set_result(SyscallResult::Ok(0));
                    }
                    ARCH_SET_GS | ARCH_GET_GS => {
                        call.set_result(SyscallResult::Error(libc::EINVAL));
                    }
                    // unknown subfunction: delegate to the embedder
                    _ => {
                        handler.do_syscall(call);
                    }
                }
            }
            libc::SYS_rt_sigaction => {
                let r = thread
                    .signals_mut()
                    .sigaction(call.args[0], call.args[1], call.args[2]);
                call.set_result(r);
            }
            libc::SYS_rt_sigprocmask => {
                let r = thread.signals_mut().sigprocmask(
                    call.args[0] as i32,
                    call.args[1],
                    call.args[2],
                );
                call.set_result(r);
            }
            libc::SYS_sigaltstack => {
                let r = thread.signals_mut().sigaltstack(call.args[0], call.args[1]);
                call.set_result(r);
            }
            libc::SYS_rt_sigreturn => {
                // Restore the pre-signal context from the frame on the guest
                // stack. `sigreturn` writes the guest's saved rax back into the
                // register file; mirror it into the call so `handle_syscall`'s
                // unconditional rax writeback (see `crate::arch::dispatch`) is a
                // no-op rather than clobbering it.
                thread.sigreturn();
                call.set_return(thread.state.regs[0] as i64);
            }
            _ => {
                handler.do_syscall(call);
            }
        };
        handler.post_syscall(call);
    }

    /// Whether a `clone3` whose `clone_args` struct lives at `args_ptr` requests
    /// `CLONE_VM`. The flags are the first `u64` of the struct; Chimera shares
    /// the guest's address space, so this is a plain read. A null pointer
    /// carries no flags — the kernel rejects it with `EFAULT` once forwarded.
    fn clone3_requests_vm(args_ptr: u64) -> bool {
        if args_ptr == 0 {
            return false;
        }
        let flags = unsafe { core::ptr::read(args_ptr as *const u64) };
        flags & libc::CLONE_VM as u64 != 0
    }
}

pub use host::syscall;
