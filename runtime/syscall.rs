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
///
/// A single handler serves every guest thread, so the trait is `Send + Sync`
/// and its methods take `&self`: each guest thread runs on its own host thread
/// and may be inside the handler concurrently. A handler that needs mutable
/// state of its own reaches for interior mutability (a `Mutex`, an atomic) —
/// the `SystemCall` it is handed is exclusive to the calling thread, but `self`
/// is shared.
pub trait SystemCalls: Send + Sync {
    /// Observe a guest syscall before Chimera or the embedder services it.
    fn pre_syscall(&self, _call: &SystemCall) {}

    /// Service a guest syscall that Chimera delegated to the embedder.
    ///
    /// The default implementation forwards the call to the host kernel.
    fn do_syscall(&self, call: &mut SystemCall) {
        call.set_result(host_syscall(call));
    }

    /// Observe a guest syscall after its final result is known, if any.
    fn post_syscall(&self, _call: &SystemCall) {}
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
    use std::ptr;

    use super::{SyscallResult, SystemCall, SystemCalls, host_syscall};
    use crate::{
        arch::dispatch::{CLONE_ARGS_SIZE_MAX, RSP, Thread, read_clone3_args},
        sys::{
            linux::exec::{exec_errno, prepare_exec},
            mmap::copy_from_guest,
        },
    };

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
    /// A `clone` creating a new thread of this process (the `CLONE_THREAD`
    /// shape) cannot be forwarded: the new host task would execute guest code
    /// with no translator context, natively. Chimera instead spawns a host
    /// thread that runs the child guest in the shared process and returns its
    /// kernel TID to the parent (see `Thread::clone_vm`). `clone3` in the same
    /// shape is handled the same way via `Thread::clone3_vm`; it carries its
    /// arguments in a `clone_args` struct (the path modern glibc's
    /// `pthread_create` takes). `CLONE_VM` without `CLONE_THREAD` — a separate
    /// process sharing this address space — is refused with `EPERM`: Chimera
    /// can provide neither native forwarding (unsandboxed) nor a faithful
    /// emulation. Without `CLONE_VM` the call is an ordinary `fork`, whose
    /// copy-on-write child carries Chimera and resumes in translated code, so
    /// it is forwarded.
    /// `vfork` always shares the address space and has no `fork`-shaped variant,
    /// so it is refused outright.
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
    /// `process_vm_writev` is refused for the same reason: it writes directly
    /// into a process's address space through a second mapping the kernel makes
    /// of the target pages, so a write can land at an already-translated guest
    /// PC behind the translator's back (a stale-translation hole), and in a
    /// future multi-process model one guest could drive another straight out of
    /// the sandbox. Its read-only sibling `process_vm_readv` modifies nothing
    /// and is forwarded. `userfaultfd` is refused for the stale-translation
    /// reason again: it hands a userspace monitor the bytes that back a page on
    /// its first fault (via `UFFDIO_COPY`/`UFFDIO_CONTINUE`), so the guest could
    /// supply different code at an already-translated PC, and the resolving
    /// thread runs outside the translator entirely.
    ///
    /// `personality` is the exception that is filtered rather than refused
    /// wholesale, like `clone`'s `CLONE_VM` check. Its `READ_IMPLIES_EXEC`
    /// persona makes the kernel add `PROT_EXEC` to every readable mapping,
    /// silently undoing the `PROT_EXEC` stripping above and leaving guest pages
    /// executable in the host page tables — a W^X defeat — so a call that sets
    /// that bit is refused with `EPERM`. The query form
    /// (`personality(0xffffffff)`, which returns the current persona without
    /// changing it) and benign personas such as `ADDR_NO_RANDOMIZE` carry no
    /// such risk and are forwarded.
    pub fn syscall(thread: &mut Thread, call: &mut SystemCall, handler: &dyn SystemCalls) {
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
                    let mut space = thread.addr_space();
                    space.add_region(addr as usize, call.args[1] as usize);
                    // Reset SMC bookkeeping for the (possibly reused) addresses,
                    // so the new mapping arms from a clean slate.
                    space.note_map(addr as usize, call.args[1] as usize);
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
                // A protection change resets the host page (un-arming it) and may
                // precede a rewrite of code on a W^X-toggled JIT page, so drop the
                // affected pages' stale translations.
                if call.return_value() >= 0 {
                    thread
                        .addr_space()
                        .note_prot(call.args[0] as usize, call.args[1] as usize);
                }
            }
            libc::SYS_munmap => {
                let result = host_syscall(call);
                if matches!(result, SyscallResult::Ok(_)) {
                    let mut space = thread.addr_space();
                    space.remove_region(call.args[0] as usize, call.args[1] as usize);
                    space.note_unmap(call.args[0] as usize, call.args[1] as usize);
                }
                call.set_result(result);
            }
            libc::SYS_mremap => {
                let result = host_syscall(call);
                if let SyscallResult::Ok(new_start) = result {
                    let flags = call.args[3] as libc::c_int;
                    let dontunmap = (flags & libc::MREMAP_DONTUNMAP) != 0;
                    let old_start = call.args[0] as usize;
                    let old_len = call.args[1] as usize;
                    let new_len = call.args[2] as usize;
                    let mut space = thread.addr_space();
                    space.remap_region(old_start, old_len, new_start as usize, new_len, dontunmap);
                    // The bytes moved: drop translations at the old addresses
                    // (unless kept by MREMAP_DONTUNMAP) and at the new ones, which
                    // now hold whatever the move placed there.
                    if !dontunmap {
                        space.note_unmap(old_start, old_len);
                    }
                    space.note_unmap(new_start as usize, new_len);
                }
                call.set_result(result);
            }
            libc::SYS_exit => {
                // Thread-local: end only this thread. The run loop stops on its
                // next iteration; the main thread's stop ends the run (and the
                // process), a worker's stop just ends that host thread. This is
                // the `pthread_exit` / thread-return path.
                thread.exit_code = call.args[0] as i32;
                thread.running = false;
            }
            libc::SYS_exit_group => {
                // Process-wide: terminate the whole thread group, from any
                // thread. Publish the request on the shared process; every
                // thread's run loop observes it at its next boundary and stops,
                // so the main thread returns this code from the run and the
                // process exits. Forwarding to the host instead would kill the
                // embedder's process, which the runtime must not do.
                let code = call.args[0] as i32;
                thread.process().request_exit_group(code, &thread.state);
                thread.exit_code = code;
                thread.running = false;
            }
            libc::SYS_execve | libc::SYS_execveat => {
                // Intercepted, never forwarded to the host kernel: forwarding would
                // have the host replace Chimera's whole process image — runtime,
                // code cache, and translation map included — with an untranslated
                // program running natively, outside the sandbox.
                //
                // The replacement image is validated here, in the calling thread,
                // the way the kernel sequences an exec: a failure reports `-errno`
                // to the caller and disturbs no sibling. Only a loadable image
                // commits the exec — publishing it on the shared process stops
                // every thread (Linux's `de_thread` kills the siblings before a
                // new image is installed, whichever thread called exec), and the
                // main host thread waits out the stragglers, tears down the old
                // image, and enters the new one (see `crate::sys::linux::exec`).
                match prepare_exec(call.number, &call.args) {
                    Ok(prepared) => {
                        // A spawn child reaching a loadable image is a successful
                        // spawn: unblock the parent (see `spawned`) so it returns
                        // the child PID before the new image runs.
                        thread.report_spawn_success();
                        // First committer wins. A refused commit means a
                        // sibling's exec (or an exit_group) is already
                        // dissolving this group, this thread with it — the
                        // kernel's loser is killed by the winner's de_thread
                        // and never observes a return value, and neither does
                        // this one: the pending stop takes the thread at its
                        // next boundary, before the guest could read rax.
                        thread.process().request_exec(prepared, &thread.state);
                        // Report success so observers see the allowed call; the
                        // guest never reads it — a successful execve does not
                        // return.
                        call.set_result(SyscallResult::Ok(0));
                    }
                    // `prepare_exec` only parses, so every failure carries an
                    // errno (`EIO` is an unreachable fallback).
                    Err(err) => {
                        let errno = exec_errno(&err).unwrap_or(libc::EIO);
                        // A spawn child's failed exec is reported to the blocked
                        // parent, which surfaces it as `posix_spawn`'s errno.
                        thread.report_spawn_failure(errno);
                        call.set_result(SyscallResult::Error(errno));
                    }
                }
            }
            // A `clone` that creates a new thread of this process — the
            // `CLONE_THREAD` shape, which the kernel requires to carry
            // `CLONE_SIGHAND` and `CLONE_VM` — cannot be forwarded: the host
            // kernel would run guest code on the new task natively, with no
            // Chimera context, never through the translator. Instead Chimera
            // spawns a host thread that runs the child guest in the shared
            // process and returns its kernel TID (see `Thread::clone_vm`). A
            // malformed `CLONE_THREAD` falls through to the fork arm for the
            // kernel's authoritative `EINVAL`.
            libc::SYS_clone if is_thread_clone(call.args[0]) => {
                let tid = thread.clone_vm(&call.args);
                call.set_return(tid);
            }
            // `CLONE_VM | CLONE_VFORK` without `CLONE_THREAD` is the
            // `vfork`/`posix_spawn` pattern: a child that shares the parent's
            // memory only until it `execve`s (`CLONE_VFORK` promises it does
            // nothing else first). Chimera cannot run it as a host thread — its
            // `execve` would replace the whole shared image, tearing down the
            // parent. Emulate it as `fork`: strip `CLONE_VM`/`CLONE_VFORK` so the
            // child gets a copy-on-write image, runs its file actions and
            // `execve` (becoming an independent process) while the parent
            // continues. The `CLONE_VFORK` contract limits the child to
            // `exec`/`_exit`, so the copy is observationally equivalent — except
            // an exec failure is reported to the parent through the child's exit
            // status, not the shared errno.
            //
            // The guest passes a child stack (it pre-stashes the child's entry
            // function and argument there); the child must run on it. But that
            // stack must NOT reach the host `fork` — the host would set *Chimera's*
            // own stack pointer to a guest address and crash. So clear the host
            // stack argument, let the host child keep its copy-on-write Chimera
            // stack, and set the guest child's `rsp` to the requested stack.
            libc::SYS_clone
                if call.args[0] & libc::CLONE_VM as u64 != 0
                    && call.args[0] & libc::CLONE_THREAD as u64 == 0
                    && call.args[0] & libc::CLONE_VFORK as u64 != 0 =>
            {
                let guest_stack = call.args[1];
                call.args[0] &= !((libc::CLONE_VM | libc::CLONE_VFORK) as u64);
                call.args[1] = 0;
                spawned(thread, call, handler, guest_stack);
            }
            // `CLONE_VM` without `CLONE_THREAD` or `CLONE_VFORK` asks for a
            // *separate process* that keeps sharing this address space. Chimera
            // can honor neither half — forwarding would run the child natively
            // outside the sandbox; a host thread would give it this process's PID
            // and no `waitpid`; and a fork would silently break the memory
            // sharing the caller asked for. Refuse it visibly, like `vfork`. No
            // libc path reaches this shape.
            libc::SYS_clone
                if call.args[0] & libc::CLONE_VM as u64 != 0
                    && call.args[0] & libc::CLONE_THREAD as u64 == 0 =>
            {
                call.set_result(SyscallResult::Error(libc::EPERM));
            }
            // `clone3` carries its arguments in the `clone_args` struct `args[0]`
            // points at. Copy it out fault-safely (a bad pointer or a size the
            // kernel would reject falls through to a plain forward, so the kernel
            // reports the authoritative `EFAULT`/`EINVAL`/`E2BIG`). Then split the
            // same three ways as `clone`: a thread runs on a host thread; a
            // `CLONE_VM`-without-`CLONE_THREAD` spawn is forked (the stripped
            // flags written back into the guest struct first); everything else is
            // an ordinary forwarded `fork`.
            libc::SYS_clone3 => match read_clone3_args(call.args[0], call.args[1]) {
                Some(cargs) if is_thread_clone(cargs[0]) => {
                    let tid = thread.clone3_vm(&cargs);
                    call.set_return(tid);
                }
                // `vfork`/`posix_spawn` (see the `SYS_clone` arm above).
                Some(cargs)
                    if cargs[0] & libc::CLONE_VM as u64 != 0
                        && cargs[0] & libc::CLONE_THREAD as u64 == 0
                        && cargs[0] & libc::CLONE_VFORK as u64 != 0 =>
                {
                    // `clone_args`: [flags, pidfd, child_tid, parent_tid,
                    // exit_signal, stack, stack_size, tls]. The child runs on
                    // `stack + stack_size`. Strip `CLONE_VM`/`CLONE_VFORK` and
                    // zero the stack fields so the host `fork` keeps Chimera's
                    // own stack, then set the guest child's `rsp` to the stack
                    // top. `clone3` only requires the struct to be readable, so
                    // rather than rewrite the guest's copy (illegal if it is in a
                    // read-only mapping, and observable to the caller after),
                    // forward a private edited copy in Chimera's own memory.
                    let guest_stack_top = if cargs[5] != 0 {
                        cargs[5].wrapping_add(cargs[6])
                    } else {
                        0
                    };
                    let size = call.args[1] as usize;
                    let mut buf = [0u8; CLONE_ARGS_SIZE_MAX as usize];
                    if !copy_from_guest(call.args[0], &mut buf[..size]) {
                        // Unreadable at the declared size: let the kernel report
                        // the authoritative EFAULT by forwarding the original.
                        forked(thread, call, handler, 0);
                    } else {
                        let stripped = cargs[0] & !((libc::CLONE_VM | libc::CLONE_VFORK) as u64);
                        buf[0..8].copy_from_slice(&stripped.to_ne_bytes());
                        buf[40..56].fill(0); // stack, stack_size
                        call.args[0] = buf.as_ptr() as u64;
                        spawned(thread, call, handler, guest_stack_top);
                    }
                }
                // A separate shared-memory process (no `CLONE_VFORK`): refused,
                // as for `clone`.
                Some(cargs)
                    if cargs[0] & libc::CLONE_VM as u64 != 0
                        && cargs[0] & libc::CLONE_THREAD as u64 == 0 =>
                {
                    call.set_result(SyscallResult::Error(libc::EPERM));
                }
                _ => forked(thread, call, handler, 0),
            },
            // `vfork` shares memory and suspends the parent until the child
            // execs/exits. Emulate as `fork` by forwarding a `clone` carrying only
            // the child-exit signal: the child gets a copy-on-write image and
            // `execve`s into a new process. A `vfork` child may only `exec`/`_exit`
            // (so the copy is equivalent), and a `fork` parent need not be
            // suspended.
            libc::SYS_vfork => {
                call.number = libc::SYS_clone as u64;
                call.args = [libc::SIGCHLD as u64, 0, 0, 0, 0, 0];
                forked(thread, call, handler, 0);
            }
            // The variants reached here: genuine `fork` shapes, and malformed
            // thread shapes the kernel rejects with `EINVAL` (the `CLONE_VM`
            // cases were serviced above). Forward as an ordinary fork.
            libc::SYS_clone | libc::SYS_fork => forked(thread, call, handler, 0),
            // `set_tid_address` records the calling thread's `clear_child_tid`
            // word — the one the runtime zeroes and futex-wakes on exit so a
            // joiner returns. It is virtualized, never forwarded: forwarding
            // would point the host kernel at the guest's word and clobber the
            // `clear_child_tid` the host thread runtime relies on. Returns the
            // caller's TID, like the kernel.
            libc::SYS_set_tid_address => {
                let tid = thread.set_clear_child_tid(call.args[0]);
                call.set_return(tid);
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
            // `process_vm_writev` writes into a process's address space out of
            // band: the kernel maps the target pages a second time and copies
            // into them, so the write never passes through the translator and
            // can change the bytes at an already-translated guest PC (a
            // stale-translation hole). Refuse it with `EPERM`. The read-only
            // counterpart `process_vm_readv` mutates nothing and is forwarded.
            libc::SYS_process_vm_writev => {
                call.set_result(SyscallResult::Error(libc::EPERM));
            }
            // `userfaultfd` registers a userspace page-fault handler: when the
            // guest first touches a registered page, a monitor thread chooses
            // the bytes to fill it with (`UFFDIO_COPY`) or which existing page
            // to map (`UFFDIO_CONTINUE`). That lets the guest hand the
            // translator one set of bytes at translation time and a different
            // set at the faulting PC afterwards (a stale-translation hole), and
            // the monitor runs outside the translator. Refuse it with `EPERM`.
            libc::SYS_userfaultfd => {
                call.set_result(SyscallResult::Error(libc::EPERM));
            }
            // The `READ_IMPLIES_EXEC` persona makes the kernel add `PROT_EXEC`
            // to every readable mapping, which would undo the `PROT_EXEC`
            // stripping in the `mmap`/`mprotect` arms above and leave guest
            // pages executable in the host page tables (a W^X defeat). Refuse a
            // `personality` call that sets it with `EPERM`. The query form
            // (`0xffffffff`, returns the current persona unchanged) and benign
            // personas fall through to the default arm and are forwarded.
            libc::SYS_personality
                if call.args[0] as libc::c_uint != 0xffff_ffff
                    && call.args[0] as libc::c_uint & libc::READ_IMPLIES_EXEC as libc::c_uint
                        != 0 =>
            {
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
            libc::SYS_rt_sigpending => {
                let r = thread.signals_mut().sigpending(call.args[0], call.args[1]);
                call.set_result(r);
            }
            libc::SYS_rt_sigsuspend => {
                let r = thread.signals_mut().sigsuspend(call.args[0], call.args[1]);
                call.set_result(r);
            }
            libc::SYS_rt_sigtimedwait => {
                let r = thread.signals_mut().sigtimedwait(
                    call.args[0],
                    call.args[1],
                    call.args[2],
                    call.args[3],
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

    /// Whether clone flags describe a new thread of the calling process. The
    /// kernel demands the full shape — `CLONE_THREAD` requires `CLONE_SIGHAND`,
    /// which requires `CLONE_VM` (anything less is `EINVAL`) — so gating the
    /// host-thread intercept on all three bits means a malformed thread clone
    /// falls through to forwarding and gets the kernel's authoritative error.
    fn is_thread_clone(flags: u64) -> bool {
        const THREAD_SHAPE: u64 =
            libc::CLONE_THREAD as u64 | libc::CLONE_SIGHAND as u64 | libc::CLONE_VM as u64;
        flags & THREAD_SHAPE == THREAD_SHAPE
    }

    /// Forward a `fork`-shaped duplication and fix up the child. The fork runs
    /// under [`Process::lock_for_fork`](crate::process::Process::lock_for_fork)
    /// — the pthread_atfork discipline — so the multithreaded host program's
    /// child inherits every `Process` lock unlocked rather than orphaned-locked
    /// for a sibling that does not exist there. In the child (the call returns
    /// 0) the pending-signal set is cleared and the thread and process
    /// bookkeeping rebuilt around the one surviving thread.
    ///
    /// `guest_stack_top`, when non-zero, is the stack a `vfork`/`posix_spawn`
    /// child must run its guest code on: the caller has already kept it out of
    /// the host `fork` (which would point Chimera's own `rsp` at a guest
    /// address), so the child's guest `rsp` is set to it here.
    fn forked(
        thread: &mut Thread,
        call: &mut SystemCall,
        handler: &dyn SystemCalls,
        guest_stack_top: u64,
    ) {
        let fork_locks = thread.process().lock_for_fork();
        handler.do_syscall(call);
        drop(fork_locks);
        if call.return_value() == 0 {
            thread.signals_mut().reset_pending_after_fork();
            thread.reset_after_fork();
            if guest_stack_top != 0 {
                thread.state.regs[RSP] = guest_stack_top;
            }
        }
    }

    /// Fork emulation for the `vfork`/`posix_spawn` case ([`forked`] plus the
    /// `CLONE_VFORK` semantics its callers rely on). A pipe carries the child's
    /// `execve` outcome back to the parent, which blocks on it exactly as
    /// `CLONE_VFORK` blocks until the child execs or exits: the child's `execve`
    /// closes the pipe on success (parent reads EOF → returns the child PID) or
    /// writes the errno on failure (parent reads it → returns `-errno`). That is
    /// the same negative-clone-return path glibc's `posix_spawn` reports an exec
    /// error through, so a missing or unloadable program fails the call
    /// synchronously rather than only surfacing via the child's exit status.
    fn spawned(
        thread: &mut Thread,
        call: &mut SystemCall,
        handler: &dyn SystemCalls,
        guest_stack_top: u64,
    ) {
        let mut fds = [0i32; 2];
        // Without the pipe, fall back to a plain fork: the spawn still works, it
        // just loses synchronous exec-error reporting.
        if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
            forked(thread, call, handler, guest_stack_top);
            return;
        }
        let read_fd = fds[0];
        // Move the write end clear of the low descriptors the child's file
        // actions typically remap, so a `dup2`/`close` there does not clobber it.
        let mut write_fd = fds[1];
        let moved = unsafe { libc::fcntl(write_fd, libc::F_DUPFD_CLOEXEC, 100) };
        if moved >= 0 {
            unsafe { libc::close(write_fd) };
            write_fd = moved;
        }

        let is_child = {
            let fork_locks = thread.process().lock_for_fork();
            handler.do_syscall(call);
            let is_child = call.return_value() == 0;
            drop(fork_locks);
            is_child
        };

        if is_child {
            unsafe { libc::close(read_fd) };
            thread.set_spawn_report_fd(write_fd);
            thread.signals_mut().reset_pending_after_fork();
            thread.reset_after_fork();
            if guest_stack_top != 0 {
                thread.state.regs[RSP] = guest_stack_top;
            }
            return;
        }

        // Parent: block until the child reports its `execve` outcome.
        let child_pid = call.return_value() as libc::pid_t;
        unsafe { libc::close(write_fd) };
        let mut buf = [0u8; 4];
        let n = unsafe { libc::read(read_fd, buf.as_mut_ptr().cast(), buf.len()) };
        unsafe { libc::close(read_fd) };
        if n == buf.len() as isize {
            let errno = i32::from_ne_bytes(buf);
            if errno != 0 {
                // The child's `execve` failed and it will `_exit`; reap it so it
                // leaves no zombie (the caller gets no PID to wait on), and
                // report the errno.
                unsafe { libc::waitpid(child_pid, ptr::null_mut(), 0) };
                call.set_result(SyscallResult::Error(errno));
            }
        }
        // EOF or a zero errno: the child execed (or exited); the child PID the
        // host fork returned stays as the result.
    }
}

pub use host::syscall;
