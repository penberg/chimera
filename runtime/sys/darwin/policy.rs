//! The Darwin syscall policy: the driver the arm64 dispatcher calls per guest
//! syscall. It services the calls Chimera must own itself — `thread_set_tsd
//! _base` (forwarding it would clobber the runtime's own `TPIDRRO_EL0`) and the
//! `mmap` family (W^X `PROT_EXEC` stripping plus address-space bookkeeping) —
//! and hands everything else to the embedder hooks. BSD `exit` is intercepted
//! earlier, in the dispatch loop.

use crate::{SyscallResult, SystemCall, SystemCalls, sys::host_syscall};

use super::thread::Thread;

// Darwin/arm64 BSD syscall numbers.
const SYS_FORK: u64 = 2;
const SYS_TERMINATE_WITH_PAYLOAD: u64 = 520;
const SYS_ABORT_WITH_PAYLOAD: u64 = 521;
const SYS_EXECVE: u64 = 59;
const SYS_VFORK: u64 = 66;
const SYS_POSIX_SPAWN: u64 = 244;
const SYS_MMAP: u64 = 197;
/// Mach VM traps, which arrive through the same `svc` path with negative
/// numbers. libpthread reaches for these rather than the BSD calls for some
/// of its own memory — a terminated thread's pthread struct among it — so a
/// policy that watches only `mmap`/`munmap` never sees those regions appear
/// or disappear.
const MACH_VM_ALLOCATE: u64 = (-10i64) as u64;
const MACH_VM_DEALLOCATE: u64 = (-12i64) as u64;
const MACH_VM_MAP: u64 = (-15i64) as u64;
const SYS_MUNMAP: u64 = 73;
const SYS_MPROTECT: u64 = 74;
const SYS_SIGACTION: u64 = 46;
const SYS_SIGPROCMASK: u64 = 48;
const SYS_SIGPENDING: u64 = 52;
const SYS_SIGALTSTACK: u64 = 53;
const SYS_SIGSUSPEND: u64 = 111;
const SYS_SIGSUSPEND_NOCANCEL: u64 = 410;
/// Returns its two descriptors in `(x0, x1)` — the pair-return syscall ABI.
const SYS_PIPE: u64 = 42;
/// `pthread_create`'s kernel primitive: create a new BSD thread.
const SYS_BSDTHREAD_CREATE: u64 = 360;
/// libpthread's thread-registration syscall (redundant with the runtime's).
const SYS_BSDTHREAD_REGISTER: u64 = 366;
/// Machine-dependent syscall that installs the guest's `TPIDRRO_EL0` value.
const THREAD_SET_TSD_BASE: u64 = 0x8000_0000;

/// `__proc_info`, the kernel's process-introspection call. `libproc`'s
/// `proc_pidpath` is `(PROC_INFO_CALL_PIDINFO, pid, PROC_PIDPATHINFO, 0,
/// buffer, buffersize)`.
const SYS_PROC_INFO: u64 = 336;
const PROC_INFO_CALL_PIDINFO: u64 = 2;
const PROC_PIDPATHINFO: u64 = 11;

/// This process's own Mach task port, the `target` a guest passes when it is
/// operating on its own address space.
fn mach_task_self() -> u64 {
    unsafe extern "C" {
        static mut mach_task_self_: libc::c_uint;
    }
    unsafe { mach_task_self_ as u64 }
}

pub fn syscall(thread: &mut Thread, call: &mut SystemCall, handler: &dyn SystemCalls) {
    handler.pre_syscall(call);

    match call.number {
        // The guest is ending the process with a reason (`abort_with_payload`
        // is where a modern `abort()` and the library fork-safety guards
        // land). The kernel keeps the reason for the system crash log only;
        // say it on stderr first, or the death is undiagnosable from the
        // terminal. `args[4]` is the reason string.
        SYS_ABORT_WITH_PAYLOAD | SYS_TERMINATE_WITH_PAYLOAD => {
            let reason = crate::sys::mmap::read_guest_cstr(call.args[4], 1024, 0)
                .map(|s| String::from_utf8_lossy(&s).into_owned())
                .unwrap_or_default();
            eprintln!(
                "chimera: guest aborts (ns={} code={:#x}): {reason}",
                call.args[0], call.args[1]
            );
            handler.do_syscall(call);
        }
        // The Mach VM traps, kept in the same books as `mmap`/`munmap`. Their
        // ranges are guest memory like any other: translations taken from a
        // range that is freed must be dropped, or a later allocation reusing
        // those addresses runs stale code, and a range that is never recorded
        // cannot be reasoned about when a fault lands in it. `args[0]` is the
        // target task — only this process's own mappings are ours to record.
        MACH_VM_DEALLOCATE => {
            let (target, addr, len) = (call.args[0], call.args[1], call.args[2]);
            let result = host_syscall(call);
            if matches!(result, SyscallResult::Ok(_)) && target == mach_task_self() {
                let mut space = thread.addr_space();
                space.remove_region(addr as usize, len as usize);
                space.note_unmap(addr as usize, len as usize);
            }
            call.set_result(result);
        }
        // `mach_vm_allocate(target, &address, size, flags)` and
        // `mach_vm_map(target, &address, size, …)` both return the address
        // through a pointer argument rather than the result register.
        MACH_VM_ALLOCATE | MACH_VM_MAP => {
            let (target, addr_ptr, len) = (call.args[0], call.args[1], call.args[2]);
            let result = host_syscall(call);
            if matches!(result, SyscallResult::Ok(_)) && target == mach_task_self() {
                let mut addr = [0u8; 8];
                if crate::sys::mmap::copy_from_guest(addr_ptr, &mut addr) {
                    let addr = u64::from_ne_bytes(addr) as usize;
                    let mut space = thread.addr_space();
                    space.add_region(addr, len as usize);
                    space.note_map(addr, len as usize);
                }
            }
            call.set_result(result);
        }
        // Runtime-owned: forward it ourselves so the guest-mapping bookkeeping
        // stays authoritative, and strip PROT_EXEC first so a freshly mapped
        // page is never executable in the host page tables (W^X — the guest
        // runs translated, never its own pages natively).
        SYS_MMAP => {
            strip_exec(call);
            strip_map_jit(call);
            // A file-backed mapping names a guest fd that the handler may be
            // virtualizing; swap in the real host fd it resolves to before
            // forwarding, so the kernel maps the right file.
            let fd = call.args[4] as i32;
            if fd >= 0
                && let Some(host_fd) = handler.resolve_fd(fd)
            {
                call.args[4] = host_fd as u64;
            }
            let result = host_syscall(call);
            if let SyscallResult::Ok(addr) = result {
                if crate::trace::trace() {
                    eprintln!("chimera: mmap -> {addr:#x} len={:#x}", call.args[1]);
                }
                let mut space = thread.addr_space();
                space.add_region(addr as usize, call.args[1] as usize);
                space.note_map(addr as usize, call.args[1] as usize);
            }
            call.set_result(result);
        }
        // Not runtime-owned: strip PROT_EXEC, then let the embedder service it,
        // and drop stale translations for the reprotected range afterward.
        SYS_MPROTECT => {
            strip_exec(call);
            handler.do_syscall(call);
            if !matches!(call.result(), Some(SyscallResult::Error(_))) {
                thread
                    .addr_space()
                    .note_prot(call.args[0] as usize, call.args[1] as usize);
            }
        }
        SYS_MUNMAP => {
            let result = host_syscall(call);
            if matches!(result, SyscallResult::Ok(_)) {
                let mut space = thread.addr_space();
                space.remove_region(call.args[0] as usize, call.args[1] as usize);
                space.note_unmap(call.args[0] as usize, call.args[1] as usize);
            }
            call.set_result(result);
        }
        // Forwarding this would overwrite TPIDRRO_EL0, which Chimera's own
        // libSystem uses to find its pthread state — every later runtime libc
        // call would then crash. Record the base in the ctx instead: the
        // translator rewrites every guest `mrs <r>, TPIDRRO_EL0` to read
        // `ctx.guest_tsd`, so the guest observes exactly the base it set while
        // the real register stays bound to the runtime.
        THREAD_SET_TSD_BASE => {
            thread.state.guest_tsd = call.args[0];
            call.set_result(SyscallResult::Ok(0));
        }
        // A guest asking the kernel for its own executable path gets the
        // *runtime's*: Chimera and the guest are one host process, and the
        // kernel knows only the image it exec'd. Tools that re-dispatch on
        // their own identity depend on this (rustup's proxies pick which tool
        // to be from it), so answer with the guest image instead. Only this
        // process's path is virtualized — a query about any other pid is a
        // genuine question about a genuine process, and is forwarded.
        SYS_PROC_INFO
            if call.args[0] == PROC_INFO_CALL_PIDINFO
                && call.args[2] == PROC_PIDPATHINFO
                && call.args[1] as i32 == unsafe { libc::getpid() } =>
        {
            let result = match crate::sys::darwin::write_executable_path(
                call.args[4],
                call.args[5] as usize,
            ) {
                // The kernel reports success as 0 and leaves the caller to
                // measure the string; `proc_pidpath` returns its length.
                Ok(()) => SyscallResult::Ok(0),
                Err(errno) => SyscallResult::Error(errno),
            };
            call.set_result(result);
        }
        // Redundant with the runtime's own libpthread registration; ack it.
        SYS_BSDTHREAD_REGISTER => {
            call.set_result(SyscallResult::Ok(0));
        }
        // Never forwarded: the host kernel would start a native thread running
        // the guest untranslated, outside the sandbox. Refused until guest
        // threads are run on host threads of Chimera's own; `pthread_create`
        // reports the failure to the guest.
        SYS_BSDTHREAD_CREATE => {
            call.set_result(SyscallResult::Error(libc::ENOTSUP));
        }
        // Kernel-side fork+exec in one trap: forwarding it would start the
        // spawned image natively, outside the sandbox. Refused until it is
        // emulated as fork plus the in-process exec; libc's `posix_spawn`
        // hands the error back to the guest.
        SYS_POSIX_SPAWN => {
            call.set_result(SyscallResult::Error(libc::ENOTSUP));
        }
        // Signal virtualization: dispositions, masks, and stacks are recorded
        // by the runtime, never forwarded raw — a forwarded `sigaction` would
        // install an untranslated guest handler on the host slot, and the
        // kernel would then call guest code outside the sandbox.
        SYS_SIGACTION => {
            let a = &call.args;
            let result = thread.signals_mut().sigaction(a[0], a[1], a[2]);
            call.set_result(result);
        }
        SYS_SIGPROCMASK => {
            let a = &call.args;
            let result = thread.signals_mut().sigprocmask(a[0] as i32, a[1], a[2]);
            call.set_result(result);
        }
        SYS_SIGPENDING => {
            let result = thread.signals_mut().sigpending(call.args[0]);
            call.set_result(result);
        }
        SYS_SIGALTSTACK => {
            let a = &call.args;
            let sp = thread.state.sp;
            let result = thread.signals_mut().sigaltstack(a[0], a[1], sp);
            call.set_result(result);
        }
        SYS_SIGSUSPEND | SYS_SIGSUSPEND_NOCANCEL => {
            let result = thread.sigsuspend(call.args[0]);
            call.set_result(result);
        }
        // A fork's copy-on-write child carries Chimera itself and resumes in
        // translated code, so the trap is forwarded — but runtime-owned, for
        // its pair-return ABI: the pid comes back in `x0` and an is-child flag
        // in `x1`, which libc's `__fork` stub tests to pick its side (the
        // generic forward path would discard it). The guest's wrapper brackets
        // the trap with the shared libSystem's own atfork handlers — prepare
        // has already run, translated, when the trap arrives here, and the
        // parent/child handlers run translated after it returns on each side —
        // so libSystem repairs itself in both processes. Chimera repairs only
        // its own state, under the Linux policy's discipline: every `Process`
        // lock held across the copy by the forking thread, and the child's
        // bookkeeping rebuilt around its one surviving thread. (Chimera-side
        // code in this window must not touch libSystem malloc — prepare holds
        // the zone locks — which mimalloc as the global allocator guarantees.)
        //
        // `vfork` must not be forwarded at all: xnu runs a vfork child on the
        // *parent's* thread until it execs or exits, which would re-enter the
        // dispatch loop on this same `ThreadState`. Emulate it as fork — a
        // vfork child may only `exec`/`_exit`, so a copy-on-write image is
        // observationally equivalent, and a fork parent need not be suspended.
        SYS_FORK | SYS_VFORK => {
            let (result, is_child) = crate::sys::darwin::spawn::forked(thread);
            if matches!(result, SyscallResult::Ok(_)) {
                thread.state.regs[1] = is_child;
            }
            call.set_result(result);
        }
        // Intercepted, never forwarded to the host kernel: forwarding would
        // have the host replace Chimera's whole process image — runtime, code
        // cache, and translation map included — with an untranslated program
        // running natively, outside the sandbox.
        //
        // The replacement image is validated (and, on Darwin, already mapped
        // into its own slid reservation) here, in the calling thread, the way
        // the kernel sequences an exec: a failure reports `-errno` to the
        // caller and disturbs no sibling. Only a loadable image commits the
        // exec — publishing it on the shared process stops every thread, and
        // the main host thread waits out the stragglers, tears down the old
        // image, and enters the new one (see `crate::sys::darwin::exec`).
        SYS_EXECVE => match crate::sys::darwin::exec::prepare_exec(&call.args, handler) {
            Ok(prepared) => {
                // First committer wins. A refused commit means a sibling's
                // exec (or an exit_group) is already dissolving this group,
                // this thread with it — the loser never observes a return
                // value: the pending stop takes the thread at its next
                // boundary. Its prepared mapping must not leak, though.
                if let Err(loser) = thread.process().request_exec(prepared, &thread.state) {
                    loser.discard();
                }
                // Report success so observers see the allowed call; the guest
                // never reads it — a successful execve does not return.
                call.set_result(SyscallResult::Ok(0));
            }
            Err(err) => {
                let errno = crate::sys::darwin::exec::exec_errno(&err).unwrap_or(libc::EIO);
                call.set_result(SyscallResult::Error(errno));
            }
        },
        // Runtime-owned because of its pair-return ABI: the second descriptor
        // comes back in `x1`, which the generic forward path discards.
        SYS_PIPE => {
            let (result, fd1) = crate::sys::darwin::syscall::host_syscall2(call);
            if matches!(result, SyscallResult::Ok(_)) {
                thread.state.regs[1] = fd1;
            }
            call.set_result(result);
        }
        _ => handler.do_syscall(call),
    }

    handler.post_syscall(call);
}

/// Clear `PROT_EXEC` from a mapping/protection request's `prot` argument
/// (always `args[2]`), substituting `PROT_READ` so an execute-only request does
/// not collapse to `PROT_NONE` and the translator can still read the bytes.
fn strip_exec(call: &mut SystemCall) {
    let prot = call.args[2] as libc::c_int;
    if prot & libc::PROT_EXEC != 0 {
        call.args[2] = ((prot & !libc::PROT_EXEC) | libc::PROT_READ) as u64;
    }
}

/// `MAP_JIT`, from `<sys/mman.h>` — not in the `libc` crate for this target.
const MAP_JIT: u64 = 0x0800;

/// Clear `MAP_JIT` from an `mmap` request's flags (`args[3]`). A `MAP_JIT`
/// page's writability is governed by the calling thread's APRR state, which
/// Chimera owns for its own code cache — leaving the flag on would make the
/// guest's writes to its JIT page fault whenever Chimera has the thread in
/// execute mode. The guest loses nothing: `PROT_EXEC` is stripped anyway (it
/// never executes its own pages — the translator reads them), so an ordinary
/// read-write mapping is exactly the capability it is left with. Paired with
/// the `pthread_jit_write_protect_np` interception in the dispatch loop.
fn strip_map_jit(call: &mut SystemCall) {
    call.args[3] &= !MAP_JIT;
}
