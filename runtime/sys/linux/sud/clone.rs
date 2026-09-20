//! The clone family: threads, forks, and the `posix_spawn` shape.
//!
//! None of it can simply be forwarded. A thread-shaped `clone` would make a
//! task that returns from the syscall *inside Chimera's trap handler*, on the
//! guest's thread stack, with no per-thread state and — since dispatch
//! configuration survives a clone no better than it survives a fork — no
//! interception at all; Chimera creates the host thread itself (see
//! [`spawn_thread`]). A fork *is* forwarded, but the child comes back with
//! its dispatch configuration cleared and must re-arm before its first guest
//! instruction (see [`forward_fork`]). And the `posix_spawn` shape,
//! `clone(CLONE_VM | CLONE_VFORK)`, degrades to a fork with a pipe carrying
//! the child's exec outcome back (see [`spawned`]), because a child sharing
//! the arena bump pointer and the `fs` cells would race its parent.

use std::{ptr, sync::Arc};

use crate::{SyscallResult, SystemCall, sys::mmap::copy_to_guest};

use super::{
    super::syscall::{CLONE_CLEAR_SIGHAND, Clone3Args, host_syscall, is_thread_clone},
    SigsysInfo,
    signal::reset_guest_signals,
    sud_on,
    thread::{Thread, enter_thread},
};

/// `clone`, split by shape: a thread runs on a host thread; the
/// `posix_spawn` shape degrades to a fork; `CLONE_VM` without either — a
/// second process sharing this address space, and with it the arena bump
/// pointer and every thread's state — is refused; anything else is a fork.
pub fn do_clone(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t, info: &SigsysInfo) {
    let flags = call.args[0];
    if is_thread_clone(flags) {
        let result = spawn_thread(
            t,
            uc,
            info,
            CloneRequest {
                flags,
                child_stack: call.args[1],
                parent_tid: call.args[2],
                child_tid: call.args[3],
                tls: call.args[4],
            },
        );
        call.set_result(result);
        return;
    }
    let vm = flags & libc::CLONE_VM as u64 != 0;
    let vfork = flags & libc::CLONE_VFORK as u64 != 0;
    if vm && !vfork {
        call.set_result(SyscallResult::Error(libc::EPERM));
        return;
    }
    if vm && vfork {
        call.args[0] = flags & !(libc::CLONE_VM as u64 | libc::CLONE_VFORK as u64);
        // The stack argument must not reach the kernel with it. `clone` sets
        // the child's stack pointer whatever the flags, so a forwarded fork
        // carrying one comes back *inside Chimera's own trap handler* running
        // on the guest's spawn stack — a few pages with no frame under them —
        // and the first thing the runtime touches faults. Dropped here, the
        // child keeps the parent's stack, copy-on-write, the way a real fork
        // does; the guest still needs to resume on the stack it asked for, so
        // the value is installed into the child's resume context instead.
        let child_stack = (call.args[1] != 0).then_some(call.args[1]);
        call.args[1] = 0;
        spawned(t, call, uc, child_stack);
        return;
    }
    forward_fork(t, call, uc, None);
}

/// `clone3`, split the same three ways as [`do_clone`]. A shape that needs
/// patching before it reaches the host is forwarded from a private copy of
/// the `clone_args` (see [`Clone3Args`]); an unreadable or missized struct
/// fails closed rather than being forwarded for the kernel's verdict, since
/// a thread the kernel made would run unintercepted.
pub fn do_clone3(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t, info: &SigsysInfo) {
    let mut cargs = match Clone3Args::read(call.args[0], call.args[1]) {
        Ok(cargs) => cargs,
        Err(errno) => {
            call.set_result(SyscallResult::Error(errno));
            return;
        }
    };
    let mut flags = cargs.flags();
    if is_thread_clone(flags) {
        let fields = cargs.fields();
        let result = spawn_thread(
            t,
            uc,
            info,
            CloneRequest {
                flags,
                child_stack: cargs.child_stack_top(),
                parent_tid: fields[3],
                child_tid: fields[2],
                tls: fields[7],
            },
        );
        call.set_result(result);
        return;
    }
    let vm = flags & libc::CLONE_VM as u64 != 0;
    let vfork = flags & libc::CLONE_VFORK as u64 != 0;
    if vm && !vfork {
        call.set_result(SyscallResult::Error(libc::EPERM));
        return;
    }
    let mut child_stack = None;
    let is_spawn = vm && vfork;
    if is_spawn {
        flags &= !(libc::CLONE_VM as u64 | libc::CLONE_VFORK as u64);
        // See `do_clone`: the stack must not reach the kernel with a
        // fork-shaped clone.
        child_stack = Some(cargs.child_stack_top()).filter(|&top| top != 0);
        cargs.clear_stack();
    }
    // `CLONE_CLEAR_SIGHAND` is emulated on the guest's virtual table in the
    // child instead, where it means what the guest asked for — caught
    // handlers revert to `SIG_DFL`, ignored ones stay ignored.
    let clear_sighand = flags & CLONE_CLEAR_SIGHAND != 0;
    flags &= !CLONE_CLEAR_SIGHAND;
    cargs.set_flags(flags);
    call.args[0] = cargs.as_ptr();
    if is_spawn {
        spawned(t, call, uc, child_stack);
    } else {
        forward_fork(t, call, uc, child_stack);
    }
    if clear_sighand && matches!(call.result(), Some(SyscallResult::Ok(0))) {
        reset_guest_signals(t);
    }
}

/// The arguments a thread-creating `clone` carries, in whichever shape it
/// arrived.
struct CloneRequest {
    flags: u64,
    child_stack: u64,
    parent_tid: u64,
    child_tid: u64,
    tls: u64,
}

/// Create a guest thread on a host thread of its own.
///
/// The child enters guest code exactly where the kernel would have put it:
/// at the instruction after the guest's own `syscall`, with the parent's
/// register file, `rax` zeroed to report the child's side of the clone, and
/// its own stack. The parent gets the child's kernel TID, which is the TID
/// the guest sees, so its later `futex` and `tgkill` reach this host thread.
fn spawn_thread(
    t: &Thread,
    uc: &libc::ucontext_t,
    info: &SigsysInfo,
    req: CloneRequest,
) -> SyscallResult {
    let process = Arc::clone(&t.process);
    let mut child_ctx = uc.uc_mcontext.gregs;
    child_ctx[libc::REG_RAX as usize] = 0;
    child_ctx[libc::REG_RSP as usize] = req.child_stack as libc::greg_t;
    child_ctx[libc::REG_RIP as usize] = info.call_addr as libc::greg_t;
    // `CLONE_SETTLS` gives the child its own thread pointer; without it the
    // child inherits the parent's, as the kernel does.
    let guest_fs = if req.flags & libc::CLONE_SETTLS as u64 != 0 {
        req.tls
    } else {
        t.guest_fs.get()
    };
    let clear_child_tid = (req.flags & libc::CLONE_CHILD_CLEARTID as u64 != 0
        && req.child_tid != 0)
        .then_some(req.child_tid);
    let inherited_mask = t.sig.mask.get();

    // The parent must return the child's TID, but only the child can read its
    // own; hand it back over a one-shot channel and wait for it.
    let (tx, rx) = std::sync::mpsc::channel::<i32>();
    let spawned = std::thread::Builder::new()
        .name("chimera-guest".to_string())
        .spawn(move || {
            // Leaked, not stack-held: the `gs` base points at this for as long
            // as the thread runs guest code, and the trap handler dereferences
            // it from contexts that know nothing of this frame.
            let child: &'static Thread = Box::leak(Box::new(Thread::new(process, false)));
            child.guest_fs.set(guest_fs);
            child.clear_child_tid.set(clear_child_tid);
            // A new thread inherits its creator's signal mask.
            child.sig.mask.set(inherited_mask);

            // Replicate the kernel's set-TID writes before any guest code
            // runs: the kernel fills these at clone time, so the child must
            // observe its own TID from its first instruction. glibc points
            // them at the thread's control block and reads the value during
            // early thread setup and as the thread's identity for, among
            // other things, `pthread_rwlock` writer ownership. Both are
            // guest-controlled addresses, so the stores are best-effort — the
            // kernel's own `put_user` there is unchecked.
            if req.flags & libc::CLONE_PARENT_SETTID as u64 != 0 {
                copy_to_guest(req.parent_tid, &child.tid.get().to_ne_bytes());
            }
            if req.flags & libc::CLONE_CHILD_SETTID as u64 != 0 {
                copy_to_guest(req.child_tid, &child.tid.get().to_ne_bytes());
            }
            let _ = tx.send(child.tid.get());

            let code = match enter_thread(child, &child_ctx) {
                Ok(code) => code,
                Err(err) => {
                    eprintln!("chimera: guest thread failed: {err}");
                    127
                }
            };
            // A `fork` in this thread made it the only thread — and the
            // leader — of a whole new process (see `forward_fork`). This host
            // thread is all that process has, so its guest's status is the
            // process's, and simply returning would end the thread and leave
            // the process to exit 0 behind it.
            if child.is_leader.get() {
                std::process::exit(code);
            }
        });

    match spawned {
        // The handle is dropped: the host thread is detached and reclaims
        // itself when its closure returns, and the child is tracked by its
        // kernel TID rather than by a retained handle, which under thread
        // churn would only accumulate.
        Ok(_handle) => match rx.recv() {
            Ok(tid) => SyscallResult::Ok(tid as i64),
            Err(_) => SyscallResult::Error(libc::EAGAIN),
        },
        Err(_) => SyscallResult::Error(libc::EAGAIN),
    }
}

/// Forward a fork-shaped call and re-arm dispatch in the child.
///
/// The kernel does **not** inherit the syscall-user-dispatch configuration
/// across `fork`/`clone`: the child's `SYSCALL_WORK_SYSCALL_USER_DISPATCH`
/// work flag is cleared, so without this its every syscall would go straight
/// to the host kernel — the guest's child escaping the sandbox entirely, and
/// silently, since an escaped syscall succeeds. The child re-arms here,
/// before it returns to guest code, so the first guest instruction it
/// executes is already intercepted. This is the one place a fork is
/// forwarded, and the whole backend's confinement of child processes rests
/// on it.
///
/// The runtime's and the handler's locks are held across the copy — the
/// `pthread_atfork` discipline, applied at the one place a fork is forwarded
/// (see `Process::lock_for_fork` and `SystemCalls::lock_for_fork`).
pub fn forward_fork(
    t: &Thread,
    call: &mut SystemCall,
    uc: &mut libc::ucontext_t,
    child_stack: Option<u64>,
) {
    let process_hold = t.process.lock_for_fork();
    let handler_hold = t.process.handler.lock_for_fork();
    let result = host_syscall(call);
    drop(handler_hold);
    drop(process_hold);
    if let SyscallResult::Ok(0) = result {
        // The guest asked for its child to run on a stack of its own (the
        // `posix_spawn` shape); the kernel was not allowed to install it, so
        // it goes into the context the child resumes through.
        if let Some(sp) = child_stack {
            uc.uc_mcontext.gregs[libc::REG_RSP as usize] = sp as libc::greg_t;
        }
        sud_on();
        // The pid the guest-memory writes are aimed at is cached, and the
        // cache is the parent's. Left stale, every `copy_to_guest` in the
        // child would land in the *parent's* address space and report
        // success — the child's own writes silently lost and the parent's
        // memory corrupted. libc's `pthread_atfork` hook does not cover this
        // fork: the guest's `clone` is forwarded as a raw syscall and never
        // runs libc's handlers.
        crate::sys::mmap::reset_cached_pid();
        // POSIX hands the child an empty pending set. The kernel clears its
        // own; the deferred set Chimera keeps is ordinary memory the fork
        // copied, so it has to be cleared by hand or the child would take a
        // signal only its parent was sent.
        t.sig.pending.clear();
        t.become_fork_child();
    }
    call.set_result(result);
}

/// The `posix_spawn` shape, which needs more than a fork.
///
/// glibc issues `clone(CLONE_VM | CLONE_VFORK)` and relies on both flags: it
/// stays suspended until the child execs or exits, and reads the child's
/// error out of the memory they share. A fork gives neither, so the outcome
/// travels back over a pipe instead and the parent blocks on it, which is
/// what makes a missing program fail `posix_spawn` synchronously with
/// `ENOENT` rather than only surfacing as the child's exit status.
///
/// The report waits for the child's *exit*, not its first failed `execve`:
/// `posix_spawnp` walks `$PATH` inside the child, one `execve` per candidate,
/// and an early failure is routinely followed by one that succeeds.
fn spawned(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t, child_stack: Option<u64>) {
    let mut fds = [0i32; 2];
    // Without the pipe the spawn still works; it just loses the synchronous
    // error report.
    if unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        forward_fork(t, call, uc, child_stack);
        return;
    }
    let read_fd = fds[0];
    // Move the write end clear of the low descriptors a spawn's file actions
    // typically remap, so the child's own `dup2`/`close` cannot clobber it.
    let mut write_fd = fds[1];
    let moved = unsafe { libc::fcntl(write_fd, libc::F_DUPFD_CLOEXEC, 100) };
    if moved >= 0 {
        unsafe { libc::close(write_fd) };
        write_fd = moved;
    }

    forward_fork(t, call, uc, child_stack);
    let result = call.result();

    if let Some(SyscallResult::Ok(0)) = result {
        unsafe { libc::close(read_fd) };
        t.spawn_report_fd.set(Some(write_fd));
        return;
    }
    unsafe { libc::close(write_fd) };
    let Some(SyscallResult::Ok(child_pid)) = result else {
        unsafe { libc::close(read_fd) };
        return;
    };

    let mut buf = [0u8; 4];
    let n = unsafe { libc::read(read_fd, buf.as_mut_ptr().cast(), buf.len()) };
    unsafe { libc::close(read_fd) };
    if n == buf.len() as isize {
        let errno = i32::from_ne_bytes(buf);
        if errno != 0 {
            // The child's exec failed and it is about to `_exit`; reap it so
            // it leaves no zombie — the caller never gets a PID to wait on —
            // and report the errno the way the shared-memory path would have.
            unsafe { libc::waitpid(child_pid as libc::pid_t, ptr::null_mut(), 0) };
            call.set_result(SyscallResult::Error(errno));
        }
    }
}

/// Report a spawn child's `execve` outcome to its blocked parent. A committed
/// exec closes the pipe with nothing written, which the parent reads as EOF
/// and takes for success; a child that exits without one writes the errno of
/// its last failed attempt.
pub fn report_spawn(t: &Thread, errno: i32) {
    let Some(fd) = t.spawn_report_fd.take() else {
        return;
    };
    if errno != 0 {
        let buf = errno.to_ne_bytes();
        unsafe { libc::write(fd, buf.as_ptr().cast(), buf.len()) };
    }
    unsafe { libc::close(fd) };
}
