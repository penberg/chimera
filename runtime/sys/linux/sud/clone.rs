//! The clone family: forks and the `posix_spawn` shape.
//!
//! A fork is forwarded, but the child comes back with its dispatch
//! configuration cleared and must re-arm before its first guest instruction
//! (see [`forward_fork`]). The `posix_spawn` shape, `clone(CLONE_VM |
//! CLONE_VFORK)`, degrades to a fork with a pipe carrying the child's exec
//! outcome back (see [`spawned`]), because a child sharing the arena bump
//! pointer and the `fs` cells would race its parent. A thread-shaped clone
//! is refused: a second native guest thread would race the one thread's
//! state here.

use std::ptr;

use crate::{SyscallResult, SystemCall};

use super::{
    super::syscall::{CLONE_CLEAR_SIGHAND, Clone3Args, host_syscall, is_thread_clone},
    signal::reset_guest_signals,
    sud_on,
    thread::Thread,
};

/// `clone`, split by shape: the `posix_spawn` shape degrades to a fork;
/// `CLONE_VM` in any other form — a thread, or a second process sharing this
/// address space — is refused; anything else is a fork.
pub fn do_clone(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t) {
    let flags = call.args[0];
    let vm = flags & libc::CLONE_VM as u64 != 0;
    let vfork = flags & libc::CLONE_VFORK as u64 != 0;
    if is_thread_clone(flags) || (vm && !vfork) {
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

/// `clone3`, split the same way as [`do_clone`]. A shape that needs patching
/// before it reaches the host is forwarded from a private copy of the
/// `clone_args` (see [`Clone3Args`]); an unreadable or missized struct fails
/// closed rather than being forwarded for the kernel's verdict, since a task
/// the kernel made would run unintercepted.
pub fn do_clone3(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t) {
    let mut cargs = match Clone3Args::read(call.args[0], call.args[1]) {
        Ok(cargs) => cargs,
        Err(errno) => {
            call.set_result(SyscallResult::Error(errno));
            return;
        }
    };
    let mut flags = cargs.flags();
    let vm = flags & libc::CLONE_VM as u64 != 0;
    let vfork = flags & libc::CLONE_VFORK as u64 != 0;
    if is_thread_clone(flags) || (vm && !vfork) {
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
/// The handler's locks are held across the copy, the `pthread_atfork`
/// discipline the translating backend applies for the same reason (see
/// `SystemCalls::lock_for_fork`).
pub fn forward_fork(
    t: &Thread,
    call: &mut SystemCall,
    uc: &mut libc::ucontext_t,
    child_stack: Option<u64>,
) {
    let hold = t.process.handler.lock_for_fork();
    let result = host_syscall(call);
    drop(hold);
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
