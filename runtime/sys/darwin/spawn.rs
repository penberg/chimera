//! The Darwin `fork` trap. A fork's copy-on-write child carries Chimera
//! itself and resumes in translated code, so the trap is forwarded — under
//! the `Process` fork-lock discipline the Linux policy established, and with
//! the child's bookkeeping rebuilt around its one surviving thread. The
//! guest's own libc wrapper brackets the trap with libSystem's atfork
//! handlers, translated, so libSystem repairs itself on both sides; only
//! Chimera's own state is repaired here.

use crate::{SyscallResult, SystemCall};

use super::thread::Thread;

/// Forward the fork trap and rebuild the child's bookkeeping. Returns the trap
/// result and the kernel's is-child flag, which libc's `__fork` stub reads
/// from `x1` to pick its side.
pub fn forked(thread: &mut Thread) -> (SyscallResult, u64) {
    let call = SystemCall::new(2, [0; 8]);
    // The handler's locks ride along with the process's own (see
    // `SystemCalls::lock_for_fork`): a descriptor table pairs memory with
    // kernel state, and a copy landing between the two halves of an update
    // hands the child an entry whose backing the parent went on to close.
    let fork_locks = thread.process().lock_for_fork();
    let handler_locks = thread.process().handler.lock_for_fork();
    let (result, is_child) = super::syscall::host_syscall2(&call);
    drop(handler_locks);
    drop(fork_locks);
    if matches!(result, SyscallResult::Ok(_)) && is_child != 0 {
        thread.signals_mut().reset_pending_after_fork();
        thread.reset_after_fork();
    }
    (result, is_child)
}
