//! Guest system-call interception: the [`SystemCall`] value handed to embedder
//! handlers, the [`SystemCalls`] trait, the runtime's syscall driver
//! [`syscall`], and the default [`Passthrough`] handler.

use crate::sys::host_syscall;

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
    /// The argument registers. arm64 passes up to eight (x0..x7) — Darwin Mach
    /// traps such as `mach_msg2_trap` use all of them — while the x86-64 ABI
    /// fills only the first six and leaves the rest zero.
    pub args: [u64; 8],
    return_value: i64,
    /// Whether the outcome is a failure. Recorded as portable intent so the
    /// host writeback ([`crate::sys::write_syscall_result`]) can commit the
    /// error to the guest in whichever way its ABI expects — the sign of the
    /// return register on Linux, the NZCV carry flag on Darwin — rather than
    /// this struct baking one convention in.
    is_error: bool,
    has_result: bool,
}

impl SystemCall {
    /// Record `result` as this call's outcome. This is the portable path
    /// handlers should prefer: the success/failure intent is stored here and
    /// the host writeback ([`crate::sys::write_syscall_result`]) encodes it into
    /// the guest register file per the host ABI. The return slot keeps the
    /// Linux `-errno` sign encoding for `Error`, so [`SystemCall::result`] and
    /// [`SystemCall::return_value`] read back unchanged.
    pub fn set_result(&mut self, result: SyscallResult) {
        match result {
            SyscallResult::Ok(value) => {
                self.return_value = value;
                self.is_error = false;
            }
            SyscallResult::Error(errno) => {
                self.return_value = -(errno as i64);
                self.is_error = true;
            }
        }
        self.has_result = true;
    }

    /// Set the raw value the guest will see in its return register after this
    /// syscall, in the host's own convention (a `-errno` on Linux). Most
    /// handlers should use [`SystemCall::set_result`] instead, which records
    /// the portable success/failure intent for you.
    pub fn set_return(&mut self, value: i64) {
        self.return_value = value;
        self.is_error = false;
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
        // An explicit error flag (from `set_result`) denotes failure. On Linux
        // a raw `-errno` in the kernel's `[-4095, -1]` range (from a
        // `set_return` escape hatch) does too. That range test is meaningless
        // on Darwin, where failure travels in the carry flag and a negative
        // return is a legitimate success value — the `ULF_NO_ERRNO` ulock
        // calls return `-errno` as data, and treating the joiner-wake's
        // `-ENOENT` as a failure sends the guest's libsyscall down `cerror`
        // and aborts `pthread_join`.
        let raw_errno = cfg!(target_os = "linux") && (-4095..0).contains(&self.return_value);
        if self.is_error || raw_errno {
            Some(SyscallResult::Error((-self.return_value) as i32))
        } else {
            Some(SyscallResult::Ok(self.return_value))
        }
    }

    /// Linux-only callers (fork/mremap policy and the register writeback).
    #[cfg(target_os = "linux")]
    pub(crate) fn return_value(&self) -> i64 {
        self.return_value
    }

    pub(crate) fn new(number: u64, args: [u64; 8]) -> Self {
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
/// correctness (`exit`, `execve`, `arch_prctl`, `mmap`, `munmap`, `mremap`,
/// `mprotect`, `pkey_mprotect`).
///
/// Chimera also rewrites the `prot` argument of `mmap`, `mprotect`, and
/// `pkey_mprotect` before servicing them, clearing `PROT_EXEC` (see
/// [`crate::syscall::syscall`]). `pre_syscall` sees the guest's original
/// request; every later stage, including `do_syscall`, sees the rewritten one.
/// `mmap` always stays runtime-owned, while `mprotect`/`pkey_mprotect` are only
/// runtime-owned for ranges entirely inside the guest address space; other
/// ranges still reach `do_syscall`.
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

    /// Map a guest file descriptor to the host descriptor that backs it, for
    /// the runtime-owned `mmap`. A handler that virtualizes descriptors (an fd
    /// table over a userspace filesystem) returns the real host fd a file-backed
    /// `mmap` must use; the default leaves the descriptor untouched, since with
    /// [`Passthrough`] a guest fd already *is* a host fd.
    ///
    /// Only consulted for a file-backed `mmap` (one whose `fd` is not `-1`).
    fn resolve_fd(&self, _guest_fd: i32) -> Option<std::os::fd::RawFd> {
        None
    }

    /// Resolve an `execve`/`execveat` target to the host path the runtime's ELF
    /// loader should read, applying the handler's namespace and confinement.
    /// `dirfd` is the `execveat` directory fd (`AT_FDCWD` for `execve`), `path`
    /// the raw guest pathname, `flags` the `execveat` flags.
    ///
    /// `Some(Ok(path))` names a host file to load; `Some(Err(errno))` fails the
    /// exec with that errno; `None` defers to the runtime's default handling of
    /// the raw path, which is what [`Passthrough`] wants.
    fn resolve_exec(
        &self,
        _dirfd: i32,
        _path: &[u8],
        _flags: i32,
    ) -> Option<Result<std::path::PathBuf, i32>> {
        None
    }

    /// The guest committed an `execve`: drop whatever close-on-exec state the
    /// handler virtualizes, before the runtime's own sweep applies host-side
    /// `FD_CLOEXEC` (see `close_cloexec_fds` in `crate::sys::linux::exec`). A
    /// handler that owns a descriptor table closes its close-on-exec entries
    /// here — their close-on-exec flag lives in the table, not on a host fd,
    /// so the runtime's sweep cannot see it. The default does nothing, since
    /// with [`Passthrough`] every guest fd is a host fd and the sweep alone is
    /// exact. `path` names the replacement image, so a handler that
    /// virtualizes `/proc/self/exe` can track what the link now points at.
    fn on_execve(&self, _path: &std::path::Path) {}

    /// Take every lock the handler owns, to be held across a forwarded `fork`
    /// — the `pthread_atfork` discipline the runtime already applies to its
    /// own locks (see `Process::lock_for_fork` in `crate::process`). fork
    /// copies the whole address space: a handler mutex a *sibling* thread
    /// holds at the copy is locked forever in the child, and handler state
    /// paired with kernel state — a descriptor table and the host fds backing
    /// it — is torn if the copy lands between the two halves of an update.
    /// Held by the forking thread instead, the snapshot is consistent and
    /// both processes' copies of the guards unlock on drop. The returned
    /// bundle is opaque; the runtime only holds it across the fork and drops
    /// it. The default holds nothing, since [`Passthrough`] has no state.
    fn lock_for_fork(&self) -> Box<dyn ForkHold + '_> {
        Box::new(())
    }
}

/// Opaque bundle of handler locks held across a forwarded `fork`; see
/// [`SystemCalls::lock_for_fork`]. Blanket-implemented so a handler returns a
/// plain struct of its `MutexGuard`s.
pub trait ForkHold {}

impl<T: ?Sized> ForkHold for T {}

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

pub use crate::sys::policy::syscall;
