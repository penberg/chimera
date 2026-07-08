//! The translate-execute loop and the guest register file it operates on.
//!
//! `ThreadState` holds the guest register file. The GS segment base is set to
//! point at it via `arch_prctl`, so translated code can reach any field with a
//! plain `gs:[disp]` access — no reserved GPR required.
//!
//! [`Thread::run`] is the loop: deliver any pending guest signal, translate the
//! next block if it isn't already cached, enter the cache through [`dispatch`],
//! handle whatever caused the cache to exit (a block boundary or a syscall), and
//! repeat until the guest issues `exit_group` or `exit`. The boundary-crossing
//! assembly lives in [`super::trampoline`].

use std::{
    arch::asm,
    sync::{
        Arc, MutexGuard,
        atomic::{AtomicI32, AtomicU32, Ordering},
        mpsc,
    },
    thread,
};

use crate::{
    Error, SystemCall,
    process::Process,
    sys::{
        linux::signal::Signals,
        mmap::{AddressSpace, copy_from_guest, copy_to_guest},
    },
};

use super::trampoline::{dispatch, exit_block, exit_syscall, exit_trap};

/// Linux x86-64 `arch_prctl` subfunction code: set the GS base.
const ARCH_SET_GS: libc::c_int = 0x1001;

const EXIT_KIND_BLOCK: u64 = 0;
pub const EXIT_KIND_SYSCALL: u64 = 1;
/// A guest breakpoint (`int3`) exited the cache: the run loop raises `SIGTRAP`.
pub const EXIT_KIND_TRAP: u64 = 2;

/// `SIGTRAP`, raised on a guest `int3`.
const SIGTRAP: u32 = 5;

/// `SIGSEGV`, raised on a guest jump to unmapped memory (a fetch fault).
const SIGSEGV: u32 = 11;

/// Why [`Thread::run`] returned: the guest terminated, or it `execve`d and the
/// caller must load the new image and re-enter the thread on it.
pub enum ExitReason {
    /// The guest issued `exit`/`exit_group`. Carries the exit code.
    Exited(i32),
    /// A guest thread committed an `execve`/`execveat`: the validated, parsed
    /// replacement image is published on the shared [`Process`], and the
    /// caller must wait out the dissolving thread group, then install the
    /// image and re-enter the run (see `crate::sys::linux::exec`).
    Execve,
}

/// Size of the XSAVE area in [`ThreadState::fpstate`]. The standard (non-
/// compacted) XSAVE layout for x87+SSE+AVX+AVX-512 ends at architecturally
/// fixed offsets totaling ~2688 bytes; 4096 leaves comfortable margin. The
/// trampoline saves/restores with the `0xe7` component mask, which never
/// selects AMX, so the area size is bounded regardless of the host's XCR0.
const XSAVE_AREA_SIZE: usize = 4096;

/// Removes a thread from the process's thread list when its run loop ends, on
/// every exit path. Holds the thread's identity — the address of its
/// `ThreadState`, stable across a fork where the TID is not — which
/// unregistration compares and never dereferences. See
/// [`Process::register_thread`].
struct ThreadGuard {
    process: Arc<Process>,
    state: *const ThreadState,
}

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        self.process.unregister_thread(self.state);
    }
}

/// The `Thread` struct represents a guest thread: the per-thread register state
/// and a handle to the process-wide [`Process`] (the shared address space and
/// code cache) it runs in. The purpose of this struct is similar to the
/// `task_struct` in the Linux kernel. `running` and `exit_code` mirror the
/// kernel's task state: a syscall implementation can mark a thread done by
/// clearing `running` and recording an `exit_code`, and the run loop
/// terminates on its next iteration.
pub struct Thread {
    pub state: Box<ThreadState>,
    /// The shared process state. Every thread of a guest process holds an `Arc`
    /// to the same `Process`, so they translate into and map within one address
    /// space.
    process: Arc<Process>,
    /// Per-process guest signal state: handler table, blocked mask, alt stack.
    signals: Signals,
    /// Whether the run loop should keep iterating. Set true on entry to
    /// [`Thread::run`]; cleared by the `exit`/`exit_group` syscall
    /// implementation in [`crate::syscall::syscall`].
    pub running: bool,
    /// The status code the run loop returns once `running` is cleared.
    pub exit_code: i32,
    /// Set when the most recently forwarded syscall returned `EINTR` and is
    /// restartable: `(resume rip after the syscall, original syscall number)`.
    /// Consumed at the next signal delivery to honor `SA_RESTART`. Cleared
    /// after every syscall, so it reflects only the immediately preceding one.
    restart: Option<(u64, u64)>,
    /// Guest address of this thread's `clear_child_tid` word, set by
    /// `set_tid_address` or a `clone` with `CLONE_CHILD_CLEARTID`. When the
    /// thread exits, the runtime zeroes this word and futex-wakes it, exactly as
    /// the kernel does — this is what a joiner (`pthread_join`) blocks on.
    clear_child_tid: Option<u64>,
    /// Whether this is the process's initial thread. The main thread's run
    /// returning ends the process, so when it exits on its own (`pthread_exit`)
    /// while siblings are still alive, it must wait for them rather than tear the
    /// process down. A `clone(CLONE_VM)` child is never main.
    is_main: bool,
    /// Write end of the report pipe a `vfork`/`posix_spawn` child uses to hand
    /// its `execve` outcome back to the parent that is blocked in the clone
    /// (see the spawn path in `crate::syscall`). Set only in such a child; its
    /// `execve` writes the outcome and closes it, so the parent — which mirrors
    /// `CLONE_VFORK` by blocking on the read end — can return the child PID or
    /// the exec errno synchronously.
    spawn_report_fd: Option<i32>,
    /// The errno of the spawn child's most recent failed `execve` attempt,
    /// reported to the parent only when the child exits without a committed
    /// exec (see [`Thread::report_spawn_failure`]).
    spawn_exec_errno: Option<i32>,
}

impl Thread {
    /// Create a new guest thread that runs in the given shared [`Process`]. The
    /// process's first thread is created with a fresh `Process`; future
    /// `clone(CLONE_VM)` siblings are created with a clone of this `Arc`, so
    /// they share the address space, code cache, and syscall handler.
    pub fn new(process: Arc<Process>, rip: u64, rsp: u64) -> Result<Self, Error> {
        let guest_fs_base = current_fs_base();
        let signals = Signals::new(Arc::clone(&process.sig_table));
        let mut thread = Self {
            state: Box::new(ThreadState {
                regs: [0; 16],
                rip: 0,
                rflags: 0,
                chimera_rsp: 0,
                host_pc_target: 0,
                exit_kind: 0,
                guest_fs_base: 0,
                chimera_fs_base: 0,
                ib_lookup: 0,
                ib_flags: 0,
                ib_target: 0,
                ib_rcx: 0,
                ib_rdx: 0,
                ib_host: 0,
                exit_requested: AtomicU32::new(0),
                fp_in_regs: 0,
                fp_flags: 0,
                fp_scratch: 0,
                fpstate: [0; XSAVE_AREA_SIZE],
                fs_is_guest: 0,
                pending_set: 0,
                tid: AtomicI32::new(0),
            }),
            process,
            signals,
            running: false,
            exit_code: 0,
            restart: None,
            clear_child_tid: None,
            is_main: true,
            spawn_report_fd: None,
            spawn_exec_errno: None,
        };
        thread.state.pending_set = thread.signals.pending_set_ptr() as u64;
        thread.reset(rip, rsp, guest_fs_base);
        Ok(thread)
    }

    /// Assemble a thread from an already-built register state that runs in the
    /// given shared process. Used for `clone(CLONE_VM)` children: they inherit
    /// the parent's process (address space, code cache, handler) and a copy of
    /// its register file. The signal-disposition table is shared with the rest
    /// of the thread group (via the process), as POSIX requires; the per-thread
    /// blocked mask and alternate stack start cleared.
    /// `clear_child_tid` carries the `CLONE_CHILD_CLEARTID` word, if any.
    fn from_state(
        process: Arc<Process>,
        mut state: Box<ThreadState>,
        clear_child_tid: Option<u64>,
    ) -> Self {
        let signals = Signals::new(Arc::clone(&process.sig_table));
        state.pending_set = signals.pending_set_ptr() as u64;
        Self {
            state,
            process,
            signals,
            running: false,
            exit_code: 0,
            restart: None,
            clear_child_tid,
            is_main: false,
            spawn_report_fd: None,
            spawn_exec_errno: None,
        }
    }

    /// Whether this is (or has become) the process's initial thread; see
    /// [`Thread::reset_after_fork`] for how a clone child is promoted.
    pub fn is_main(&self) -> bool {
        self.is_main
    }

    /// Rebuild bookkeeping in the child of a fork. The caller is the child's
    /// only thread and its thread-group leader, whatever it was in the
    /// parent: its run returning must end the process with the guest's
    /// status, and its `execve` must take the image-installing path, so it
    /// is promoted to main and the copied [`Process`] group state is reset
    /// around it.
    pub fn reset_after_fork(&mut self) {
        // The guest-copy pid cache still holds the parent's pid; drop it
        // before anything reads guest memory in the child.
        crate::sys::mmap::reset_cached_pid();
        self.is_main = true;
        self.state.tid.store(
            unsafe { libc::syscall(libc::SYS_gettid) } as i32,
            Ordering::Release,
        );
        self.process.reset_after_fork(&self.state);
    }

    /// Record this thread's `clear_child_tid` word (from `set_tid_address`). A
    /// null pointer clears it. Returns the caller's TID (its host `gettid`), the
    /// way the kernel's `set_tid_address` does.
    pub fn set_clear_child_tid(&mut self, addr: u64) -> i64 {
        self.clear_child_tid = (addr != 0).then_some(addr);
        unsafe { libc::syscall(libc::SYS_gettid) }
    }

    /// On thread exit, honor `CLONE_CHILD_CLEARTID`/`set_tid_address`: zero the
    /// `clear_child_tid` word and wake one futex waiter on it, just as the kernel
    /// does for a real task. A `pthread_join` blocked on that word (the joined
    /// thread's TID slot) then returns. The wake is non-private, matching the
    /// kernel's clear-child-tid wake and glibc's wait on it.
    ///
    /// The word is guest memory the guest may have unmapped by now, so the
    /// store is a fault-safe best-effort write, exactly the kernel's exit
    /// path: its `put_user(0, clear_child_tid)` is unchecked, the futex wake
    /// is attempted regardless (`FUTEX_WAKE` on a bad address just returns
    /// `EFAULT`), and the thread exits either way.
    fn clear_tid_and_wake(&self) {
        let Some(addr) = self.clear_child_tid else {
            return;
        };
        copy_to_guest(addr, &0u32.to_ne_bytes());
        unsafe {
            libc::syscall(libc::SYS_futex, addr, libc::FUTEX_WAKE, 1, 0, 0, 0);
        }
    }

    /// Service a `clone`/`clone3` that creates a new thread of this process
    /// (the `CLONE_THREAD` shape, which the kernel requires to include
    /// `CLONE_SIGHAND` and `CLONE_VM`). Chimera cannot forward it: the host
    /// kernel would run guest code on the new task natively, outside the
    /// translator. Instead it spawns a host thread that runs the child guest
    /// in the shared process, and returns the child's kernel TID to the
    /// parent. The guest-visible TID is
    /// the host TID, so the guest's later `futex`/`tgkill` on it reach this host
    /// thread. On spawn failure it returns `-EAGAIN`.
    ///
    /// `child_stack` is the child's stack pointer (its top); the child resumes
    /// at the same post-syscall PC as the parent with `rax = 0` and its own
    /// stack (see [`ThreadState::clone_for_child`]). The remaining arguments are
    /// the thread-ID and TLS words the relevant flags select.
    fn spawn_clone(
        &self,
        flags: u64,
        child_stack: u64,
        parent_tid: u64,
        child_tid: u64,
        tls: u64,
    ) -> i64 {
        // `CLONE_SETTLS` gives the child its own thread pointer; without it the
        // child inherits the parent's FS base, as the kernel does. This is how
        // each pthread gets private TLS.
        let settls = (flags & libc::CLONE_SETTLS as u64 != 0).then_some(tls);
        // `CLONE_CHILD_CLEARTID` registers a word the runtime zeroes and wakes
        // when the child exits (the basis of `pthread_join`).
        let clear_child_tid =
            (flags & libc::CLONE_CHILD_CLEARTID as u64 != 0 && child_tid != 0).then_some(child_tid);
        let child_state = self.state.clone_for_child(child_stack, settls);
        let process = Arc::clone(&self.process);

        // The parent needs the child's kernel TID to return from `clone`, but
        // only the child can read its own `gettid`, so hand it back over a
        // one-shot channel and block until it arrives.
        let (tx, rx) = mpsc::channel::<i64>();
        let spawned = thread::Builder::new()
            .name("chimera-guest".to_string())
            .spawn(move || {
                let tid = unsafe { libc::syscall(libc::SYS_gettid) } as i64;
                // Replicate the kernel's set-TID writes *before* running any guest
                // code. The kernel populates the TID word(s) at clone time, so the
                // child observes its own TID from the first instruction. glibc
                // points these at `&pd->tid` (the thread's TCB identity) and reads
                // it during early thread setup; it is also the thread's identity
                // for, e.g., `pthread_rwlock` writer ownership (`__cur_writer`).
                // Writing from the parent after the child is already running would
                // race those reads with a stale TID.
                // Both words are guest-controlled addresses, so the stores are
                // fault-safe best-effort writes, matching the kernel exactly:
                // its `put_user` for the set-TID words is unchecked, and a
                // clone whose TID pointers are bogus still succeeds with the
                // writes silently skipped (verified natively).
                if tid > 0 {
                    if flags & libc::CLONE_PARENT_SETTID as u64 != 0 {
                        copy_to_guest(parent_tid, &(tid as i32).to_ne_bytes());
                    }
                    if flags & libc::CLONE_CHILD_SETTID as u64 != 0 {
                        copy_to_guest(child_tid, &(tid as i32).to_ne_bytes());
                    }
                }
                // A failed send means the parent already gave up; nothing to do.
                let _ = tx.send(tid);
                let mut child = Thread::from_state(process, child_state, clear_child_tid);
                // The child runs until its guest issues `exit`/`exit_group`,
                // which returns the run loop and ends this host thread — the
                // rest of the process keeps running.
                let reason = child.run();
                // A fork inside this thread made it the main — and only —
                // thread of a new process (see `Thread::reset_after_fork`).
                // This host thread is all that child process has, so drive it
                // the way `execv` drives the initial thread — installing any
                // committed execve image and re-entering — and end the
                // process with the guest's status. Nothing to clear-and-wake:
                // a fork child has no joiner inside it.
                if child.is_main()
                    && let Ok(reason) = reason
                {
                    let code =
                        crate::sys::linux::exec::drive(&mut child, reason).unwrap_or_else(|err| {
                            eprintln!("chimera: fork child failed: {err}");
                            127
                        });
                    std::process::exit(code);
                }
                // Honor CLONE_CHILD_CLEARTID: clear the word and wake a joiner.
                child.clear_tid_and_wake();
            });

        match spawned {
            // Drop the join handle: the host thread is detached and reclaims
            // itself when its closure returns. The child is tracked by its
            // kernel TID through the shared address space, never by a retained
            // handle (an accumulating roster would leak under thread churn).
            Ok(_handle) => rx.recv().unwrap_or(-(libc::EAGAIN as i64)),
            Err(_) => -(libc::EAGAIN as i64),
        }
    }

    /// `clone(CLONE_VM)`: the arguments are in registers. `args[1]` is the child
    /// stack pointer; `args[2..=4]` are `parent_tid`, `child_tid`, `tls`.
    pub fn clone_vm(&self, args: &[u64; 6]) -> i64 {
        self.spawn_clone(args[0], args[1], args[2], args[3], args[4])
    }

    /// `clone3(CLONE_VM)`: the arguments come from the base `clone_args`
    /// struct already copied out of guest memory by [`read_clone3_args`], as
    /// 8 `u64` fields (uapi `<linux/sched.h>` order): flags, pidfd, child_tid,
    /// parent_tid, exit_signal, stack, stack_size, tls. Unlike `clone`, the
    /// `stack` field is the *lowest* address of the child stack and
    /// `stack_size` its length, so the child's stack pointer is their sum.
    /// Modern glibc's `pthread_create` takes this path.
    pub fn clone3_vm(&self, args: &[u64; 8]) -> i64 {
        let flags = args[0];
        let child_tid = args[2];
        let parent_tid = args[3];
        let child_stack = args[5].wrapping_add(args[6]); // stack + stack_size
        let tls = args[7];
        self.spawn_clone(flags, child_stack, parent_tid, child_tid, tls)
    }

    /// Reset the thread to a new entry point and a stack.
    pub fn reset(&mut self, rip: u64, rsp: u64, guest_fs_base: u64) {
        self.addr_space().reset();
        self.state.reset(rip, rsp, guest_fs_base);
        self.running = false;
        self.exit_code = 0;
    }

    /// Lock and return the shared guest address space. The guard must be dropped
    /// before re-entering the cache or issuing a blocking syscall — never held
    /// across `dispatch()`.
    pub fn addr_space(&self) -> MutexGuard<'_, AddressSpace> {
        self.process.addr_space.lock().unwrap()
    }

    /// The shared process [`Process`] this thread runs in. Used by the syscall
    /// layer to publish a process-wide `exit_group` request.
    pub fn process(&self) -> &Process {
        &self.process
    }

    pub fn signals_mut(&mut self) -> &mut Signals {
        &mut self.signals
    }

    /// Record the write end of a spawn child's `execve`-outcome report pipe.
    pub fn set_spawn_report_fd(&mut self, fd: i32) {
        self.spawn_report_fd = Some(fd);
        self.spawn_exec_errno = None;
    }

    /// A spawn child's `execve` succeeded: close the report pipe so the parent
    /// blocked on its read end sees EOF and returns the child PID. A no-op on a
    /// thread that is not a spawn child.
    pub fn report_spawn_success(&mut self) {
        if let Some(fd) = self.spawn_report_fd.take() {
            unsafe { libc::close(fd) };
        }
    }

    /// A spawn child's `execve` failed. Not reported to the parent yet: glibc's
    /// `posix_spawnp` walks `$PATH` inside the child, issuing one `execve` per
    /// candidate, so a failure may be followed by a retry that succeeds —
    /// natively the parent learns the errno from the `CLONE_VM`-shared spawn
    /// state only once the vfork child dies. The last failure becomes final
    /// when the child exits without a committed exec
    /// ([`Thread::report_spawn_exit`]).
    pub fn report_spawn_failure(&mut self, errno: i32) {
        if self.spawn_report_fd.is_some() {
            self.spawn_exec_errno = Some(errno);
        }
    }

    /// A spawn child is exiting without a committed `execve`: send the last
    /// exec failure, if any, to the blocked parent (which returns it from
    /// `posix_spawn`) and close the pipe. A child that never attempted an exec
    /// reports nothing — the parent sees EOF, returns the PID, and the exit
    /// status travels through `waitpid` as `CLONE_VFORK` semantics demand. A
    /// no-op on a thread that is not a spawn child.
    pub fn report_spawn_exit(&mut self) {
        if let Some(fd) = self.spawn_report_fd.take() {
            if let Some(errno) = self.spawn_exec_errno.take() {
                let bytes = errno.to_ne_bytes();
                unsafe {
                    libc::write(fd, bytes.as_ptr().cast(), bytes.len());
                }
            }
            unsafe { libc::close(fd) };
        }
    }

    /// Restore the pre-signal guest context on a guest `rt_sigreturn`.
    pub fn sigreturn(&mut self) {
        let state = &mut *self.state;
        self.signals.restore(state);
    }

    /// Deliver one pending, unblocked guest signal at a safe point (a block
    /// boundary), building its frame and redirecting the guest to the handler.
    /// Entry into the handler then happens through the normal `dispatch` path.
    fn deliver_pending_signals(&mut self) {
        if let Some(signo) = self.signals.pending_take_one() {
            let restart = self.restart.take();
            let state = &mut *self.state;
            self.signals.deliver(state, signo, restart);
        }
    }

    /// Recompute the safepoint exit flag the translated loop-closing polls read.
    /// It is set exactly when a signal is pending and not blocked, so a fully
    /// linked guest loop is forced back into this run loop within one iteration to
    /// deliver it; once nothing is deliverable it clears, leaving warm loops
    /// poll-free at runtime. Clear first, then re-arm only if deliverable — never
    /// re-clearing — so a same-thread catcher that sets the flag from a signal
    /// handler between the clear and the recheck is not lost. The `compiler_fence`
    /// keeps the clear ordered before the recheck (signal delivery on this thread
    /// is itself a serialization point, so no CPU fence is needed).
    fn refresh_exit_requested(&mut self) {
        self.state.exit_requested.store(0, Ordering::Relaxed);
        core::sync::atomic::compiler_fence(Ordering::SeqCst);
        let deliverable = self.signals.pending_snapshot() & !self.signals.blocked;
        if deliverable != 0 {
            self.state.exit_requested.store(1, Ordering::Relaxed);
        }
    }

    /// Set the thread's entry registers for a freshly mapped image, without
    /// touching its address space. Used to re-enter after an `execve` once the
    /// caller has torn down the old image and mapped the new one.
    pub fn enter(&mut self, rip: u64, rsp: u64) {
        self.state.reset(rip, rsp, current_fs_base());
    }

    /// Run the guest using the thread's current entry state. Returns when the
    /// guest issues `exit`/`exit_group` (with the code) or an allowed
    /// `execve`/`execveat` (for the caller to act on); neither syscall is
    /// forwarded to the host kernel. The handler observes the call first. The
    /// embedder's handler is reached through the shared [`Process`].
    pub fn run(&mut self) -> Result<ExitReason, Error> {
        // GS is host-thread-local, so bind it on the OS thread that is
        // actually about to execute the translated guest.
        self.setup_gs()?;
        self.state.capture_chimera_fs();
        self.signals.mirror_host_mask();
        self.running = true;

        // Join the thread list so a sibling's process-wide stop can reach
        // this thread: the interrupt signal if it parks in a host syscall, the
        // registered `exit_requested` safepoint slot if it is executing fully
        // linked translated code. The guard removes it on every exit path
        // (including the `execve` early return).
        self.state.tid.store(
            unsafe { libc::syscall(libc::SYS_gettid) } as i32,
            Ordering::Release,
        );
        self.process.register_thread(&self.state);
        let _thread_guard = ThreadGuard {
            process: Arc::clone(&self.process),
            state: &*self.state,
        };

        let block_exit = exit_block as *const () as usize as u64;
        let syscall_exit = exit_syscall as *const () as usize as u64;
        let trap_exit = exit_trap as *const () as usize as u64;

        // Emit (once per cache) the shared inline indirect-branch lookup routine
        // and record its address so translated indirect branches can reach it.
        self.state.ib_lookup = self
            .process
            .addr_space
            .lock()
            .unwrap()
            .code
            .ensure_ib_lookup(block_exit)?;

        while self.running {
            self.deliver_pending_signals();
            self.refresh_exit_requested();

            // Observe a sibling's process-wide stop only after
            // `refresh_exit_requested`: the refresh clears this thread's
            // safepoint slot, so checking first would open a window where a
            // stop armed between the check and the clear is wiped — the flag
            // already checked, the slot no longer set — and a fully linked
            // loop runs on unwatched. Checked in this order, a stop armed
            // before the refresh is caught by these flags, and one armed
            // after it leaves the slot set for the in-cache polls.
            //
            // Another thread may have issued `exit_group`, which ends the
            // whole thread group: stop with the process-wide code so the main
            // thread returns it from the run and the process exits.
            if self.process.is_exiting() {
                self.exit_code = self.process.exit_code.load(Ordering::Relaxed);
                break;
            }
            // A sibling committed an `execve`: the thread group is dissolving,
            // the way Linux's `de_thread` kills every other thread before a
            // new image is installed. Stop at this boundary — a worker's stop
            // ends its host thread; the main thread hands the run over to the
            // exec driver below.
            if self.process.exec_pending() {
                break;
            }

            let ts_ptr: *mut ThreadState = &mut *self.state;

            let rip = unsafe { (*ts_ptr).rip };
            // Hold the address-space lock only long enough to resolve (and, on a
            // miss, translate) the next block; the guard drops at the end of this
            // statement, before `dispatch()` runs the block.
            let host_pc = match self.process.addr_space.lock().unwrap().resolve(
                rip,
                block_exit,
                syscall_exit,
                trap_exit,
            ) {
                Ok(host_pc) => host_pc,
                // The guest jumped to unmapped memory — a wild indirect branch
                // through a corrupted pointer, say. Natively the fetch faults;
                // raise the same SIGSEGV: it enters the guest's handler, or
                // (default action) terminates the process faithfully.
                Err(Error::BadAccess(_)) => {
                    let restart = self.restart.take();
                    let state = &mut *self.state;
                    self.signals.deliver(state, SIGSEGV, restart);
                    continue;
                }
                Err(e) => panic!("translate failed at {:#x}: {}", rip, e),
            };
            unsafe {
                (*ts_ptr).exit_kind = EXIT_KIND_BLOCK;
            }
            self.process.addr_space.lock().unwrap().code_deny_writes();
            unsafe {
                dispatch(ts_ptr, host_pc);
            }
            self.process.addr_space.lock().unwrap().code_allow_writes();
            match unsafe { (*ts_ptr).exit_kind } {
                EXIT_KIND_SYSCALL => {
                    if let Some(reason) = self.handle_syscall() {
                        return Ok(reason);
                    }
                }
                EXIT_KIND_TRAP => {
                    // A guest `int3` exited here with `rip` already at the
                    // instruction after the breakpoint. Raise SIGTRAP: it enters
                    // the guest's handler, or (default action) terminates the
                    // process with a faithful SIGTRAP status.
                    let restart = self.restart.take();
                    let state = &mut *self.state;
                    self.signals.deliver(state, SIGTRAP, restart);
                }
                _ => {}
            }
        }
        // Absent an `exit_group`, the kernel reports the status of the last
        // thread to exit as the process's `wait(2)` status, so every thread
        // records its own on the way out — while it is still in the thread list.
        self.process.record_exit_status(self.exit_code);
        if self.is_main {
            // A committed execve dissolves the group instead of ending the
            // process: hand the run to the exec driver, which waits out the
            // last sibling and installs the published image.
            if self.process.exec_pending() {
                return Ok(ExitReason::Execve);
            }
            // The main thread's run returning ends the process. If it exited
            // on its own (`pthread_exit` / thread-local `exit` / raw
            // `SYS_exit`) rather than via a process-wide `exit_group`, POSIX
            // keeps the process alive until the last thread finishes — so wait
            // for the siblings and adopt the final status: the last exiter's
            // (this thread's own, just recorded, if no sibling outlives it),
            // or an `exit_group`'s code. A sibling may also commit an execve
            // while this thread waits; the group then dissolves into the new
            // image rather than exiting.
            if !self.process.is_exiting() {
                self.exit_code = self.process.wait_for_others(&self.state);
                if self.process.exec_pending() {
                    return Ok(ExitReason::Execve);
                }
            }
        }
        Ok(ExitReason::Exited(self.exit_code))
    }

    /// Service the syscall that just exited the cache. Returns `Some` only when
    /// the guest issued an allowed `execve`/`execveat`, in which case the run
    /// loop hands the request back to its caller to re-enter on the new image.
    fn handle_syscall(&mut self) -> Option<ExitReason> {
        let number = self.state.regs[RAX];
        let args = [
            self.state.regs[RDI],
            self.state.regs[RSI],
            self.state.regs[RDX],
            self.state.regs[R10],
            self.state.regs[R8],
            self.state.regs[R9],
        ];
        let mut call = SystemCall::new(number, args);
        // Clone the `Arc` so the handler borrow comes from the local handle
        // rather than `self`, leaving `self` free to be passed mutably to the
        // syscall driver.
        let process = self.process.clone();
        crate::syscall::syscall(self, &mut call, process.handler.as_ref());
        let result = call.return_value();
        self.state.regs[RAX] = result as u64;

        // Record a restart candidate for SA_RESTART: a forwarded slow syscall
        // interrupted by a signal returns EINTR, and the dispatcher must be able
        // to re-issue it if the delivered handler asked to restart. `state.rip`
        // is the instruction after the `syscall` (the syscall is 2 bytes wide).
        // The never-restart interfaces always surface EINTR, so they are
        // excluded here. Cleared on any other syscall outcome.
        self.restart = if result == -(libc::EINTR as i64) && !never_restart(number) {
            Some((self.state.rip, number))
        } else {
            None
        };

        if (number == libc::SYS_execve as u64 || number == libc::SYS_execveat as u64)
            && self.process.exec_pending()
        {
            // Only a committed exec leaves the run — a failed one already
            // wrote `-errno` into rax above and the guest just resumes. The
            // calling thread's own stop mirrors `de_thread`: a worker
            // dissolves (its image request is already published, and the main
            // thread picks it up at its next boundary), while the main thread
            // returns to the exec driver to install the new image itself.
            if self.is_main {
                return Some(ExitReason::Execve);
            }
            self.running = false;
        }
        None
    }

    fn setup_gs(&self) -> Result<(), Error> {
        let state_addr = &*self.state as *const ThreadState as usize;
        let ret = unsafe { libc::syscall(libc::SYS_arch_prctl, ARCH_SET_GS, state_addr) };
        if ret != 0 {
            return Err(Error::last_os_error("arch_prctl(ARCH_SET_GS)"));
        }
        Ok(())
    }
}

/// The base `struct clone_args` (`CLONE_ARGS_SIZE_VER0`): the 8 `u64` fields
/// every `clone3` must supply. The kernel rejects a smaller struct with
/// `EINVAL` and one larger than a page with `E2BIG`.
const CLONE_ARGS_SIZE_VER0: u64 = 64;
pub const CLONE_ARGS_SIZE_MAX: u64 = 4096;

/// Copy the base `clone_args` struct a `clone3` points at out of guest
/// memory, fault-safely (see [`copy_from_guest`]). `None` — returned for an
/// unreadable struct and for a guest-declared `size` outside the kernel's
/// accepted range — tells the caller to forward the call, so the kernel
/// reports the authoritative error (`EFAULT`, `EINVAL`, `E2BIG`) exactly as
/// it would natively.
pub fn read_clone3_args(args_ptr: u64, size: u64) -> Option<[u64; 8]> {
    if !(CLONE_ARGS_SIZE_VER0..=CLONE_ARGS_SIZE_MAX).contains(&size) {
        return None;
    }
    let mut raw = [0u8; CLONE_ARGS_SIZE_VER0 as usize];
    if !copy_from_guest(args_ptr, &mut raw) {
        return None;
    }
    let mut args = [0u64; 8];
    for (slot, chunk) in args.iter_mut().zip(raw.chunks_exact(8)) {
        *slot = u64::from_ne_bytes(chunk.try_into().unwrap());
    }
    Some(args)
}

/// Whether a syscall interrupted by a signal must always fail with `EINTR`,
/// never restarting even under `SA_RESTART`. These are the interfaces the kernel
/// documents as non-restartable (`signal(7)`): the signal/event waits, the
/// multiplexing calls, sleeps, and System V IPC. Any other interrupted slow
/// syscall is a restart candidate.
fn never_restart(number: u64) -> bool {
    matches!(
        number as i64,
        libc::SYS_pause
            | libc::SYS_rt_sigsuspend
            | libc::SYS_rt_sigtimedwait
            | libc::SYS_poll
            | libc::SYS_ppoll
            | libc::SYS_select
            | libc::SYS_pselect6
            | libc::SYS_epoll_wait
            | libc::SYS_epoll_pwait
            | libc::SYS_nanosleep
            | libc::SYS_clock_nanosleep
            | libc::SYS_msgrcv
            | libc::SYS_msgsnd
            | libc::SYS_semop
            | libc::SYS_semtimedop
            | libc::SYS_io_getevents
    )
}

/// Guest register file plus a few bookkeeping slots. The exact byte layout is
/// load-bearing: the offsets are consumed by `trampoline.S` (via `offset_of!`
/// in [`super::trampoline`]) and by the per-block exit stubs emitted by the
/// translator. The struct is 64-byte aligned, and the fields are arranged so
/// `fpstate` falls on a 64-byte boundary (offset 256) — XSAVE/XRSTOR `#GP`
/// on a misaligned save area.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct ThreadState {
    /// Guest GPRs: rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8..r15.
    pub regs: [u64; 16],
    /// Guest program counter; set on exit, read on entry.
    pub rip: u64,
    /// Guest rflags; set on exit, read on entry.
    pub rflags: u64,
    /// Chimera's stack pointer, saved on entry and restored on exit.
    pub chimera_rsp: u64,
    /// Host PC for the next entry, used by `dispatch` after it has already
    /// clobbered `rsi`.
    pub host_pc_target: u64,
    /// Why the last exit happened. Read by the run loop after every entry,
    /// reset to `BLOCK` before each entry.
    pub exit_kind: u64,
    /// Guest's FS base. Loaded into the FS MSR on every entry, restored
    /// from the FS MSR on every exit. Updated by `syscall` when it
    /// intercepts `arch_prctl(ARCH_SET_FS, ...)`.
    pub guest_fs_base: u64,
    /// Chimera's FS base, captured on the host thread immediately before
    /// guest execution starts. Restored on every exit so the runtime's own
    /// TLS works after the guest has changed FS.
    pub chimera_fs_base: u64,
    /// Host address of the shared inline indirect-branch lookup routine in the
    /// code cache (`CodeCache::ensure_ib_lookup`). Each translated indirect
    /// branch ends in `jmp gs:[ib_lookup]`; set once per run before the loop.
    pub ib_lookup: u64,
    /// Scratch slots used only by the inline indirect-branch lookup routine,
    /// which has no free registers of its own: the guest's flags (via
    /// `lahf`/`seto`), the branch target, the borrowed rcx/rdx, and the
    /// resolved host PC. Live only for the duration of one lookup.
    pub ib_flags: u64,
    pub ib_target: u64,
    pub ib_rcx: u64,
    pub ib_rdx: u64,
    pub ib_host: u64,
    /// Asynchronous-exit flag polled by translated code at loop-closing edges.
    /// The host signal catcher sets it; the run loop recomputes it each iteration
    /// as "a deliverable signal is pending" so it self-clears. When set, a fully
    /// linked, syscall-free guest loop is dragged back to the run loop within one
    /// iteration, where a pending signal is delivered at a real block boundary.
    /// Reached from translated code as `gs:[]`.
    ///
    /// `AtomicU32` because the signal catcher writes it asynchronously (from the
    /// handler, via `gs:[]`) while the run loop reads and writes it; plain accesses
    /// would be a data race the compiler could miscompile (e.g. drop the clear in
    /// [`Thread::refresh_exit_requested`] as a dead store). The in-memory layout is
    /// an identical 32-bit word at the same offset, so the `gs:[]` poll is
    /// unaffected.
    pub exit_requested: AtomicU32,
    /// Whether the physical FP/SIMD registers currently hold this thread's
    /// guest state. `dispatch` clears it on every cache entry (the Rust code
    /// that ran since the last exit has clobbered the vector registers); the
    /// first translated block that actually touches FP restores `fpstate` and
    /// sets it (see the translator's block prologue). The exit trampolines save
    /// `fpstate` only when it is set, so a residency that never touches FP pays
    /// neither XRSTOR nor XSAVE. Read and written by both `trampoline.S` and the
    /// emitted prologue, so it sits in the gs-reachable region.
    /// 32-bit so it packs against `exit_requested` and `fpstate` stays at
    /// offset 256; only 0 and 1 are ever stored.
    pub fp_in_regs: u32,
    /// Scratch slot the block prologue uses to park the guest's status flags
    /// (via `lahf`/`seto`) across its checks, for the blocks whose first
    /// instructions read flags set by a predecessor. Live only for the few
    /// instructions of one prologue.
    pub fp_flags: u64,
    /// Scratch slot the block prologue uses to park the guest's rdx across the
    /// `xrstor64` (whose `edx` mask half clobbers it). Live only for the
    /// duration of one restore.
    pub fp_scratch: u64,
    /// XSAVE area for the guest's extended FP/SIMD state (x87, SSE, AVX,
    /// AVX-512). The canonical copy whenever no translated code is running;
    /// saved and restored only around blocks that touch FP (see `fp_in_regs`).
    /// Must be 64-byte aligned; the field layout above guarantees offset 256.
    pub fpstate: [u8; XSAVE_AREA_SIZE],
    /// Whether the FS base register currently holds the guest's base (rather
    /// than Chimera's, used by Rust TLS). Mirrors `fp_in_regs` for the FS-base
    /// swap: `dispatch` clears it and leaves FS holding Chimera's base, the
    /// first block that reads guest TLS (`fs:`) installs `guest_fs_base` and
    /// sets it, and the exit trampolines restore Chimera's base only when it is
    /// set. A residency that never touches `fs:` keeps both `wrfsbase`s off the
    /// path. Placed after `fpstate` so that field stays 64-byte aligned.
    pub fs_is_guest: u64,
    /// Address of this thread's `PendingSet` (owned by its `Signals`), read by
    /// the host signal catcher via `gs:[]` so a caught signal is recorded on
    /// the thread that caught it — pending state is per-thread, and this
    /// pointer is how the async-signal-safe catcher finds the right set with
    /// no TLS. Placed after `fpstate` so every offset above is unchanged.
    pub pending_set: u64,
    /// The kernel TID of the host thread backing this guest thread. Written by
    /// the owning thread at run entry — and rewritten in a fork child, where
    /// the copied value names a thread that no longer exists — and read by
    /// siblings through the thread list as the target for the reserved interrupt
    /// signal. Atomic because those reads are cross-thread; placed after
    /// `fpstate` so every `gs:[]` offset above is unchanged.
    pub tid: AtomicI32,
}

// XSAVE/XRSTOR #GP unless the save area is 64-byte aligned. The struct's
// align(64) handles the allocation; this guards the field offset against a
// future reordering.
const _: () = assert!(
    core::mem::offset_of!(ThreadState, fpstate) % 64 == 0,
    "ThreadState::fpstate must be 64-byte aligned for XSAVE/XRSTOR"
);

impl ThreadState {
    fn reset(&mut self, rip: u64, rsp: u64, guest_fs_base: u64) {
        self.regs = [0; 16];
        self.rip = 0;
        self.rflags = 0;
        self.chimera_rsp = 0;
        self.host_pc_target = 0;
        self.exit_kind = 0;
        self.guest_fs_base = guest_fs_base;
        self.chimera_fs_base = 0;
        self.ib_lookup = 0;
        self.ib_flags = 0;
        self.ib_target = 0;
        self.ib_rcx = 0;
        self.ib_rdx = 0;
        self.ib_host = 0;
        self.exit_requested.store(0, Ordering::Relaxed);
        self.fp_in_regs = 0;
        self.fp_flags = 0;
        self.fp_scratch = 0;
        self.fs_is_guest = 0;
        self.fpstate.fill(0);

        // XRSTOR loads MXCSR from the legacy region (bytes 24..28) of the save
        // area on every entry, regardless of XSTATE_BV. A zeroed area would load
        // MXCSR = 0, which unmasks every SSE floating-point exception and turns
        // ordinary float math into a SIGFPE. Seed it with the ABI default
        // 0x1f80 (all exceptions masked) — the value the Linux kernel gives a
        // fresh process — for the first entry. After that the guest's own MXCSR
        // round-trips through XSAVE/XRSTOR. The x87 control word and all vector
        // registers initialize correctly from the zeroed XSTATE_BV.
        self.fpstate[24..28].copy_from_slice(&0x0000_1f80u32.to_le_bytes());
        self.regs[RSP] = rsp;
        self.rip = rip;
    }

    fn capture_chimera_fs(&mut self) {
        self.chimera_fs_base = current_fs_base();
    }

    /// Build the register state for a `clone(CLONE_VM)` child. The guest-visible
    /// registers, flags, and FP/SIMD state are copied from the parent, so the
    /// child resumes exactly where the parent's `clone` returns — except `rax`,
    /// which is 0 (the child's `clone` return value), and `rsp`, which is the
    /// child's own stack. `tls` is the child's thread pointer (FS base) when the
    /// clone requested `CLONE_SETTLS`, else the child inherits the parent's. The
    /// per-run bookkeeping slots (the Chimera/FS scratch and the indirect-branch
    /// lookup fields) start cleared; `run` repopulates them on the child's own
    /// host thread.
    fn clone_for_child(&self, child_stack: u64, tls: Option<u64>) -> Box<ThreadState> {
        let mut child = Box::new(ThreadState {
            regs: self.regs,
            rip: self.rip,
            rflags: self.rflags,
            chimera_rsp: 0,
            host_pc_target: 0,
            exit_kind: 0,
            guest_fs_base: tls.unwrap_or(self.guest_fs_base),
            chimera_fs_base: 0,
            ib_lookup: 0,
            ib_flags: 0,
            ib_target: 0,
            ib_rcx: 0,
            ib_rdx: 0,
            ib_host: 0,
            exit_requested: AtomicU32::new(0),
            fp_in_regs: 0,
            fp_flags: 0,
            fp_scratch: 0,
            fpstate: self.fpstate,
            fs_is_guest: 0,
            // Cleared, not inherited: the child's own `PendingSet` address is
            // published by `Thread::from_state`. Copying the parent's pointer
            // would let the catcher record the child's signals on the parent.
            pending_set: 0,
            // Cleared for the same reason as the run-entry store: the child
            // publishes its own TID before it runs any guest code.
            tid: AtomicI32::new(0),
        });
        child.regs[RAX] = 0;
        child.regs[RSP] = child_stack;
        child
    }
}

fn current_fs_base() -> u64 {
    let fs_base: u64;
    unsafe {
        asm!(
            "rdfsbase {0}",
            out(reg) fs_base,
            options(nomem, nostack, preserves_flags),
        );
    }
    fs_base
}

pub const RAX: usize = 0;
pub const RDX: usize = 3;
pub const RSI: usize = 4;
pub const RDI: usize = 5;
pub const RSP: usize = 7;
pub const R8: usize = 8;
pub const R9: usize = 9;
pub const R10: usize = 10;
