//! Process-wide guest state shared by every thread.
//!
//! A guest process is one [`Process`] and one or more [`crate::arch::dispatch`]
//! `Thread`s. The `Thread` holds the per-thread register file; the `Process`
//! holds everything the threads share — today the guest address space and its
//! translated-block cache, the analogue of the kernel's `mm_struct`. Each
//! `Thread` keeps an `Arc<Process>`, so a future `clone(CLONE_VM)` hands the
//! child the same `Arc` and the two threads translate into and map within one
//! address space.

use std::sync::{
    Condvar, Mutex, MutexGuard,
    atomic::{AtomicBool, AtomicI32, Ordering},
};

use crate::{
    Error, SystemCalls,
    arch::dispatch::ThreadState,
    sys::{exec::PreparedExec, mmap::AddressSpace, signal},
};

/// The shared state of a guest process: one per process, referenced by every
/// thread through an `Arc`.
pub struct Process {
    /// The guest address space and its translated-block cache. Guarded by a
    /// mutex because every thread translates into and maps within the one
    /// space, so cache insertion and region bookkeeping must be serialized. The
    /// lock is held only across a translation, a lookup, or a region update —
    /// never across `dispatch()` or a blocking host syscall, so a thread
    /// running guest code or parked in the kernel never holds it.
    pub addr_space: Mutex<AddressSpace>,
    /// The embedder's system-call handler, shared by every thread of the
    /// process. `SystemCalls` is `Send + Sync` and dispatched by `&self`
    /// (see [`crate::SystemCalls`]), so all threads — including the host threads
    /// that will back `clone(CLONE_VM)` siblings — drive the one handler
    /// instance concurrently, reaching it through this shared `Process`.
    pub handler: Box<dyn SystemCalls>,
    /// The guest's signal-disposition table, shared by every thread of the
    /// process. POSIX keeps dispositions process-wide, so `clone(CLONE_VM)`
    /// siblings hand their per-thread [`signal::Signals`] a clone of this `Arc`
    /// rather than a fresh table — otherwise a thread-directed signal would be
    /// delivered against a stale default on a thread that did not install the
    /// handler. See [`signal::SharedSigTable`].
    pub sig_table: signal::SharedSigTable,
    /// Set when any thread issues `exit_group`: that call terminates the whole
    /// thread group, not just the caller. Every thread's run loop observes this
    /// at its next iteration (a block/syscall boundary) and stops, so the main
    /// thread returns `exit_code` from the run and the process ends. A plain
    /// `exit` (thread-local) does not set it. See [`Process::request_exit_group`].
    pub exiting: AtomicBool,
    /// The status the process exits with once `exiting` is set. Written before
    /// `exiting` is published, so a thread that observes `exiting` reads the
    /// final code.
    pub exit_code: AtomicI32,
    /// The threads currently running a guest, each entry the address of the
    /// thread's own [`ThreadState`] — the kernel's shape: the group list holds
    /// the task itself, not a parallel record. The address is the thread's
    /// identity (stable across a fork, where the TID is not), and the only
    /// fields reached through it are the atomic `tid` and `exit_requested`,
    /// so a process-wide stop (`exit_group`, a committed `execve`) can reach
    /// every sibling — armed safepoint slot for one executing translated
    /// code, interrupt signal for one parked in a host syscall — and the main
    /// thread can wait for the others to finish. A pointer is valid for
    /// exactly the registration window: `Thread::run` registers after its
    /// state is pinned and unregisters, under the same mutex, before the
    /// state can drop, so a reader holding the `threads` lock never
    /// touches a dead state.
    threads: Mutex<Vec<*const ThreadState>>,
    /// Signalled whenever `threads` shrinks or a process-wide exit is
    /// requested, so a main thread parked in [`Process::wait_for_others`] wakes.
    exit_cv: Condvar,
    /// The guest exit status of the most recent thread to finish its run.
    /// Absent an `exit_group`, the kernel reports the status of the *last*
    /// thread to exit as the process's `wait(2)` status — whichever thread that
    /// is — so every run loop records its status here on the way out and a main
    /// thread that outwaits its siblings adopts the final value.
    last_exit_status: AtomicI32,
    /// Set when a thread commits an `execve`: the parsed replacement image is
    /// waiting in `exec_request` and the thread group is dissolving. Every run
    /// loop observes this at its next boundary and stops, mirroring Linux's
    /// `de_thread`, which kills every other thread before installing a new
    /// image. See [`Process::request_exec`].
    execing: AtomicBool,
    /// The committed replacement image, published by whichever thread's
    /// `execve` validated it and consumed by the exec driver on the main host
    /// thread once the group has drained. Written before `execing` is set.
    exec_request: Mutex<Option<PreparedExec>>,
    /// Exit handlers the guest registered with `atexit`/`__cxa_atexit`, in
    /// registration order (they run in reverse). Held here rather than in the
    /// C library the runtime shares with the guest, which would call these
    /// guest pointers natively — see `Thread::escape`. Process-wide, like the
    /// library's own list.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    atexit: Mutex<Vec<(u64, u64)>>,
}

thread_local! {
    /// How many address-space guards this thread currently holds. The fault
    /// handler reads it to decide whether taking the lock could deadlock
    /// against the faulting thread itself — see [`addr_space_held`].
    static ADDR_SPACE_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Whether the calling thread already holds the address-space lock. A fault
/// handler that wants the lock must not block when this is true — the holder
/// is the faulting thread — and must block when it is false, because the
/// holder is a sibling that will release it. Guessing from the faulting pc
/// instead gets this wrong in both directions.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn addr_space_held() -> bool {
    ADDR_SPACE_DEPTH.with(|d| d.get() != 0)
}

/// The address-space lock, held with the depth counter maintained. Derefs to
/// the [`AddressSpace`] like the raw guard it wraps.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub struct AddrSpaceGuard<'a> {
    inner: MutexGuard<'a, AddressSpace>,
}

impl std::ops::Deref for AddrSpaceGuard<'_> {
    type Target = AddressSpace;

    fn deref(&self) -> &AddressSpace {
        &self.inner
    }
}

impl std::ops::DerefMut for AddrSpaceGuard<'_> {
    fn deref_mut(&mut self) -> &mut AddressSpace {
        &mut self.inner
    }
}

impl Drop for AddrSpaceGuard<'_> {
    fn drop(&mut self) {
        ADDR_SPACE_DEPTH.with(|d| d.set(d.get() - 1));
    }
}

impl Process {
    pub fn new(handler: Box<dyn SystemCalls>, code_cache_size: usize) -> Result<Self, Error> {
        crate::sys::thread::install_interrupt_handler();
        // Own the host SIGSEGV/SIGBUS slot so self-modifying-code write traps are
        // caught synchronously (see [`crate::sys::fault`]).
        crate::sys::fault::install();
        Ok(Self {
            addr_space: Mutex::new(AddressSpace::new(code_cache_size)?),
            handler,
            sig_table: signal::new_shared_table(),
            exiting: AtomicBool::new(false),
            exit_code: AtomicI32::new(0),
            threads: Mutex::new(Vec::new()),
            exit_cv: Condvar::new(),
            last_exit_status: AtomicI32::new(0),
            execing: AtomicBool::new(false),
            exec_request: Mutex::new(None),
            atexit: Mutex::new(Vec::new()),
        })
    }

    /// Lock the guest address space, recording that this thread holds it so
    /// a fault taken while holding can tell itself apart from a fault racing
    /// a sibling (see [`addr_space_held`]).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn lock_addr_space(&self) -> AddrSpaceGuard<'_> {
        let inner = self.addr_space.lock().unwrap();
        ADDR_SPACE_DEPTH.with(|d| d.set(d.get() + 1));
        AddrSpaceGuard { inner }
    }

    /// Register a thread as running a guest, by the address of its
    /// [`ThreadState`] (see [`Process::threads`] for the discipline that
    /// keeps the pointer valid). The state carries everything a sibling's
    /// stop needs: the `exit_requested` safepoint slot the translated
    /// back-edge and IB-hit polls read, and the `tid` the interrupt signal
    /// targets. Balanced by [`Process::unregister_thread`] when the run loop
    /// ends.
    pub fn register_thread(&self, state: &ThreadState) {
        self.threads
            .lock()
            .unwrap()
            .push(state as *const ThreadState);
    }

    /// Remove a thread (identified by its `ThreadState` address) from the
    /// thread list when its run loop ends, and wake any main thread waiting for
    /// the last sibling to finish. The pointer is compared, never
    /// dereferenced.
    pub fn unregister_thread(&self, state: *const ThreadState) {
        let mut threads = self.threads.lock().unwrap();
        if let Some(pos) = threads.iter().position(|&t| t == state) {
            threads.swap_remove(pos);
        }
        self.exit_cv.notify_all();
    }

    /// The guest PC each registered thread is currently at, for the sampling
    /// profiler (see [`crate::trace::profile`]). Every exit stub stores the
    /// next guest PC into its thread's `pc` before branching — including the
    /// linked edges that never return to the run loop — so this names where a
    /// guest is spending its time even inside a warm chain.
    ///
    /// The reads race with the threads writing those slots, deliberately: a
    /// profiler wants a sample, not a barrier, and a `u64` slot yields one
    /// value or the other. Nothing is dereferenced.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn sample_guest_pcs(&self) -> Vec<u64> {
        let threads = self.threads.lock().unwrap();
        threads
            .iter()
            .map(|&t| unsafe { (*t).guest_pc() })
            .collect()
    }

    /// Record `status` as the calling thread's guest exit status. Every run
    /// loop calls this on the way out, before it leaves the thread list, so once
    /// the group drains the slot holds the last exiter's status — the value
    /// `wait(2)` reports absent an `exit_group`. The mutex inside
    /// `unregister_thread`, taken after this store, orders it before any waiter
    /// that observes the shrunken thread list.
    pub fn record_exit_status(&self, status: i32) {
        self.last_exit_status.store(status, Ordering::Relaxed);
    }

    /// Block until every guest thread other than the caller (identified by its
    /// safepoint slot) has finished, or a
    /// process-wide `exit_group` is requested; returns the resulting process
    /// status. Used when the main thread exits on its own (`pthread_exit` /
    /// thread-local `exit` / raw `SYS_exit`): POSIX keeps the process alive
    /// until the last thread terminates, and the status is that last thread's —
    /// the caller's own, recorded before waiting, if no sibling outlives it —
    /// unless an `exit_group` set the whole group's.
    pub fn wait_for_others(&self, self_state: &ThreadState) -> i32 {
        let self_state = self_state as *const ThreadState;
        let mut threads = self.threads.lock().unwrap();
        loop {
            if self.is_exiting() {
                return self.exit_code.load(Ordering::Relaxed);
            }
            // A sibling committed an execve while this thread waited: the
            // returned status is moot — the caller re-checks `exec_pending`
            // and hands the run over to the exec driver instead of exiting.
            if self.exec_pending() {
                return self.last_exit_status.load(Ordering::Relaxed);
            }
            if threads.iter().all(|&t| t == self_state) {
                return self.last_exit_status.load(Ordering::Relaxed);
            }
            threads = self.exit_cv.wait(threads).unwrap();
        }
    }

    /// Record a process-wide exit (`exit_group`) requested by the thread whose
    /// safepoint slot is `self_slot`. The code is published before the flag, with
    /// release/acquire ordering, so any thread that later sees `is_exiting()`
    /// reads this exact code. Every other thread is then interrupted with the
    /// reserved signal (see [`crate::sys::thread::interrupt`]) so one parked in
    /// a host syscall returns `EINTR` and observes the exit at its run-loop
    /// boundary, rather than waiting for a syscall that may never return.
    pub fn request_exit_group(&self, code: i32, self_state: &ThreadState) {
        self.exit_code.store(code, Ordering::Relaxed);
        self.exiting.store(true, Ordering::Release);
        self.interrupt_others(self_state);
    }

    /// Publish a committed `execve`: the calling thread validated and parsed
    /// the replacement image (a failed one never gets here — the caller took
    /// `-errno` and no sibling was disturbed), so the thread group now
    /// dissolves, the way Linux's `de_thread` kills every sibling before a new
    /// image is installed. Every run loop observes the pending exec at its
    /// next boundary and stops; the main host thread then waits out the
    /// stragglers and installs the image (see `crate::sys::exec`).
    ///
    /// First committer wins: hands the request back unpublished when the
    /// group is already dissolving, under an earlier committed exec or an
    /// `exit_group`. Concurrent execs race natively too, but only one
    /// survives; the loser is killed by the winner's `de_thread` and never
    /// observes a return value. A refused caller only disposes of its
    /// prepared image: the stop already pending takes its thread at the next
    /// boundary, exactly like any other sibling.
    pub fn request_exec(
        &self,
        prepared: PreparedExec,
        self_state: &ThreadState,
    ) -> Result<(), PreparedExec> {
        let mut request = self.exec_request.lock().unwrap();
        if request.is_some() || self.exec_pending() || self.is_exiting() {
            return Err(prepared);
        }
        *request = Some(prepared);
        self.execing.store(true, Ordering::Release);
        drop(request);
        self.interrupt_others(self_state);
        Ok(())
    }

    /// Whether a committed `execve` is waiting to be installed; every run loop
    /// treats it as a stop request, like [`Process::is_exiting`].
    pub fn exec_pending(&self) -> bool {
        self.execing.load(Ordering::Acquire)
    }

    /// Take the pending replacement image for installation and clear the
    /// request, so the new image starts with no exec state. Called by the exec
    /// driver once the group has drained.
    pub fn take_exec_request(&self) -> Option<PreparedExec> {
        let prepared = self.exec_request.lock().unwrap().take();
        self.execing.store(false, Ordering::Release);
        prepared
    }

    /// Block until every guest thread has left the thread list, so tearing down
    /// the old image cannot pull mappings out from under a straggler still
    /// translating or executing old blocks. The caller runs outside any run
    /// loop (the main host thread, after its own run returned), so "empty"
    /// means the whole group is gone.
    pub fn wait_exec_quiesce(&self) {
        let mut threads = self.threads.lock().unwrap();
        while !threads.is_empty() {
            threads = self.exit_cv.wait(threads).unwrap();
        }
    }

    /// Force every thread except the caller to its run-loop boundary,
    /// where the pending process-wide stop (an `exit_group`, a committed
    /// `execve`) is observed. Two mechanisms, one per way a sibling can be
    /// away from its boundary: arming the thread's `exit_requested` safepoint
    /// slot pops one executing fully linked translated code out of the cache
    /// (the back-edge and IB-hit polls read it), and the reserved interrupt
    /// signal (see [`crate::sys::thread::interrupt`]) makes one parked in a
    /// blocking host syscall return `EINTR`. Also wakes a main thread parked in
    /// [`Process::wait_for_others`].
    fn interrupt_others(&self, self_state: &ThreadState) {
        let self_state = self_state as *const ThreadState;
        let threads = self.threads.lock().unwrap();
        for &t in threads.iter() {
            if t != self_state {
                // SAFETY: registration (see [`Process::threads`])
                // guarantees the state outlives its entry in the set, and
                // holding the lock keeps the entry alive; only the atomic
                // `exit_requested` and `tid` fields are touched.
                let tid = unsafe {
                    (*t).exit_requested.store(1, Ordering::Release);
                    (*t).tid.load(Ordering::Acquire)
                };
                crate::sys::thread::interrupt(tid);
            }
        }
        self.exit_cv.notify_all();
    }

    /// Whether a thread has requested a process-wide exit; paired with
    /// [`Process::exit_code`] (read after a true result).
    pub fn is_exiting(&self) -> bool {
        self.exiting.load(Ordering::Acquire)
    }

    /// The status a process-wide `exit_group` published. Read only after
    /// [`Process::is_exiting`] returns true, whose `Acquire` orders this load
    /// after the requester's `Release` store.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn group_exit_code(&self) -> i32 {
        self.exit_code.load(Ordering::Relaxed)
    }

    /// Record a guest `atexit`/`__cxa_atexit` registration. Process-wide,
    /// like the C library's own list: any thread's `exit` runs all of them.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn push_atexit(&self, func: u64, arg: u64) {
        self.atexit.lock().unwrap().push((func, arg));
    }

    /// Take the most recently registered exit handler, if any. Handlers run
    /// in reverse order of registration, and taking them one at a time lets
    /// a handler register another (which then runs first).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn pop_atexit(&self) -> Option<(u64, u64)> {
        self.atexit.lock().unwrap().pop()
    }

    /// Discard every registered exit handler: an `execve` replaces the image
    /// that owns them, and POSIX says the new image starts with none.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn clear_atexit(&self) {
        self.atexit.lock().unwrap().clear();
    }

    /// Take every `Process` lock, to be held across a forwarded `fork`. fork
    /// copies the whole address space, mutexes included: one held by a
    /// sibling thread at the moment of the copy would be locked forever in
    /// the child, which has no sibling to release it. Held by the forking
    /// thread instead, the child's copies belong to that thread's own copied
    /// guards, which unlock on drop in both processes — the `pthread_atfork`
    /// discipline, applied at the one place Chimera forwards a fork. The
    /// lock order nests like `request_exec` (`exec_request`, then
    /// `threads`); `addr_space` is never nested with either.
    pub fn lock_for_fork(&self) -> ForkLocks<'_> {
        ForkLocks {
            _exec_request: self.exec_request.lock().unwrap(),
            _threads: self.threads.lock().unwrap(),
            _addr_space: self.addr_space.lock().unwrap(),
        }
    }

    /// Rebuild the process bookkeeping in the child of a fork. The copy
    /// still describes the parent's whole thread group, but the child has
    /// exactly one thread — the caller. Keep only the caller's entry,
    /// found by its `ThreadState` address (the identity that survives a fork;
    /// the caller has already rewritten the state's TID to its own in the
    /// child — a signal aimed at the parent-era TID would miss, leaving the
    /// thread unreachable when parked in a host syscall). Then clear any
    /// group-wide exit or exec the parent had in flight: the child is a fresh
    /// process, not a participant in the parent's stop.
    pub fn reset_after_fork(&self, self_state: &ThreadState) {
        let self_state = self_state as *const ThreadState;
        self.threads.lock().unwrap().retain(|&t| t == self_state);
        *self.exec_request.lock().unwrap() = None;
        self.execing.store(false, Ordering::Release);
        self.exiting.store(false, Ordering::Release);
        self.exit_code.store(0, Ordering::Relaxed);
        self.last_exit_status.store(0, Ordering::Relaxed);
    }
}

/// Every `Process` lock, held together across a `fork` forward; see
/// [`Process::lock_for_fork`].
pub struct ForkLocks<'a> {
    _exec_request: MutexGuard<'a, Option<PreparedExec>>,
    _threads: MutexGuard<'a, Vec<*const ThreadState>>,
    _addr_space: MutexGuard<'a, AddressSpace>,
}

// SAFETY: the `handler` field is already `Send + Sync` (the `SystemCalls`
// supertrait requires it). Two fields are not auto-`Send`/`Sync`, both for
// raw pointers. `addr_space`'s code cache holds `*mut u8` into the
// translated-code and indirect-branch mappings — ordinary process-wide `mmap`
// regions with no thread affinity, and the whole `AddressSpace` sits behind a
// `Mutex`, so all access is serialized. `threads` holds each registered
// thread's `*const ThreadState` — the only fields reached through it are the
// atomic `tid` and `exit_requested` (safe to access from any thread), and the
// registration discipline documented on the field keeps every pointer in the
// set valid while the `threads` mutex is held. Sharing one `Process` across the host threads that back the
// guest threads is therefore sound. This is what makes `Arc<Process>` (rather
// than a single-threaded `Rc`) the right handle: the address space is built
// to be shared by `clone(CLONE_VM)` siblings.
unsafe impl Send for Process {}
unsafe impl Sync for Process {}
