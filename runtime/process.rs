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
    Condvar, Mutex, MutexGuard, Once,
    atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering},
};

use crate::{
    Error, SystemCalls,
    sys::{linux::exec::PreparedExec, mmap::AddressSpace},
};

/// Host signal Chimera reserves to interrupt a guest thread parked in a
/// forwarded syscall (so it returns `EINTR` and re-checks the exit flag). The
/// highest real-time signal is used as the one least likely to collide with a
/// signal the guest itself installs. A do-nothing handler is installed for it,
/// without `SA_RESTART`, purely so the kernel interrupts the blocking syscall.
fn interrupt_signal() -> libc::c_int {
    libc::SIGRTMAX()
}

extern "C" fn interrupt_noop(_sig: libc::c_int) {}

static INSTALL_INTERRUPT: Once = Once::new();

/// Install the no-op handler for [`interrupt_signal`] once per process.
fn install_interrupt_handler() {
    INSTALL_INTERRUPT.call_once(|| unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = interrupt_noop as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0; // no SA_RESTART: a parked syscall must surface EINTR
        libc::sigaction(interrupt_signal(), &sa, std::ptr::null_mut());
    });
}

/// A thread in the live set: its kernel TID — the target for interrupt
/// signals, nothing more — and the address of its `ThreadState::exit_requested`
/// safepoint slot, which doubles as the thread's identity: it is stable where
/// the TID is not (a fork child keeps its registration but wakes up with a new
/// TID). The pointer is valid for exactly the registration window —
/// `Thread::run` registers after its state is pinned and unregisters, under
/// the same mutex, before the state can be dropped — so a writer holding the
/// `live_threads` lock never touches a dead slot, and identity comparisons
/// never dereference at all.
struct LiveThread {
    tid: i32,
    exit_requested: *const AtomicU32,
}

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
    /// The threads currently running a guest. Each thread adds itself when its
    /// run loop starts and removes itself when it ends, so a process-wide stop
    /// (`exit_group`, a committed `execve`) can reach every sibling — armed
    /// safepoint slot for one executing translated code, interrupt signal for
    /// one parked in a host syscall — and the main thread can wait for the
    /// others to finish.
    live_threads: Mutex<Vec<LiveThread>>,
    /// Signalled whenever `live_threads` shrinks or a process-wide exit is
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
}

impl Process {
    pub fn new(handler: Box<dyn SystemCalls>, code_cache_size: usize) -> Result<Self, Error> {
        install_interrupt_handler();
        Ok(Self {
            addr_space: Mutex::new(AddressSpace::new(code_cache_size)?),
            handler,
            exiting: AtomicBool::new(false),
            exit_code: AtomicI32::new(0),
            live_threads: Mutex::new(Vec::new()),
            exit_cv: Condvar::new(),
            last_exit_status: AtomicI32::new(0),
            execing: AtomicBool::new(false),
            exec_request: Mutex::new(None),
        })
    }

    /// Register a thread as running a guest: its kernel TID and the address of
    /// its `ThreadState::exit_requested` safepoint slot, which the translated
    /// back-edge and IB-hit polls read and which serves as the thread's
    /// identity (see [`LiveThread`]). Balanced by
    /// [`Process::unregister_thread`] when its run loop ends.
    pub fn register_thread(&self, tid: i32, exit_requested: &AtomicU32) {
        self.live_threads.lock().unwrap().push(LiveThread {
            tid,
            exit_requested: exit_requested as *const AtomicU32,
        });
    }

    /// Remove a thread (identified by its safepoint slot) from the live set
    /// when its run loop ends, and wake any main thread waiting for the last
    /// sibling to finish. The pointer is compared, never dereferenced.
    pub fn unregister_thread(&self, slot: *const AtomicU32) {
        let mut threads = self.live_threads.lock().unwrap();
        if let Some(pos) = threads.iter().position(|t| t.exit_requested == slot) {
            threads.swap_remove(pos);
        }
        self.exit_cv.notify_all();
    }

    /// Record `status` as the calling thread's guest exit status. Every run
    /// loop calls this on the way out, before it leaves the live set, so once
    /// the group drains the slot holds the last exiter's status — the value
    /// `wait(2)` reports absent an `exit_group`. The mutex inside
    /// `unregister_thread`, taken after this store, orders it before any waiter
    /// that observes the shrunken live set.
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
    pub fn wait_for_others(&self, self_slot: &AtomicU32) -> i32 {
        let self_slot = self_slot as *const AtomicU32;
        let mut threads = self.live_threads.lock().unwrap();
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
            if threads.iter().all(|t| t.exit_requested == self_slot) {
                return self.last_exit_status.load(Ordering::Relaxed);
            }
            threads = self.exit_cv.wait(threads).unwrap();
        }
    }

    /// Record a process-wide exit (`exit_group`) requested by the thread whose
    /// safepoint slot is `self_slot`. The code is published before the flag, with
    /// release/acquire ordering, so any thread that later sees `is_exiting()`
    /// reads this exact code. Every other live thread is then interrupted with
    /// [`interrupt_signal`] so one parked in a host syscall returns `EINTR` and
    /// observes the exit at its run-loop boundary, rather than waiting for a
    /// syscall that may never return on its own.
    pub fn request_exit_group(&self, code: i32, self_slot: &AtomicU32) {
        self.exit_code.store(code, Ordering::Relaxed);
        self.exiting.store(true, Ordering::Release);
        self.interrupt_others(self_slot);
    }

    /// Publish a committed `execve`: the calling thread validated and parsed
    /// the replacement image (a failed one never gets here — the caller took
    /// `-errno` and no sibling was disturbed), so the thread group now
    /// dissolves, the way Linux's `de_thread` kills every sibling before a new
    /// image is installed. Every run loop observes the pending exec at its
    /// next boundary and stops; the main host thread then waits out the
    /// stragglers and installs the image (see `crate::sys::linux::exec`).
    ///
    /// First committer wins: returns false — publishing nothing — when the
    /// group is already dissolving, under an earlier committed exec or an
    /// `exit_group`. Concurrent execs race natively too, but only one
    /// survives; the loser is killed by the winner's `de_thread` and never
    /// observes a return value. A refused caller needs to do nothing: the
    /// stop already pending takes its thread at the next boundary, exactly
    /// like any other sibling.
    pub fn request_exec(&self, prepared: PreparedExec, self_slot: &AtomicU32) -> bool {
        let mut request = self.exec_request.lock().unwrap();
        if request.is_some() || self.exec_pending() || self.is_exiting() {
            return false;
        }
        *request = Some(prepared);
        self.execing.store(true, Ordering::Release);
        drop(request);
        self.interrupt_others(self_slot);
        true
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

    /// Block until every guest thread has left the live set, so tearing down
    /// the old image cannot pull mappings out from under a straggler still
    /// translating or executing old blocks. The caller runs outside any run
    /// loop (the main host thread, after its own run returned), so "empty"
    /// means the whole group is gone.
    pub fn wait_exec_quiesce(&self) {
        let mut threads = self.live_threads.lock().unwrap();
        while !threads.is_empty() {
            threads = self.exit_cv.wait(threads).unwrap();
        }
    }

    /// Force every live thread except the caller to its run-loop boundary,
    /// where the pending process-wide stop (an `exit_group`, a committed
    /// `execve`) is observed. Two mechanisms, one per way a sibling can be
    /// away from its boundary: arming the thread's `exit_requested` safepoint
    /// slot pops one executing fully linked translated code out of the cache
    /// (the back-edge and IB-hit polls read it), and [`interrupt_signal`]
    /// makes one parked in a blocking host syscall return `EINTR`. Also wakes
    /// a main thread parked in [`Process::wait_for_others`].
    fn interrupt_others(&self, self_slot: &AtomicU32) {
        let self_slot = self_slot as *const AtomicU32;
        let pid = unsafe { libc::getpid() };
        let sig = interrupt_signal();
        let threads = self.live_threads.lock().unwrap();
        for t in threads.iter() {
            if t.exit_requested != self_slot {
                // SAFETY: registration (see `LiveThread`) guarantees the slot
                // outlives its entry in the set, and holding the lock keeps
                // the entry alive across this store.
                unsafe { (*t.exit_requested).store(1, Ordering::Release) };
                unsafe { libc::syscall(libc::SYS_tgkill, pid, t.tid, sig) };
            }
        }
        self.exit_cv.notify_all();
    }

    /// Whether a thread has requested a process-wide exit; paired with
    /// [`Process::exit_code`] (read after a true result).
    pub fn is_exiting(&self) -> bool {
        self.exiting.load(Ordering::Acquire)
    }

    /// Take every `Process` lock, to be held across a forwarded `fork`. fork
    /// copies the whole address space, mutexes included: one held by a
    /// sibling thread at the moment of the copy would be locked forever in
    /// the child, which has no sibling to release it. Held by the forking
    /// thread instead, the child's copies belong to that thread's own copied
    /// guards, which unlock on drop in both processes — the `pthread_atfork`
    /// discipline, applied at the one place Chimera forwards a fork. The
    /// lock order nests like `request_exec` (`exec_request`, then
    /// `live_threads`); `addr_space` is never nested with either.
    pub fn lock_for_fork(&self) -> ForkLocks<'_> {
        ForkLocks {
            _exec_request: self.exec_request.lock().unwrap(),
            _live_threads: self.live_threads.lock().unwrap(),
            _addr_space: self.addr_space.lock().unwrap(),
        }
    }

    /// Rebuild the process bookkeeping in the child of a fork. The copy
    /// still describes the parent's whole thread group, but the child has
    /// exactly one thread — the caller. Keep only the caller's live entry,
    /// found by its safepoint slot address (the identity that survives a
    /// fork), and rewrite its TID to `my_tid`, the caller's tid in the child
    /// — a signal aimed at the parent-era TID would miss, leaving the thread
    /// unreachable when parked in a host syscall. Then clear any group-wide
    /// exit or exec the parent had in flight: the child is a fresh process,
    /// not a participant in the parent's stop.
    pub fn reset_after_fork(&self, my_slot: &AtomicU32, my_tid: i32) {
        let my_slot = my_slot as *const AtomicU32;
        let mut threads = self.live_threads.lock().unwrap();
        threads.retain(|t| t.exit_requested == my_slot);
        for t in threads.iter_mut() {
            t.tid = my_tid;
        }
        drop(threads);
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
    _live_threads: MutexGuard<'a, Vec<LiveThread>>,
    _addr_space: MutexGuard<'a, AddressSpace>,
}

// SAFETY: the `handler` field is already `Send + Sync` (the `SystemCalls`
// supertrait requires it). Two fields are not auto-`Send`/`Sync`, both for
// raw pointers. `addr_space`'s code cache holds `*mut u8` into the
// translated-code and indirect-branch mappings — ordinary process-wide `mmap`
// regions with no thread affinity, and the whole `AddressSpace` sits behind a
// `Mutex`, so all access is serialized. `live_threads` holds each registered
// thread's `*const AtomicU32` safepoint slot — the target is an atomic (safe
// to store to from any thread), and the registration discipline documented on
// `LiveThread` keeps every pointer in the set valid while the `live_threads`
// mutex is held. Sharing one `Process` across the host threads that back the
// guest threads is therefore sound. This is what makes `Arc<Process>` (rather
// than a single-threaded `Rc`) the right handle: the address space is built
// to be shared by `clone(CLONE_VM)` siblings.
unsafe impl Send for Process {}
unsafe impl Sync for Process {}
