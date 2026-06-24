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
    Mutex, Once,
    atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering},
};

use crate::{Error, SystemCalls, sys::mmap::AddressSpace};

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

/// A thread in the live set: its kernel TID and the address of its
/// `ThreadState::exit_requested` safepoint slot. The pointer is valid for
/// exactly the registration window — `Thread::run` registers after its state
/// is pinned and unregisters, under the same mutex, before the state can be
/// dropped — so a writer holding the `live_threads` lock never touches a dead
/// slot.
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
    /// run loop starts and removes itself when it ends, so `exit_group` can
    /// reach every sibling — armed safepoint slot for one executing translated
    /// code, interrupt signal for one parked in a host syscall.
    live_threads: Mutex<Vec<LiveThread>>,
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
        })
    }

    /// Register a thread as running a guest: its kernel TID and the address of
    /// its `ThreadState::exit_requested` safepoint slot, which the translated
    /// back-edge and IB-hit polls read. Balanced by
    /// [`Process::unregister_tid`] when its run loop ends.
    pub fn register_thread(&self, tid: i32, exit_requested: &AtomicU32) {
        self.live_threads.lock().unwrap().push(LiveThread {
            tid,
            exit_requested: exit_requested as *const AtomicU32,
        });
    }

    /// Remove a thread from the live set when its run loop ends.
    pub fn unregister_tid(&self, tid: i32) {
        let mut threads = self.live_threads.lock().unwrap();
        if let Some(pos) = threads.iter().position(|t| t.tid == tid) {
            threads.swap_remove(pos);
        }
    }

    /// Record a process-wide exit (`exit_group`) requested by the thread whose
    /// kernel TID is `self_tid`. The code is published before the flag, with
    /// release/acquire ordering, so any thread that later sees `is_exiting()`
    /// reads this exact code. Every other live thread is then interrupted with
    /// [`interrupt_signal`] so one parked in a host syscall returns `EINTR` and
    /// observes the exit at its run-loop boundary, rather than waiting for a
    /// syscall that may never return on its own.
    pub fn request_exit_group(&self, code: i32, self_tid: i32) {
        self.exit_code.store(code, Ordering::Relaxed);
        self.exiting.store(true, Ordering::Release);
        let pid = unsafe { libc::getpid() };
        let sig = interrupt_signal();
        let threads = self.live_threads.lock().unwrap();
        for t in threads.iter() {
            if t.tid != self_tid {
                // SAFETY: registration (see `LiveThread`) guarantees the slot
                // outlives its entry in the set, and holding the lock keeps
                // the entry alive across this store.
                unsafe { (*t.exit_requested).store(1, Ordering::Release) };
                unsafe { libc::syscall(libc::SYS_tgkill, pid, t.tid, sig) };
            }
        }
    }

    /// Whether a thread has requested a process-wide exit; paired with
    /// [`Process::exit_code`] (read after a true result).
    pub fn is_exiting(&self) -> bool {
        self.exiting.load(Ordering::Acquire)
    }
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
