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
    Mutex,
    atomic::{AtomicBool, AtomicI32},
};

use crate::{Error, SystemCalls, sys::mmap::AddressSpace};

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
}

impl Process {
    pub fn new(handler: Box<dyn SystemCalls>, code_cache_size: usize) -> Result<Self, Error> {
        Ok(Self {
            addr_space: Mutex::new(AddressSpace::new(code_cache_size)?),
            handler,
            exiting: AtomicBool::new(false),
            exit_code: AtomicI32::new(0),
        })
    }

    /// Record a process-wide exit (`exit_group`). The code is published before
    /// the flag, with release/acquire ordering, so any thread that later sees
    /// `is_exiting()` reads this exact code.
    pub fn request_exit_group(&self, code: i32) {
        use std::sync::atomic::Ordering;
        self.exit_code.store(code, Ordering::Relaxed);
        self.exiting.store(true, Ordering::Release);
    }

    /// Whether a thread has requested a process-wide exit; paired with
    /// [`Process::exit_code`] (read after a true result).
    pub fn is_exiting(&self) -> bool {
        self.exiting.load(std::sync::atomic::Ordering::Acquire)
    }
}

// SAFETY: the `handler` field is already `Send + Sync` (the `SystemCalls`
// supertrait requires it). The only field that is not auto-`Send`/`Sync` is
// `addr_space`, because its code cache holds raw pointers (`*mut u8`) into the
// translated-code and indirect-branch mappings. Those mappings are ordinary
// process-wide `mmap` regions with no thread affinity — any thread may read or
// write them — and `addr_space` guards the whole `AddressSpace` behind a
// `Mutex`, so all access is serialized. Sharing one `Process` across the host
// threads that back the guest threads is therefore sound. This is what makes
// `Arc<Process>` (rather than a single-threaded `Rc`) the right handle: the
// address space is built to be shared by `clone(CLONE_VM)` siblings.
unsafe impl Send for Process {}
unsafe impl Sync for Process {}
