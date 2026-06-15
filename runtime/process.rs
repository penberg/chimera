//! Process-wide guest state shared by every thread.
//!
//! A guest process is one [`Process`] and one or more [`crate::arch::dispatch`]
//! `Thread`s. The `Thread` holds the per-thread register file; the `Process`
//! holds everything the threads share — today the guest address space and its
//! translated-block cache, the analogue of the kernel's `mm_struct`. Each
//! `Thread` keeps an `Arc<Process>`, so a future `clone(CLONE_VM)` hands the
//! child the same `Arc` and the two threads translate into and map within one
//! address space.

use std::sync::Mutex;

use crate::{Error, sys::mmap::AddressSpace};

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
}

impl Process {
    pub fn new(code_cache_size: usize) -> Result<Self, Error> {
        Ok(Self {
            addr_space: Mutex::new(AddressSpace::new(code_cache_size)?),
        })
    }
}

// SAFETY: `AddressSpace` is not auto-`Send`/`Sync` only because its code cache
// holds raw pointers (`*mut u8`) into the translated-code and indirect-branch
// mappings. Those mappings are ordinary process-wide `mmap` regions with no
// thread affinity — any thread may read or write them — and the sole field of
// `Process` guards the whole `AddressSpace` behind a `Mutex`, so all access is
// serialized. Sharing one `Process` across the host threads that back the guest
// threads is therefore sound. This is what makes `Arc<Process>` (rather than a
// single-threaded `Rc`) the right handle: the address space is built to be
// shared by `clone(CLONE_VM)` siblings.
unsafe impl Send for Process {}
unsafe impl Sync for Process {}
