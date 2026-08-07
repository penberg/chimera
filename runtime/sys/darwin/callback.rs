//! Guest callbacks that a system library invokes on its own threads.
//!
//! A guest hands a library a function pointer — a dispatch block, a queued
//! work item — and the library calls it later, from a thread it owns. Those
//! threads are not guest threads: the runtime never created them, they carry
//! no translator context, and calling a guest pointer there runs guest code
//! natively, off the page protections and the sandbox both. It is the same
//! hazard as the loader calling a guest `atexit` handler, but the library
//! keeps its threading, so the callback cannot simply be run at the
//! registration site: [`crate::arch::dispatch`]'s `dispatch_apply` escape can
//! do that only because that API permits it, and trying the same for
//! `dispatch_async` deadlocked libdispatch's own bookkeeping.
//!
//! So the library keeps its callback and its thread, and gets a *runtime*
//! function pointer instead of the guest's. When it calls that, [`shim`] runs
//! natively on the library's thread, builds a translator context there, and
//! runs the guest's function through it.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
};

use crate::{arch::dispatch::run_guest_callback, process::Process};

/// The running guest process, for callbacks that arrive on a thread with no
/// context of their own. An `Arc` rather than the raw pointer
/// [`super::fault`] keeps, because a callback builds a whole `Thread` around
/// it rather than touching two atomics.
static PROCESS: Mutex<Option<Arc<Process>>> = Mutex::new(None);

/// Publish the guest process for callbacks, before any guest code runs.
pub fn set_process(process: &Arc<Process>) {
    *PROCESS.lock().unwrap() = Some(Arc::clone(process));
}

/// What the library holds on the runtime's behalf: the guest function to call
/// and the argument to call it with. One per registration, owned here so the
/// library's copy/release of *its* context cannot reach guest code.
struct Callback {
    func: u64,
    arg: u64,
    /// A guest block to release, as a guest call, once the callback has run —
    /// the retain the runtime took to keep it alive on the queue.
    release: Option<(u64, u64)>,
}

/// The function pointer handed to the library in place of the guest's. Runs
/// on whatever thread the library chose.
pub extern "C" fn shim(context: *mut libc::c_void) {
    let cb = unsafe { Box::from_raw(context as *mut Callback) };
    let Some(process) = PROCESS.lock().unwrap().clone() else {
        return;
    };
    // A callback address that cannot be executed — null, misaligned (arm64
    // wants 4), or non-canonical — means the registration produced garbage
    // rather than a guest function; running it would hand the translator an
    // address it must then reject mid-fetch. Say so and drop the callback:
    // the library's bookkeeping is already satisfied by this call returning.
    if cb.func == 0 || cb.func % 4 != 0 || cb.func >> 48 != 0 {
        eprintln!("chimera: dropping a guest callback to {:#x}", cb.func);
        return;
    }
    run_guest_callback(&process, cb.func, cb.arg);
    if let Some((release, block)) = cb.release {
        run_guest_callback(&process, release, block);
    }
}

/// Wrap `(func, arg)` for the library, returning the context to hand to
/// [`shim`]. Ownership passes to the library and returns in [`shim`].
pub fn wrap(func: u64, arg: u64, release: Option<(u64, u64)>) -> u64 {
    Box::into_raw(Box::new(Callback { func, arg, release })) as u64
}

/// Whether this process is a fork child. Never cleared once set: an execve
/// here is in-process, so what fork broke stays broken for the process's
/// whole remaining lifetime.
static FORK_CHILD: AtomicBool = AtomicBool::new(false);

/// The fork child's stand-in for libdispatch's worker pool: `(context,
/// group)` items for a runtime-owned thread that runs them through [`shim`].
/// Reset by [`mark_fork_child`] — a fork inherits the sender but not the
/// worker thread behind it, so a grandchild must not send into a channel
/// nothing drains.
static WORKER: Mutex<Option<Sender<(u64, u64)>>> = Mutex::new(None);

unsafe extern "C" {
    fn dispatch_group_enter(group: u64);
    fn dispatch_group_leave(group: u64);
}

/// Record that this process is the child of a fork.
pub fn mark_fork_child() {
    FORK_CHILD.store(true, Ordering::Release);
    *WORKER.lock().unwrap() = None;
}

pub fn is_fork_child() -> bool {
    FORK_CHILD.load(Ordering::Acquire)
}

/// Queue an async callback in a fork child, where handing it to libdispatch
/// would crash: Mach port names do not survive fork, so the parent's worker
/// state — workqueue ports, pool semaphores — is stale in the child, and a
/// native exec would rebuild it where the in-process execve cannot. A
/// runtime-owned thread runs the callbacks instead, in submission order. A
/// non-zero `group` is a `dispatch_group_async`: entered here, before the
/// caller can observe the count, and left once its callback has run — group
/// membership is object-local atomics, which fork preserves.
pub fn enqueue(context: u64, group: u64) {
    if group != 0 {
        unsafe { dispatch_group_enter(group) };
    }
    let mut worker = WORKER.lock().unwrap();
    let sender = worker.get_or_insert_with(|| {
        let (tx, rx) = std::sync::mpsc::channel::<(u64, u64)>();
        std::thread::spawn(move || {
            for (context, group) in rx {
                shim(context as *mut libc::c_void);
                if group != 0 {
                    unsafe { dispatch_group_leave(group) };
                }
            }
        });
        tx
    });
    let _ = sender.send((context, group));
}
