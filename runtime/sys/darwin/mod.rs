//! Darwin-specific glue: Mach-O loading, the in-process linker, the macOS
//! process bootstrap (`argc`/`argv`/`envp`/`apple`), and the entry point that
//! hands the guest off to the dispatcher.
//!
//! Phase 1 of the Darwin port lands the loader (`macho`); the in-process linker
//! (`dyld`) and the bootstrap/entry (`exec`) follow in later tasks, at which
//! point `sys::exec` is re-exported host-neutrally the way `sys::linux` is.

use std::{
    os::unix::ffi::OsStrExt,
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicI32, AtomicU64, Ordering},
    },
};

use crate::sys::mmap::{copy_from_guest, copy_to_guest};

pub mod callback;
pub mod dyld;
pub mod exec;
pub mod fault;
pub mod handoff;
pub mod macho;
pub mod policy;
pub mod signal;
pub mod spawn;
pub mod syscall;
pub mod thread;

/// The guest image's path, as `_NSGetExecutablePath` must report it. Set by
/// `exec` for each image installed; read by the dispatch loop's interception
/// of that function (see [`ns_get_executable_path`]).
static EXECUTABLE_PATH: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// The guest program's name — its path's last component, NUL-terminated —
/// in storage whose address is stable, since `getprogname` hands the caller a
/// pointer into libSystem's own copy.
static PROGNAME: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static PROGNAME_PTR: AtomicU64 = AtomicU64::new(0);

/// The slide applied to the image now running as the guest, so a guest
/// address reported by the profiler can be turned back into a file offset.
static IMAGE_SLIDE: AtomicU64 = AtomicU64::new(0);

pub fn set_image_slide(slide: u64) {
    IMAGE_SLIDE.store(slide, Ordering::Release);
}

/// The guest main thread's stack — the runtime-allocated mapping it actually
/// runs on — and the host `pthread_t` of the thread running it. libpthread
/// records the *host* main stack for that thread, so a guest asking
/// `pthread_get_stackaddr_np`/`pthread_get_stacksize_np` about it would get
/// bounds its own stack pointer lies outside; JavaScriptCore's stack
/// sanitizer release-asserts on exactly that. The dispatch loop answers those
/// queries from here instead. Worker threads need no override: their host
/// pthreads are created with `pthread_attr_setstack` on the guest stack, so
/// libpthread's answer is already the guest's.
static MAIN_STACK_LO: AtomicU64 = AtomicU64::new(0);
static MAIN_STACK_LEN: AtomicU64 = AtomicU64::new(0);
static MAIN_PTHREAD: AtomicU64 = AtomicU64::new(0);

pub fn set_main_stack(lo: u64, len: u64) {
    MAIN_STACK_LO.store(lo, Ordering::Release);
    MAIN_STACK_LEN.store(len, Ordering::Release);
    MAIN_PTHREAD.store(unsafe { libc::pthread_self() } as u64, Ordering::Release);
}

/// `pthread_get_stackaddr_np`: a stack's base is its highest address.
pub fn guest_stackaddr(thread: u64) -> u64 {
    if thread == MAIN_PTHREAD.load(Ordering::Acquire) {
        return MAIN_STACK_LO.load(Ordering::Acquire) + MAIN_STACK_LEN.load(Ordering::Acquire);
    }
    unsafe { libc::pthread_get_stackaddr_np(thread as usize as libc::pthread_t) as u64 }
}

pub fn guest_stacksize(thread: u64) -> u64 {
    if thread == MAIN_PTHREAD.load(Ordering::Acquire) {
        return MAIN_STACK_LEN.load(Ordering::Acquire);
    }
    unsafe { libc::pthread_get_stacksize_np(thread as usize as libc::pthread_t) as u64 }
}

/// The guest's view of the malloc zone list, kept apart from the real one: a
/// zone the guest registers carries *guest* function pointers, and on the
/// shared list the runtime's own malloc would call them natively (observed as
/// a SIGBUS inside `malloc_size` once rustc's jemalloc constructor ran). The
/// virtual list starts as a snapshot of the real zones and then follows
/// libmalloc's semantics — register appends, unregister swaps the last zone
/// into the vacated slot — which is exactly what jemalloc's "reorder until
/// ours is the default" constructor loop needs to converge. `None` until the
/// guest first touches the zone API; an execve resets it; a fork child
/// rightly inherits it.
static GUEST_ZONES: Mutex<Option<Vec<u64>>> = Mutex::new(None);

/// Guest-readable handout buffer for `malloc_get_all_zones` answers: the
/// guest reads runtime memory freely (one address space).
static ZONE_HANDOUT: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

unsafe extern "C" {
    fn malloc_get_all_zones(
        task: libc::c_uint,
        reader: *mut libc::c_void,
        addresses: *mut *mut u64,
        count: *mut libc::c_uint,
    ) -> libc::c_int;
}

fn with_guest_zones<R>(f: impl FnOnce(&mut Vec<u64>) -> R) -> R {
    let mut guard = GUEST_ZONES.lock().unwrap();
    let zones = guard.get_or_insert_with(|| {
        let mut addresses: *mut u64 = std::ptr::null_mut();
        let mut count: libc::c_uint = 0;
        let real =
            unsafe { malloc_get_all_zones(0, std::ptr::null_mut(), &mut addresses, &mut count) };
        if real == 0 && !addresses.is_null() {
            (0..count as usize)
                .map(|i| unsafe { *addresses.add(i) })
                .collect()
        } else {
            Vec::new()
        }
    });
    f(zones)
}

pub fn guest_zone_register(zone: u64) {
    with_guest_zones(|zones| {
        if !zones.contains(&zone) {
            zones.push(zone);
        }
        if crate::trace::trace() {
            eprintln!("chimera: zones after register {zone:#x}: {zones:x?}");
        }
    });
}

pub fn guest_zone_unregister(zone: u64) {
    with_guest_zones(|zones| {
        if let Some(i) = zones.iter().position(|&z| z == zone) {
            // libmalloc moves the *last* zone into the vacated slot — for
            // slot 0, this is what lets a reorder dance promote a new front.
            zones.swap_remove(i);
        }
        if crate::trace::trace() {
            eprintln!("chimera: zones after unregister {zone:#x}: {zones:x?}");
        }
    });
}

pub fn guest_zone_default() -> u64 {
    with_guest_zones(|zones| zones.first().copied().unwrap_or(0))
}

/// Fill the handout buffer with the virtual list; returns `(array, count)`.
pub fn guest_zone_list() -> (u64, u32) {
    with_guest_zones(|zones| {
        let n = zones.len().min(ZONE_HANDOUT.len());
        for (slot, &zone) in ZONE_HANDOUT.iter().zip(zones.iter()).take(n) {
            slot.store(zone, Ordering::Release);
        }
        (ZONE_HANDOUT.as_ptr() as u64, n as u32)
    })
}

pub fn clear_guest_zones() {
    *GUEST_ZONES.lock().unwrap() = None;
}

/// Guest-registered `pthread_atfork` handlers, kept off libpthread's global
/// list. Registered there, they would be *guest* function pointers on a list
/// the host's own `fork(3)` wrapper walks natively: the runtime's posix_spawn
/// emulation forks natively, and an untranslated guest handler faulting
/// inside the atfork window deadlocks the fault handler against the locks
/// fork-prepare holds. Keeping them here, the spawn fork never sees them — a
/// real posix_spawn does not run user atfork handlers either — and the
/// guest's own fork runs them translated around the trap (see
/// `spawn::forked`).
static GUEST_ATFORK: Mutex<Vec<AtforkHandlers>> = Mutex::new(Vec::new());

#[derive(Clone, Copy)]
pub struct AtforkHandlers {
    pub prepare: u64,
    pub parent: u64,
    pub child: u64,
}

pub fn guest_atfork_register(prepare: u64, parent: u64, child: u64) {
    GUEST_ATFORK.lock().unwrap().push(AtforkHandlers {
        prepare,
        parent,
        child,
    });
}

pub fn guest_atfork_handlers() -> Vec<AtforkHandlers> {
    GUEST_ATFORK.lock().unwrap().clone()
}

pub fn clear_guest_atfork() {
    GUEST_ATFORK.lock().unwrap().clear();
}

pub fn image_slide() -> u64 {
    IMAGE_SLIDE.load(Ordering::Acquire)
}

/// Record the path of the image now running as the guest.
pub fn set_executable_path(path: &Path) {
    *EXECUTABLE_PATH.lock().unwrap() = path.as_os_str().as_bytes().to_vec();

    let mut name = path
        .file_name()
        .unwrap_or(path.as_os_str())
        .as_bytes()
        .to_vec();
    name.push(0);
    let mut progname = PROGNAME.lock().unwrap();
    *progname = name;
    PROGNAME_PTR.store(progname.as_ptr() as u64, Ordering::Release);
}

/// Service the guest's `getprogname()`.
///
/// libSystem's copy is the one dyld set from *its* startup argv, so a guest
/// asking its own name is told the runtime's. Tools decide real things from
/// it: clang looks up its toolchain by program name, and asked xcodebuild to
/// `-find chimera`. Hand back the guest program's name instead.
pub fn getprogname() -> u64 {
    PROGNAME_PTR.load(Ordering::Acquire)
}

/// The guest's `argc`, `argv`, and `envp`, in storage whose *address* is
/// stable: `_NSGetArgv` and friends hand back a pointer to dyld's own
/// variable, which the caller then dereferences, so these slots stand in for
/// those variables. Written by `exec` for each image installed.
static GUEST_ARGC: AtomicI32 = AtomicI32::new(0);
static GUEST_ARGV: AtomicU64 = AtomicU64::new(0);
static GUEST_ENVP: AtomicU64 = AtomicU64::new(0);
static GUEST_APPLE: AtomicU64 = AtomicU64::new(0);

/// Record the entry frame of the image now running as the guest.
pub fn set_guest_args(argc: i32, argv: u64, envp: u64, apple: u64) {
    GUEST_ARGC.store(argc, Ordering::Release);
    GUEST_ARGV.store(argv, Ordering::Release);
    GUEST_ENVP.store(envp, Ordering::Release);
    GUEST_APPLE.store(apple, Ordering::Release);
}

/// The `(argc, argv, envp, apple)` dyld would pass an image initializer —
/// a guest-`dlopen`'d module's constructors get the same frame `main` did.
pub fn guest_frame() -> [u64; 4] {
    [
        GUEST_ARGC.load(Ordering::Acquire) as u64,
        GUEST_ARGV.load(Ordering::Acquire),
        GUEST_ENVP.load(Ordering::Acquire),
        GUEST_APPLE.load(Ordering::Acquire),
    ]
}

/// Service the guest's `_NSGetArgv()` / `_NSGetArgc()`.
///
/// dyld exposes the process's argument vector through these, and the copy it
/// exposes is the one *it* captured at startup — Chimera's. A guest's `main`
/// gets the right `argc`/`argv` (the runtime builds that frame itself), but
/// anything reading them through dyld instead sees the runtime's command
/// line: Rust's `std::env::args` is one such caller, so a guest that
/// dispatches on its own name — rustup's proxies pick which tool to be that
/// way — behaves as though it were Chimera. Hand back the guest's frame.
///
/// `_NSGetEnviron` is deliberately *not* answered this way. The environment
/// is not identity: the runtime's `environ` already holds the environment the
/// guest was launched with, and it is the same array libSystem's own
/// `setenv`/`getenv` maintain — pointing the guest at a private copy would
/// split those two views, so a guest's `setenv` would not be visible to its
/// own `getenv`.
pub fn ns_get_argv() -> u64 {
    &GUEST_ARGV as *const AtomicU64 as u64
}

pub fn ns_get_argc() -> u64 {
    &GUEST_ARGC as *const AtomicI32 as u64
}

/// Service the guest's `_NSGetExecutablePath(buf, bufsize)`.
///
/// The runtime and the guest share one already-initialized dyld, whose cached
/// executable path is *Chimera's* — so a guest that asks who it is (rustup's
/// proxies dispatch on exactly this, and `std::env::current_exe` is built on
/// it) gets the wrong answer, with no way to correct it from the guest's
/// `apple[]` array, which nothing re-reads after dyld's own startup. The
/// dispatch loop therefore recognizes a call to it and answers here, the way
/// it already answers the TLV thunk.
///
/// Returns the function's own result: 0 on success, -1 with the required size
/// written back when the buffer is too small.
pub fn ns_get_executable_path(buf: u64, bufsize_ptr: u64) -> i64 {
    let path = EXECUTABLE_PATH.lock().unwrap();
    let needed = path.len() + 1;
    if bufsize_ptr == 0 {
        return -1;
    }
    let mut size = [0u8; 4];
    if !copy_from_guest(bufsize_ptr, &mut size) {
        return -1;
    }
    let size = u32::from_ne_bytes(size) as usize;
    if size < needed || buf == 0 {
        copy_to_guest(bufsize_ptr, &(needed as u32).to_ne_bytes());
        return -1;
    }
    let mut out = path.clone();
    out.push(0);
    if !copy_to_guest(buf, &out) {
        return -1;
    }
    0
}

/// Write the guest image's path into a guest buffer, NUL-terminated, as the
/// kernel's `PROC_PIDPATHINFO` does. `Err` carries the errno for a buffer the
/// path does not fit in or that cannot be written.
pub fn write_executable_path(buf: u64, bufsize: usize) -> Result<(), i32> {
    let path = EXECUTABLE_PATH.lock().unwrap();
    if buf == 0 {
        return Err(libc::EFAULT);
    }
    if bufsize < path.len() + 1 {
        return Err(libc::ENOMEM);
    }
    let mut out = path.clone();
    out.push(0);
    if copy_to_guest(buf, &out) {
        Ok(())
    } else {
        Err(libc::EFAULT)
    }
}

unsafe extern "C" {
    fn fpurge(stream: *mut libc::FILE) -> libc::c_int;
    static mut __stdinp: *mut libc::FILE;
    static mut __stdoutp: *mut libc::FILE;
    static mut __stderrp: *mut libc::FILE;
}

/// Discard whatever the guest left buffered in the shared libSystem standard
/// streams. Called when the guest traps BSD `exit`: its libc `exit(3)` has
/// already flushed, in userspace, everything it meant to flush, so any data
/// still buffered here is data a native `_exit(2)` would discard — but the
/// runtime and guest share one libSystem, and Chimera's own eventual exit
/// flushes every `FILE`, which would emit it (duplicated parent output in a
/// fork child, phantom output after `_exit`).
pub fn purge_guest_stdio() {
    unsafe {
        fpurge(__stdinp);
        fpurge(__stdoutp);
        fpurge(__stderrp);
    }
}
