//! Cached debug-trace flags.
//!
//! Read once at startup so the translate/dispatch hot path never calls `getenv`.
//! On Darwin the runtime and the guest share one libSystem, and the guest runs
//! on Chimera's own host thread. A `getenv` locks libSystem's environ
//! `os_unfair_lock`; if the guest's *translated* `getenv` is mid-flight holding
//! that lock when it exits to the dispatcher at a block boundary, a runtime
//! `getenv` on the same host thread is seen as re-locking a lock it already owns
//! and libplatform aborts ("BUG IN CLIENT OF LIBPLATFORM: Trying to recursively
//! lock an os_unfair_lock"). Reading the flags once, before the guest executes,
//! keeps the hot path off every shared libSystem facility.

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

static FLAGS: AtomicU8 = AtomicU8::new(0);

/// `CHIMERA_BREAK`: a guest PC (hex, or `main+hex` for a main-image offset
/// the dispatcher resolves against the image slide) whose blocks get a full
/// register dump on entry — a poor man's breakpoint for spins the profiler
/// can locate but not explain. Implies the dispatcher sees every block, so
/// set `CHIMERA_NO_LINK` alongside it.
static BREAK_PC: AtomicU64 = AtomicU64::new(0);
static BREAK_IS_MAIN_OFFSET: AtomicU8 = AtomicU8::new(0);
static BREAK_BUDGET: AtomicU32 = AtomicU32::new(20);
/// `CHIMERA_BREAK_SKIP`: hits to ignore before dumping, to reach the
/// interesting phase of a loop that also runs healthily early on.
static BREAK_SKIP: AtomicU32 = AtomicU32::new(0);

const TRACE: u8 = 1 << 0;
const TRACE_EMIT: u8 = 1 << 1;
const NO_LINK: u8 = 1 << 2;
const PROFILE: u8 = 1 << 3;
const DYLD_TRACE: u8 = 1 << 4;

/// Sample the trace environment once. Call before any guest block runs (from
/// `arch::init`), so no later `getenv` races a lock the guest holds.
pub fn init() {
    let mut flags = 0;
    if std::env::var_os("CHIMERA_TRACE").is_some() {
        flags |= TRACE;
    }
    if std::env::var_os("CHIMERA_TRACE_EMIT").is_some() {
        flags |= TRACE_EMIT;
    }
    if std::env::var_os("CHIMERA_NO_LINK").is_some() {
        flags |= NO_LINK;
    }
    if std::env::var_os("CHIMERA_PROFILE").is_some() {
        flags |= PROFILE;
    }
    if std::env::var_os("CHIMERA_DYLD_TRACE").is_some() {
        flags |= DYLD_TRACE;
    }
    if let Some(spec) = std::env::var_os("CHIMERA_BREAK") {
        let spec = spec.to_string_lossy();
        let (offset, is_main) = match spec.strip_prefix("main+") {
            Some(hex) => (hex, true),
            None => (spec.as_ref(), false),
        };
        if let Ok(pc) = u64::from_str_radix(offset.trim_start_matches("0x"), 16) {
            BREAK_PC.store(pc, Ordering::Relaxed);
            BREAK_IS_MAIN_OFFSET.store(is_main as u8, Ordering::Relaxed);
        }
    }
    if let Some(skip) = std::env::var_os("CHIMERA_BREAK_SKIP")
        && let Ok(n) = skip.to_string_lossy().parse::<u32>()
    {
        BREAK_SKIP.store(n, Ordering::Relaxed);
    }
    FLAGS.store(flags, Ordering::Relaxed);
}

/// Whether `pc` is the requested break PC with dump budget remaining. The
/// budget keeps a break inside a spin loop from flooding the terminal.
pub fn break_hit(pc: u64, main_base: u64) -> bool {
    let target = BREAK_PC.load(Ordering::Relaxed);
    if target == 0 {
        return false;
    }
    let target = if BREAK_IS_MAIN_OFFSET.load(Ordering::Relaxed) != 0 {
        main_base + target
    } else {
        target
    };
    pc == target
        && BREAK_SKIP
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_err()
        && BREAK_BUDGET
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
}

/// Whether `CHIMERA_TRACE` was set (per-block PC + syscall/mmap logging).
pub fn trace() -> bool {
    FLAGS.load(Ordering::Relaxed) & TRACE != 0
}

/// Whether `CHIMERA_TRACE_EMIT` was set (translated-words dump per block).
pub fn trace_emit() -> bool {
    FLAGS.load(Ordering::Relaxed) & TRACE_EMIT != 0
}

/// Whether `CHIMERA_NO_LINK` was set: keep the code cache dispatcher-only —
/// no direct-branch links and no inline indirect-branch probe, so every block
/// returns to the run loop. Slow, but it is the difference that isolates a
/// miscompiled terminator from a miscompiled instruction when a guest
/// misbehaves, and it makes the run loop's per-block trace complete again.
pub fn no_link() -> bool {
    FLAGS.load(Ordering::Relaxed) & NO_LINK != 0
}

/// Whether `CHIMERA_PROFILE` was set: sample every guest thread's PC on a
/// timer and report the hottest, in guest addresses. Answers "what is the
/// guest actually doing" for a run that is slow rather than wrong — the one
/// question a per-block trace cannot answer, since it is far too slow to
/// leave on and a linked chain never returns to the run loop to be traced.
pub fn profile() -> bool {
    FLAGS.load(Ordering::Relaxed) & PROFILE != 0
}

/// Whether `CHIMERA_DYLD_TRACE` was set (per-image linker logging). Cached
/// like the rest: the linker runs while the guest is live — a guest `dlopen`
/// is serviced through it — so reading the environment there is exactly the
/// recursive-lock abort this module exists to avoid.
pub fn dyld_trace() -> bool {
    FLAGS.load(Ordering::Relaxed) & DYLD_TRACE != 0
}
