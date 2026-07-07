//! Synchronous fault handler for self-modifying code.
//!
//! JIT engines — JavaScriptCore in Bun — rewrite their own generated machine
//! code in place. Chimera translates guest code ahead of execution and caches
//! the result, so a guest store that overwrites already-translated code would
//! leave a stale translation running. Once Chimera has translated code from a
//! writable+executable (JIT) guest page it maps that page read-only on the host
//! ([`crate::sys::mmap::AddressSpace::arm_span`]); a guest store then traps here.
//! This handler drops the page's stale translations, restores write permission,
//! and returns so the faulting store re-executes and lands — and the next
//! execution of that page re-translates the new code.
//!
//! It owns the host `SIGSEGV`/`SIGBUS` disposition. The guest's own handlers for
//! those are recorded in the disposition table but never installed on the host
//! slot ([`super::signal`]), so this handler always runs first and can classify
//! the fault. Besides SMC writes it recognizes one other recoverable fault, in
//! the style of a kernel exception table: a fault whose rip lies inside the
//! `chimera_fetch_copy` primitive — the translator probing an untrusted guest
//! address for its decode window — resumes at the primitive's fixup label,
//! which reports the readable prefix. A fault that is neither — a genuine guest
//! fault, or one taken in Chimera's own code — prints a crash report
//! ([`report_fault`]) in the style of a kernel OOPS (cause, faulting addresses,
//! the full guest register file, the faulting instruction's bytes, and a slice
//! of the stack) and is then left to terminate the process with the faithful
//! signal; precise delivery into a guest fault handler is not modeled.
//!
//! The handler is async-signal-safe: it touches only atomics, a `Mutex` and
//! `HashMap` whose operations never allocate, and `mprotect`. It must not call
//! the allocator, use TLS, or panic — it runs on whatever context the store
//! interrupted, with the guest's `FS` base still loaded.

use std::{
    mem, ptr,
    sync::{
        Once,
        atomic::{AtomicPtr, Ordering},
    },
};

use crate::{
    arch::x86::{trampoline::fetch_copy_span, translate::code_cache_contains},
    process::Process,
};

/// The running guest's shared process state, published by [`set_process`] so
/// the fault handler can reach the shared address space. A raw pointer because
/// the handler cannot hold an `Arc`; the process outlives every fault.
static PROCESS: AtomicPtr<Process> = AtomicPtr::new(ptr::null_mut());

/// Publish the guest process for the fault handler, before any guest code runs.
pub fn set_process(process: &Process) {
    PROCESS.store(process as *const Process as *mut Process, Ordering::Release);
}

/// Install the synchronous `SIGSEGV`/`SIGBUS` handler, once per process.
pub fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = chimera_fault as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = libc::SA_SIGINFO;
        libc::sigaction(libc::SIGSEGV, &sa, ptr::null_mut());
        libc::sigaction(libc::SIGBUS, &sa, ptr::null_mut());
    });
}

extern "C" fn chimera_fault(
    signo: libc::c_int,
    info: *const libc::siginfo_t,
    ucontext: *mut libc::c_void,
) {
    let fault_addr = unsafe { (*info).si_addr() } as usize;
    let si_code = unsafe { (*info).si_code };
    let rip = fault_rip(ucontext);

    // A `kill()`/`tgkill()`-raised SIGSEGV or SIGBUS (`si_code <= 0`) is not a
    // memory fault: `si_addr` is meaningless there (it aliases the sender's
    // pid/uid). The common source is the guest's own crash path — node resets
    // the disposition to SIG_DFL and `raise()`s after its handler declines a
    // fault — which expects the default action. Skip the crash report and
    // terminate faithfully.
    if si_code <= 0 {
        unsafe {
            let mut sa: libc::sigaction = mem::zeroed();
            sa.sa_sigaction = libc::SIG_DFL;
            libc::sigemptyset(&mut sa.sa_mask);
            sa.sa_flags = 0;
            libc::sigaction(signo, &sa, ptr::null_mut());
            libc::raise(signo);
        }
        return;
    }

    // A memory fault inside the guarded guest-fetch span: the translator
    // probed an unreadable guest address through `chimera_fetch_copy` — a
    // wild jump, a decode window straddling into an unmapped page (SIGSEGV),
    // or one reaching past the end of a truncated file-backed mapping
    // (SIGBUS). Resume at the fixup label, which returns the readable prefix
    // length to the translator. This must stay below the kill-raised check
    // above (those must terminate regardless of the interrupted rip) and is
    // a compare against Chimera's own .text, disjoint from the code cache
    // and from guest pages, so it can never claim an SMC trap or a guest
    // fault, and it needs no per-thread state.
    let (fetch_start, fetch_fixup) = fetch_copy_span();
    if (fetch_start..fetch_fixup).contains(&rip) {
        set_fault_rip(ucontext, fetch_fixup);
        return;
    }

    // Only a fault taken while executing translated guest code can be an SMC
    // write, and only then is the address-space lock guaranteed free on this
    // thread — it is never held across dispatch — so the handler can take it
    // without deadlocking against itself.
    if code_cache_contains(rip) {
        let process = PROCESS.load(Ordering::Acquire);
        if !process.is_null() {
            let process = unsafe { &*process };
            if process.addr_space.lock().unwrap().on_smc_write(fault_addr) {
                return; // write permission restored; re-execute the store
            }
        }
    }

    // Not an SMC write: a genuine guest fault, or a fault in Chimera itself.
    // Print a crash report, then restore the default disposition and return; the
    // faulting instruction re-executes and the kernel terminates the process with
    // the real signal (and any core dump).
    report_fault(signo, si_code, fault_addr, rip, ucontext);
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0;
        libc::sigaction(signo, &sa, ptr::null_mut());
    }
}

/// A fixed-size, allocation-free line writer to `stderr`. The crash report runs
/// in a signal handler, so it must not touch the allocator; it builds text in a
/// stack buffer and `write(2)`s it. Each `write` is unbuffered, so the most
/// important lines (the registers) reach the terminal even if a later, riskier
/// memory read (the code or stack dump) faults and the handler is killed.
struct Report {
    buf: [u8; 512],
    len: usize,
}

impl Report {
    fn new() -> Self {
        Report {
            buf: [0; 512],
            len: 0,
        }
    }

    fn flush(&mut self) {
        if self.len != 0 {
            unsafe { libc::write(2, self.buf.as_ptr() as *const libc::c_void, self.len) };
            self.len = 0;
        }
    }

    fn str(&mut self, s: &[u8]) {
        for &b in s {
            if self.len == self.buf.len() {
                self.flush();
            }
            self.buf[self.len] = b;
            self.len += 1;
        }
    }

    /// A fixed-width 16-digit hex value (`0x0000000000000000`), for aligned dumps.
    fn hex(&mut self, v: u64) {
        self.str(b"0x");
        for i in (0..16).rev() {
            let nib = ((v >> (i * 4)) & 0xf) as usize;
            let c = b"0123456789abcdef"[nib];
            if self.len == self.buf.len() {
                self.flush();
            }
            self.buf[self.len] = c;
            self.len += 1;
        }
    }

    /// One labelled register: `  name=0x....`.
    fn reg(&mut self, name: &[u8], v: u64) {
        self.str(b"  ");
        self.str(name);
        self.str(b"=");
        self.hex(v);
    }
}

/// `gregs` index of a register in the signal `ucontext` (see `<sys/ucontext.h>`).
const REG: [(&[u8], usize); 18] = [
    (b"rax", 13),
    (b"rbx", 11),
    (b"rcx", 14),
    (b"rdx", 12),
    (b"rsi", 9),
    (b"rdi", 8),
    (b"rbp", 10),
    (b"rsp", 15),
    (b"r8 ", 0),
    (b"r9 ", 1),
    (b"r10", 2),
    (b"r11", 3),
    (b"r12", 4),
    (b"r13", 5),
    (b"r14", 6),
    (b"r15", 7),
    (b"rip", 16),
    (b"efl", 17),
];

/// Print a kernel-OOPS-style crash report for a fatal guest fault: the signal and
/// its cause, the faulting addresses, the full guest register file (translated
/// code keeps the guest registers in the host registers, so the `ucontext` *is*
/// the guest state), the bytes of the faulting translated instruction, and a
/// slice of the guest stack. Allocation-free and best-effort: the register dump
/// comes first so it survives even if a later memory read faults.
fn report_fault(
    signo: libc::c_int,
    si_code: libc::c_int,
    fault_addr: usize,
    rip: usize,
    ucontext: *mut libc::c_void,
) {
    let in_cache = code_cache_contains(rip);
    let mut r = Report::new();

    r.str(b"\nchimera: guest received fatal ");
    r.str(signal_name(signo));
    r.str(b"\n");
    r.str(b"  cause:   ");
    r.str(fault_cause(signo, si_code));
    r.str(b"\n  address: ");
    r.hex(fault_addr as u64);
    r.str(b"\n  host rip:  ");
    r.hex(rip as u64);
    r.str(if in_cache {
        b"  (translated code)\n"
    } else {
        b"  (chimera runtime)\n"
    });
    // The guest PC published at the last block boundary. GS holds the running
    // thread's ThreadState whenever translated code executes, so this is only
    // meaningful — and safe to read — when the fault was taken in the cache.
    if in_cache {
        r.str(b"  guest pc: ");
        r.hex(guest_pc_slot());
        r.str(b"  (last block boundary)\n");
    }

    r.str(b"\nguest registers:\n");
    if !ucontext.is_null() {
        let gregs = unsafe { &(*(ucontext as *const libc::ucontext_t)).uc_mcontext.gregs };
        for (i, (name, idx)) in REG.iter().enumerate() {
            r.reg(name, gregs[*idx] as u64);
            if i % 4 == 3 {
                r.str(b"\n");
            }
        }
        r.str(b"\n");
    }
    r.flush();

    // Riskier reads (they dereference guest memory) go last, after the registers
    // are already on the terminal. The faulting instruction lives in the code
    // cache, which is mapped and readable, so dumping it is safe.
    if in_cache {
        r.str(b"\ncode (bytes at host rip):\n ");
        for off in 0..16usize {
            let byte = unsafe { ptr::read_volatile((rip + off) as *const u8) };
            r.str(b" ");
            let hi = b"0123456789abcdef"[(byte >> 4) as usize];
            let lo = b"0123456789abcdef"[(byte & 0xf) as usize];
            r.str(&[hi, lo]);
        }
        r.str(b"\n");
        r.flush();
    }

    // The guest stack: walk upward from rsp (away from any guard page below it),
    // so this read stays within the live stack even for a stack-overflow fault.
    if !ucontext.is_null() {
        let rsp = unsafe { (*(ucontext as *const libc::ucontext_t)).uc_mcontext.gregs[15] } as u64;
        r.str(b"\nstack (from rsp):\n");
        for i in 0..8u64 {
            let addr = rsp + i * 16;
            r.str(b"  ");
            r.hex(addr);
            r.str(b":");
            for w in 0..2u64 {
                r.str(b" ");
                r.hex(unsafe { ptr::read_volatile((addr + w * 8) as *const u64) });
            }
            r.str(b"\n");
        }
        r.flush();
    }
    r.str(b"\n");
    r.flush();
}

/// Read the guest PC the dispatcher last published, from the running thread's
/// `ThreadState` via `gs:[rip_slot]`. Caller must ensure GS holds a ThreadState
/// (true whenever a fault is taken in the code cache).
fn guest_pc_slot() -> u64 {
    // ThreadState::rip lives at offset 128 (16 GPR slots of 8 bytes each).
    let v: u64;
    unsafe {
        std::arch::asm!("mov {x}, gs:[128]", x = out(reg) v, options(nostack, readonly, preserves_flags));
    }
    v
}

/// The mnemonic for a fatal signal number.
fn signal_name(signo: libc::c_int) -> &'static [u8] {
    match signo {
        libc::SIGSEGV => b"SIGSEGV (segmentation fault)",
        libc::SIGBUS => b"SIGBUS (bus error)",
        _ => b"signal",
    }
}

/// A human-readable description of the fault's `si_code` — for SIGSEGV this
/// distinguishes a write to read-only memory (`SEGV_ACCERR`, the shape of a
/// self-modifying-code or guard-page hit) from a dereference of unmapped memory
/// (`SEGV_MAPERR`, a wild pointer).
fn fault_cause(signo: libc::c_int, si_code: libc::c_int) -> &'static [u8] {
    // `si_code` values from `<asm-generic/siginfo.h>` (not all exposed by `libc`).
    const SEGV_MAPERR: libc::c_int = 1;
    const SEGV_ACCERR: libc::c_int = 2;
    const BUS_ADRALN: libc::c_int = 1;
    const BUS_ADRERR: libc::c_int = 2;
    const BUS_OBJERR: libc::c_int = 3;
    match (signo, si_code) {
        (libc::SIGSEGV, SEGV_MAPERR) => b"address not mapped to an object",
        (libc::SIGSEGV, SEGV_ACCERR) => b"invalid permissions for the mapped object",
        (libc::SIGSEGV, _) => b"invalid memory access",
        (libc::SIGBUS, BUS_ADRALN) => b"invalid address alignment",
        (libc::SIGBUS, BUS_ADRERR) => b"non-existent physical address",
        (libc::SIGBUS, BUS_OBJERR) => b"object-specific hardware error",
        (libc::SIGBUS, _) => b"bus error",
        _ => b"unknown",
    }
}

/// Read the faulting instruction pointer from the signal's `ucontext`.
fn fault_rip(ucontext: *mut libc::c_void) -> usize {
    if ucontext.is_null() {
        return 0;
    }
    let uc = ucontext as *const libc::ucontext_t;
    unsafe { (*uc).uc_mcontext.gregs[libc::REG_RIP as usize] as usize }
}

/// Write a new instruction pointer into the signal's `ucontext`; `sigreturn`
/// resumes the interrupted thread there.
fn set_fault_rip(ucontext: *mut libc::c_void, rip: usize) {
    if ucontext.is_null() {
        return;
    }
    let uc = ucontext as *mut libc::ucontext_t;
    unsafe { (*uc).uc_mcontext.gregs[libc::REG_RIP as usize] = rip as libc::greg_t };
}
