//! Darwin synchronous-fault handling: the arm64 analogue of
//! `sys/linux/fault.rs`.
//!
//! Owns the host `SIGSEGV`/`SIGBUS` disposition. A fault taken while executing
//! translated code is a guest fault (or, once SMC arming exists on Darwin, a
//! write-trap to route to [`crate::sys::mmap::AddressSpace::on_smc_write`]);
//! it prints a kernel-OOPS-style crash report — cause, faulting addresses, the
//! guest register file, the faulting instruction words, a slice of the guest
//! stack — and then terminates the process with the faithful signal. A fault in
//! Chimera's own Rust is reported the same way (marked as runtime) so a runtime
//! bug still crashes loudly. Delivery into a guest fault handler is Phase 4.4.
//!
//! Installing a handler also displaces the `SIGSEGV`/`SIGBUS` handler Rust's
//! std installs for stack-overflow detection, which would otherwise swallow a
//! guest fault and spin re-executing the faulting instruction forever.
//!
//! The handler is async-signal-safe: it touches only atomics, the pthread TSD
//! slot (a plain array read on Darwin), a `Mutex` and `HashMap` whose
//! operations never allocate, and `mprotect`. It must not call the allocator
//! or panic. Crucially it must run on an alternate stack (`SA_ONSTACK` +
//! [`install_altstack`] per thread): while translated code executes, the host
//! `sp` *is* the guest stack, so a guest stack-overflow fault delivered on the
//! current stack would double-fault before the handler ran a single
//! instruction.

use std::{
    mem, ptr,
    sync::{
        Once,
        atomic::{AtomicPtr, AtomicUsize, Ordering},
    },
};

use crate::{
    arch::arm64::{cache::code_cache_contains, trampoline::fault_resume},
    process::Process,
    sys::darwin::signal::{MContext64, UContext, guest_handles, record_fault},
};

/// The running guest's shared process state, published by [`set_process`] so
/// the fault handler can reach the shared address space. A raw pointer because
/// the handler cannot hold an `Arc`; the process outlives every fault.
static PROCESS: AtomicPtr<Process> = AtomicPtr::new(ptr::null_mut());

/// Publish the guest process for the fault handler, before any guest code runs.
pub fn set_process(process: &Process) {
    PROCESS.store(process as *const Process as *mut Process, Ordering::Release);
}

/// Install the synchronous fault handler, once per process.
///
/// `SIGTRAP` joins `SIGSEGV`/`SIGBUS` because a guest `brk` — the instruction
/// behind `__builtin_trap`, a failed assertion, a sanitizer check — is copied
/// into the code cache verbatim and raises it from translated code. Without a
/// handler the process died silently with no report at all, which is a poor
/// way to learn that a guest asserted.
pub fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = chimera_fault as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigaction(libc::SIGSEGV, &sa, ptr::null_mut());
        libc::sigaction(libc::SIGBUS, &sa, ptr::null_mut());
        libc::sigaction(libc::SIGTRAP, &sa, ptr::null_mut());
    });
}

/// Give the calling thread an alternate signal stack, so a fault taken while
/// the host `sp` points into the guest's (possibly exhausted) stack can still
/// deliver. Called once per host thread that runs guest code; the stack is
/// leaked deliberately — it must outlive every fault the thread can take.
pub fn install_altstack() {
    unsafe {
        let mut current: libc::stack_t = mem::zeroed();
        if libc::sigaltstack(ptr::null(), &mut current) == 0 && current.ss_flags == 0 {
            return; // this thread already has one (e.g. installed by Rust std)
        }
        let size = libc::SIGSTKSZ;
        let stack = libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if stack == libc::MAP_FAILED {
            return; // degraded: faults still deliver on the current stack
        }
        let ss = libc::stack_t {
            ss_sp: stack,
            ss_size: size,
            ss_flags: 0,
        };
        libc::sigaltstack(&ss, ptr::null_mut());
    }
}

/// Terminate the process with `signo`'s default action, bypassing the fault
/// handler. The run loop uses this to reflect a wild guest branch as a
/// faithful signal death without triggering the crash report — the report's
/// register state would be Chimera's run loop, not the guest's.
pub fn die(signo: libc::c_int) -> ! {
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0;
        libc::sigaction(signo, &sa, ptr::null_mut());
        let mut set: libc::sigset_t = mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, signo);
        libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, ptr::null_mut());
        libc::raise(signo);
        // `raise` cannot return once the disposition is default-terminate, but
        // the signature demands divergence.
        libc::_exit(128 + signo);
    }
}

extern "C" fn chimera_fault(
    signo: libc::c_int,
    info: *const libc::siginfo_t,
    ucontext: *mut libc::c_void,
) {
    let fault_addr = unsafe { (*info).si_addr } as usize;
    let si_code = unsafe { (*info).si_code };
    let pc = fault_pc(ucontext);

    // A `kill()`/`raise()`-raised SIGSEGV or SIGBUS cannot be told apart from a
    // hardware fault here: unlike Linux (`si_code <= 0`), Darwin fills in
    // `si_code = SEGV_ACCERR` with `si_addr = 0` for a user-sent SIGSEGV, the
    // same shape as a genuine null dereference. Chimera's own signal-death path
    // therefore never routes through this handler — [`die`] resets the default
    // disposition before raising — and a guest-sent fatal signal simply gets
    // the crash report too.

    // A write to a page armed for self-modifying-code detection, from
    // anywhere. Usually that is the guest rewriting its own code from
    // translated code, but it can equally be Chimera's own Rust: the runtime
    // and the guest share one libSystem, and a 16 KiB page holding guest code
    // the translator armed can also hold data the runtime writes. Servicing
    // only faults taken inside the cache would leave those stranded — the
    // store would re-fault forever with nobody to restore write permission.
    //
    // A fault in translated code may block for the lock: this thread cannot
    // be holding it (it is dropped before `dispatch`), and a sibling holding
    // it — arming pages mid-translation, say — is exactly who must finish
    // first. A fault in Chimera's own code gets `try_lock` instead: that one
    // could be a fault taken *while holding* the lock, where blocking would
    // deadlock against this very thread. Losing that race falls through to
    // the crash report, the honest outcome for a runtime fault.
    let process = PROCESS.load(Ordering::Acquire);
    if !process.is_null() {
        let process = unsafe { &*process };
        let serviced = if crate::process::addr_space_held() {
            // This thread is the holder, so blocking would deadlock against
            // itself. Try, and fall through to the report if the guard is a
            // live one rather than a stale depth.
            process
                .addr_space
                .try_lock()
                .is_ok_and(|mut space| space.on_smc_write(fault_addr))
        } else {
            // A sibling holds it and will release it; wait. Deciding this
            // from the faulting pc instead — "in the code cache means a
            // sibling holds it" — was wrong in the direction that matters:
            // a fault in the runtime's own code (or in the shared libSystem,
            // running natively) on an armed page is perfectly serviceable,
            // and losing a try_lock race against a sibling mid-translation
            // turned it into a fatal crash. Under a many-threaded guest that
            // race is common, which is what made rustc's builds flaky.
            process.addr_space.lock().unwrap().on_smc_write(fault_addr)
        };
        if serviced {
            // A serviced fault re-executes; a genuine SMC service or race
            // retry succeeds on the next try, so the same address coming back
            // thousands of times is not a race — it is an unserviceable fault
            // being retried forever (a livelock, not a crash, which is
            // strictly worse). Give up and report it.
            static RETRY_ADDR: AtomicUsize = AtomicUsize::new(0);
            static RETRY_COUNT: AtomicUsize = AtomicUsize::new(0);
            let repeated = RETRY_ADDR.swap(fault_addr, Ordering::Relaxed) == fault_addr;
            let count = if repeated {
                RETRY_COUNT.fetch_add(1, Ordering::Relaxed) + 1
            } else {
                RETRY_COUNT.store(0, Ordering::Relaxed);
                0
            };
            if count < 10_000 {
                return; // write permission restored; re-execute the store
            }
        }
    }

    // A genuine guest fault for which the guest installed a handler: deliver it
    // into the guest rather than terminating. Redirects the interrupted context
    // back to the run loop, which runs the handler; returns true if it took.
    if code_cache_contains(pc) && try_deliver_guest_fault(signo, si_code, fault_addr, pc, ucontext)
    {
        return;
    }

    // Not a delivered fault: no guest handler, a fault in Chimera itself, or a
    // fault where the guest state could not be recovered. Print a crash report,
    // then restore the default disposition and return; the faulting instruction
    // re-executes and the kernel terminates the process with the real signal.
    report_fault(signo, si_code, fault_addr, pc, ucontext);
    unsafe {
        let mut sa: libc::sigaction = mem::zeroed();
        sa.sa_sigaction = libc::SIG_DFL;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0;
        libc::sigaction(signo, &sa, ptr::null_mut());
    }
}

/// Deliver a synchronous guest fault into the guest's handler, if it has one.
///
/// A fault in translated code interrupts a block mid-flight, on the alternate
/// signal stack — not at a run-loop boundary where signal delivery normally
/// happens. Rather than build the guest frame here (which would need the
/// thread's `Signals`, unreachable from an async handler), transplant the
/// interrupted state back into the run loop: the signal's mcontext *is* the
/// guest register file (translated code keeps guest registers in host
/// registers), so copy it into `ctx`, map the faulting host PC to its guest PC,
/// record the fault as a pending signal, and rewrite the interrupted context to
/// resume at [`fault_resume`] with x17 = ctx. When the handler returns, the
/// kernel resumes there, unwinds to the run loop, and the pending fault is
/// delivered through the ordinary path — building the frame, running the
/// handler, and on its return re-executing the faulting instruction (correct
/// `SIGSEGV` semantics: a handler that does not fix the fault re-faults).
///
/// Returns false — leaving the caller to crash-report and terminate — if the
/// guest has no handler, the mcontext or `ctx` is unavailable, or the faulting
/// PC does not map to a guest instruction.
fn try_deliver_guest_fault(
    signo: libc::c_int,
    si_code: libc::c_int,
    fault_addr: usize,
    pc: usize,
    ucontext: *mut libc::c_void,
) -> bool {
    let process = PROCESS.load(Ordering::Acquire);
    let ctx = guest_ctx() as *mut crate::arch::dispatch::ThreadState;
    let mc = mcontext(ucontext) as *mut MContext64;
    if process.is_null() || ctx.is_null() || mc.is_null() {
        return false;
    }
    let process = unsafe { &*process };
    if !guest_handles(&process.sig_table, signo as u32) {
        return false;
    }
    // Map the faulting host PC to the guest instruction; without it the guest
    // frame's PC would be meaningless, so decline and let the crash report run.
    let Some(guest_pc) = process
        .addr_space
        .lock()
        .unwrap()
        .code
        .guest_pc_at(pc as u64)
    else {
        return false;
    };

    // Say where the guest faulted, once, before handing the fault over. A
    // handler is free to swallow the detail — Rust's SIGSEGV handler decides
    // the fault is not a stack overflow, restores the default disposition and
    // returns, so the process dies at the sigreturn boundary with nothing
    // said about the original address. That leaves the only evidence of a
    // real bug inside the guest's own control flow.
    {
        let mut r = Report::new();
        r.str(b"chimera: guest fault ");
        r.str(signal_name(signo));
        r.str(b" at guest pc ");
        r.hex(guest_pc);
        r.str(b", address ");
        r.hex(fault_addr as u64);
        r.str(b" -> delivering to the guest's handler\n");
        r.flush();
    }

    unsafe {
        let m = &*mc;
        let c = &mut *ctx;
        // The mcontext is the live guest register file — copy it into ctx so
        // the run loop resumes exactly where the fault interrupted.
        c.regs[..29].copy_from_slice(&m.x);
        c.regs[29] = m.fp;
        c.regs[30] = m.lr;
        c.sp = m.sp;
        c.pc = guest_pc;
        c.nzcv = m.cpsr as u64;
        c.fpstate = m.ns;

        // Record the fault as pending, carrying its address to the handler's
        // siginfo. `si_addr` for a fault is the guest fault address as-is.
        let mut info: libc::siginfo_t = mem::zeroed();
        info.si_signo = signo;
        info.si_code = si_code;
        info.si_addr = fault_addr as *mut libc::c_void;
        record_fault(c.pending_set as *const _, signo as u32, &info);

        // Rewrite the interrupted context: on signal return, resume at
        // `fault_resume` with x17 = ctx, which unwinds to the run loop.
        let m = &mut *mc;
        m.x[17] = ctx as u64;
        m.pc = fault_resume as *const () as usize as u64;
    }
    true
}

/// Read the faulting program counter from the signal's `ucontext`.
fn fault_pc(ucontext: *mut libc::c_void) -> usize {
    if ucontext.is_null() {
        return 0;
    }
    let uc = ucontext as *const UContext;
    unsafe {
        let mc = (*uc).uc_mcontext;
        if mc.is_null() { 0 } else { (*mc).pc as usize }
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

    /// A NUL-terminated C string, bounded.
    fn cstr(&mut self, s: *const libc::c_char) {
        for i in 0..256 {
            let b = unsafe { *s.add(i) as u8 };
            if b == 0 {
                break;
            }
            self.str(&[b]);
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

/// Print a kernel-OOPS-style crash report for a fatal guest fault: the signal
/// and its cause, the faulting addresses, the full guest register file
/// (translated code keeps the guest registers in the host registers, so the
/// `ucontext` *is* the guest state), the words of the faulting translated
/// instruction, and a slice of the guest stack. Allocation-free and
/// best-effort: the register dump comes first so it survives even if a later
/// memory read faults.
fn report_fault(
    signo: libc::c_int,
    si_code: libc::c_int,
    fault_addr: usize,
    pc: usize,
    ucontext: *mut libc::c_void,
) {
    let in_cache = code_cache_contains(pc);
    let mut r = Report::new();

    r.str(b"\nchimera: guest received fatal ");
    r.str(signal_name(signo));
    r.str(b"\n");
    r.str(b"  cause:   ");
    r.str(fault_cause(signo, si_code));
    r.str(b"\n  address: ");
    r.hex(fault_addr as u64);
    r.str(b"\n  host pc: ");
    r.hex(pc as u64);
    r.str(if in_cache {
        b"  (translated code)\n"
    } else {
        b"  (chimera runtime)\n"
    });
    // The faulting guest PC, mapped from the host PC through the block's
    // pc-map. `try_lock`: a crash report must never block, whoever holds the
    // lock and whatever led here — the fallback below (the guest PC published
    // at the last block boundary, from the faulting thread's ThreadState TSD
    // slot) is close enough to be worth far more than a hang.
    if in_cache {
        let mut precise = None;
        let process = PROCESS.load(Ordering::Acquire);
        if !process.is_null() {
            let process = unsafe { &*process };
            precise = process
                .addr_space
                .try_lock()
                .ok()
                .and_then(|space| space.code.guest_pc_at(pc as u64));
        }
        if let Some(guest_pc) = precise {
            r.str(b"  guest pc: ");
            r.hex(guest_pc);
            r.str(b"  (faulting instruction)\n");
        } else {
            let ctx = guest_ctx();
            if !ctx.is_null() {
                r.str(b"  guest pc: ");
                r.hex(unsafe { (*ctx).pc });
                r.str(b"  (last block boundary)\n");
            }
        }
    }

    // What the fault address is: the recorded guest region (if any), the
    // page's SMC state, and the kernel's live protection — the difference
    // between "a page that should be readable isn't" and "a wild pointer".
    {
        let process = PROCESS.load(Ordering::Acquire);
        if !process.is_null() {
            match unsafe { &*process }.addr_space.try_lock() {
                Ok(space) => {
                    let (region, armed, granted) = space.describe_addr(fault_addr);
                    r.str(b"  fault addr: ");
                    match region {
                        Some((start, len, runtime_owned)) => {
                            r.str(b"in guest region ");
                            r.hex(start as u64);
                            r.str(b"+");
                            r.hex(len as u64);
                            r.str(if runtime_owned {
                                b" (runtime-owned)"
                            } else {
                                b" (guest-owned)"
                            });
                        }
                        None => r.str(b"not in a recorded guest region"),
                    }
                    if armed {
                        r.str(b", page armed");
                    }
                    if granted {
                        r.str(b", page granted");
                    }
                    r.str(b"\n");
                }
                Err(_) => r.str(
                    b"  fault addr: (address-space lock busy: a sibling was mid-translation)\n",
                ),
            }
        }
        // Distinguish the two things a zero protection can mean, since they
        // lead opposite ways: a mapped PROT_NONE page (a guard page — the
        // fault is a stack overflow or a walk off the end of an allocation)
        // versus no mapping at all (something was unmapped that should not
        // have been).
        let (mapped, prot, max_prot) = region_protection(fault_addr as u64);
        r.str(b"  fault page:  ");
        if !mapped {
            r.str(b"NOT MAPPED (nothing there at all)\n");
        } else {
            r.str(b"prot=");
            r.hex(prot as u64);
            r.str(b" maxprot=");
            r.hex(max_prot as u64);
            r.str(if prot == 0 {
                b"  MAPPED PROT_NONE (a guard page)\n"
            } else {
                b"  (r=1 w=2 x=4)\n"
            });
        }
    }

    r.str(b"\nguest registers:\n");
    let mc = mcontext(ucontext);
    if !mc.is_null() {
        let mc = unsafe { &*mc };
        for (i, x) in mc.x.iter().enumerate() {
            let name = if i < 10 {
                [b'x', b'0' + i as u8, b' ']
            } else {
                [b'x', b'0' + (i / 10) as u8, b'0' + (i % 10) as u8]
            };
            r.reg(&name, *x);
            if i % 4 == 3 {
                r.str(b"\n");
            }
        }
        r.reg(b"fp ", mc.fp);
        r.str(b"\n");
        r.reg(b"lr ", mc.lr);
        r.reg(b"sp ", mc.sp);
        r.reg(b"pc ", mc.pc);
        r.reg(b"cpsr", mc.cpsr as u64);
        r.str(b"\n");
    }
    r.flush();

    // Riskier reads (they dereference guest memory) go last, after the registers
    // are already on the terminal. The faulting instruction lives in the code
    // cache, which is mapped and readable, so dumping it is safe.
    if in_cache {
        r.str(b"\ncode (words at host pc):\n ");
        for off in 0..4usize {
            let word = unsafe { ptr::read_volatile((pc + off * 4) as *const u32) };
            r.str(b" ");
            r.hex(word as u64);
        }
        r.str(b"\n");
        r.flush();
    }

    // The guest stack: walk upward from sp (away from any guard page below it),
    // so this read stays within the live stack even for a stack-overflow fault.
    if !mc.is_null() {
        let sp = unsafe { (*mc).sp };
        r.str(b"\nstack (from sp):\n");
        for i in 0..8u64 {
            let addr = sp + i * 16;
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

    // A native fault: walk the frame-pointer chain and name each return
    // address with `dladdr`, which knows every image the host dyld loaded —
    // Chimera itself, the shared cache, and any dylib the in-process linker
    // pulled in natively. Guest-image addresses come back unnamed, which is
    // itself the tell: a bare address here is guest code on a native path.
    if !in_cache && !mc.is_null() {
        r.str(b"\nnative frames (fp chain):\n");
        let mut lr = unsafe { (*mc).pc };
        let mut fp = unsafe { (*mc).fp };
        for _ in 0..16 {
            r.str(b"  ");
            r.hex(lr);
            let mut info: libc::Dl_info = unsafe { core::mem::zeroed() };
            if unsafe { libc::dladdr(lr as *const libc::c_void, &mut info) } != 0 {
                if !info.dli_sname.is_null() {
                    r.str(b"  ");
                    r.cstr(info.dli_sname);
                    r.str(b" + ");
                    r.hex(lr - info.dli_saddr as u64);
                } else if !info.dli_fname.is_null() {
                    r.str(b"  in ");
                    r.cstr(info.dli_fname);
                }
            }
            r.str(b"\n");
            if fp == 0 || fp % 8 != 0 {
                break;
            }
            let frame = unsafe {
                [
                    ptr::read_volatile(fp as *const u64),
                    ptr::read_volatile((fp + 8) as *const u64),
                ]
            };
            if frame[0] <= fp || frame[1] == 0 {
                break;
            }
            fp = frame[0];
            lr = frame[1];
        }
        r.flush();
    }
    r.str(b"\n");
    r.flush();
}

/// The kernel's live protection for the region containing `addr`, via
/// `mach_vm_region`: `(mapped, prot, max_prot)`. `mapped` is false when
/// nothing covers the address — `mach_vm_region` reports the *next* region
/// above an unmapped one, so the returned range must be checked rather than
/// trusted.
fn region_protection(addr: u64) -> (bool, i32, i32) {
    const VM_REGION_BASIC_INFO_64: i32 = 9;
    const VM_REGION_BASIC_INFO_COUNT_64: u32 = 9;
    #[repr(C)]
    struct BasicInfo64 {
        protection: i32,
        max_protection: i32,
        inheritance: u32,
        shared: u32,
        reserved: u32,
        offset: u64,
        behavior: i32,
        user_wired_count: u16,
    }
    unsafe extern "C" {
        fn mach_vm_region(
            target: libc::c_uint,
            address: *mut u64,
            size: *mut u64,
            flavor: i32,
            info: *mut BasicInfo64,
            info_count: *mut u32,
            object_name: *mut libc::c_uint,
        ) -> libc::c_int;
        static mut mach_task_self_: libc::c_uint;
    }
    let mut address = addr;
    let mut size = 0u64;
    let mut info: BasicInfo64 = unsafe { mem::zeroed() };
    let mut count = VM_REGION_BASIC_INFO_COUNT_64;
    let mut object = 0;
    let kr = unsafe {
        mach_vm_region(
            mach_task_self_,
            &mut address,
            &mut size,
            VM_REGION_BASIC_INFO_64,
            &mut info,
            &mut count,
            &mut object,
        )
    };
    // mach_vm_region rounds up to the NEXT region when `addr` is unmapped;
    // only report when the region actually covers the address.
    if kr == 0 && address <= addr && addr < address + size {
        (true, info.protection, info.max_protection)
    } else {
        (false, 0, 0)
    }
}

fn mcontext(ucontext: *mut libc::c_void) -> *const MContext64 {
    if ucontext.is_null() {
        return ptr::null();
    }
    unsafe { (*(ucontext as *const UContext)).uc_mcontext }
}

/// The faulting thread's `ThreadState`, from the reserved ctx TSD slot — null
/// on a thread that never ran guest code. A single `mrs` plus a load, so it is
/// async-signal-safe (unlike `pthread_getspecific`).
fn guest_ctx() -> *const crate::arch::dispatch::ThreadState {
    crate::arch::dispatch::current_ctx()
}

/// The mnemonic for a fatal signal number.
fn signal_name(signo: libc::c_int) -> &'static [u8] {
    match signo {
        libc::SIGSEGV => b"SIGSEGV (segmentation fault)",
        libc::SIGBUS => b"SIGBUS (bus error)",
        libc::SIGTRAP => b"SIGTRAP (trace trap)",
        _ => b"signal",
    }
}

/// A human-readable description of the fault's `si_code` — for SIGSEGV this
/// distinguishes a write to read-only memory (`SEGV_ACCERR`, the shape of a
/// self-modifying-code or guard-page hit) from a dereference of unmapped memory
/// (`SEGV_MAPERR`, a wild pointer).
fn fault_cause(signo: libc::c_int, si_code: libc::c_int) -> &'static [u8] {
    // `si_code` values from `<sys/signal.h>`.
    const SEGV_MAPERR: libc::c_int = 1;
    const SEGV_ACCERR: libc::c_int = 2;
    const BUS_ADRALN: libc::c_int = 1;
    const BUS_ADRERR: libc::c_int = 2;
    const BUS_OBJERR: libc::c_int = 3;
    const TRAP_BRKPT: libc::c_int = 1;
    match (signo, si_code) {
        (libc::SIGTRAP, TRAP_BRKPT) => b"breakpoint or trap instruction",
        (libc::SIGTRAP, _) => b"trace trap",
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
