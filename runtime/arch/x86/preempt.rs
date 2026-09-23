//! Asynchronous preemption of translated code.
//!
//! A guest signal, or a sibling's process-wide stop, is acted on by the run
//! loop — the one place where the guest register file is canonical in
//! `ThreadState` and signal delivery can build a frame. A thread executing
//! translated code has no reason to return there on its own: linked blocks
//! and the inline indirect-branch lookup keep a hot loop in the cache
//! indefinitely. [`preempt`] is what the host signal catchers call to end that
//! residency immediately. It runs inside the catcher, on the interrupted
//! thread, and rewrites the interrupted `ucontext` so that `sigreturn` lands in
//! `exit_now` instead of back in the cache: the thread leaves through the
//! ordinary block-exit trampoline, which saves the live registers, and the
//! run loop finds a precise guest state — the registers and PC of the very
//! instruction boundary the signal interrupted — exactly as it would after a
//! block boundary.
//!
//! Recovering that state from an arbitrary host rip is the whole problem. In
//! the body of a block the host registers *are* the guest registers (the body
//! is a verbatim copy) and only the guest PC is missing, which the block's
//! [`Entry`] list supplies. Everywhere else — the lazy-install prologue, the
//! terminators and their exit stubs, the shared lookup routine, `dispatch`
//! itself — the translator has borrowed registers through `ThreadState` slots,
//! and each such span carries a [`recipe`] naming what to put back and where
//! the guest PC is. A span already bound for the dispatcher (an exit stub, a
//! syscall sequence) needs nothing at all.
//!
//! Everything here is async-signal-safe and runs under whatever FS base the
//! guest left installed: plain loads and stores, a lock-free binary search,
//! `gs:`-relative access to the thread's own `ThreadState`. No allocation, no
//! locks, no TLS.

use std::mem::offset_of;

use libc::{
    REG_EFL, REG_R8, REG_R9, REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15, REG_RAX,
    REG_RBP, REG_RBX, REG_RCX, REG_RDI, REG_RDX, REG_RIP, REG_RSI, REG_RSP,
};

use super::{
    dispatch::ThreadState,
    trampoline,
    translate::{self, BlockMeta, Entry, recipe},
};

/// `REG_*` index of each `ThreadState::regs` slot (rax, rbx, rcx, rdx, rsi,
/// rdi, rbp, rsp, r8..r15), for the recipes that put a parked register back.
const GREG_OF_STATE_REG: [i32; 16] = [
    REG_RAX, REG_RBX, REG_RCX, REG_RDX, REG_RSI, REG_RDI, REG_RBP, REG_RSP, REG_R8, REG_R9,
    REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15,
];

/// The `lahf` status-flag bits (SF, ZF, AF, PF, CF) and OF, the flags the
/// translator parks in `lahf`/`seto` form.
const LAHF_MASK: u64 = 0xd5;
const OF: u64 = 1 << 11;

/// Force the interrupted thread back to its run loop. Called from a host
/// signal catcher with the interrupted `ucontext_t`. Returns `true` when the
/// thread will reach the run loop without further help — it was redirected
/// out of the cache, or was already on its way there — and `false` when it was
/// interrupted in Chimera's own code, where the caller must arm
/// `ThreadState::exit_requested` so the next cache entry bows out instead
/// (see `dispatch` in `trampoline.S`): the run loop may already be past its
/// pending-signal check and about to enter a linked loop that would otherwise
/// never come back.
pub fn preempt(ucontext: *mut libc::c_void) -> bool {
    let uc = unsafe { &mut *(ucontext as *mut libc::ucontext_t) };
    let gregs = &mut uc.uc_mcontext.gregs;
    let rip = gregs[REG_RIP as usize] as usize;

    // Inside `dispatch`: the guest state is still canonical in `ThreadState`
    // (the entry only reads it) and the block about to be entered is named by
    // rsi until it is stashed, by `host_pc_target` after. Re-aim that jump at
    // `exit_now`, which runs the loaded registers straight back out.
    let span = trampoline::dispatch_span();
    if (span.lo..span.hi).contains(&rip) {
        let exit_now = trampoline::exit_now_addr() as u64;
        if rip < span.stashed {
            gregs[REG_RSI as usize] = exit_now as i64;
        } else {
            gs_store(offset_of!(ThreadState, host_pc_target), exit_now);
        }
        return true;
    }

    if !translate::code_cache_contains(rip) {
        return false;
    }

    // The shared inline indirect-branch lookup: the guest has taken the branch,
    // so it resumes at the target. Before the routine has stashed its scratch
    // slots the target is in rax (first instruction) or its slot and the other
    // registers are live; from the stash boundary on, every slot the routine's
    // own miss path restores from is valid, so that path is the redirect —
    // it rebuilds the registers and exits with the target as the next PC.
    if let Some(ib) = translate::ib_lookup_span()
        && (ib.lo..ib.hi).contains(&rip)
    {
        if rip < ib.stashed {
            let target = if rip == ib.lo {
                gregs[REG_RAX as usize] as u64
            } else {
                gs_load(offset_of!(ThreadState, ib_target))
            };
            gregs[REG_RAX as usize] = gs_load(rax_slot()) as i64;
            redirect(gregs, target);
        } else {
            gregs[REG_RIP as usize] = ib.miss as i64;
            fix_fs();
        }
        return true;
    }

    let Some((meta, entries, off)) = translate::lookup_block(rip) else {
        // Every reachable cache byte belongs to a published block or the
        // lookup routine; there is nothing to recover from and nothing to do.
        debug_assert!(false, "preempted at unindexed cache rip {rip:#x}");
        return true;
    };
    let Some((entry, guest_pc, at_boundary)) = translate::walk_entries(entries, off, meta.guest_pc)
    else {
        debug_assert!(
            false,
            "preempted past the entries of block {:#x}",
            meta.guest_pc
        );
        return true;
    };
    apply(gregs, meta, entry, guest_pc, at_boundary)
}

/// Rewrite the interrupted registers per `entry`'s recipe and aim the thread
/// at `exit_now`. Returns `false` only for a span that reaches the dispatcher
/// by itself.
fn apply(
    gregs: &mut [i64; 23],
    meta: &BlockMeta,
    entry: Entry,
    guest_pc: u64,
    at_boundary: bool,
) -> bool {
    if entry.fix != 0 {
        let idx = (entry.fix - 1) as usize;
        gregs[GREG_OF_STATE_REG[idx] as usize] =
            gs_load(offset_of!(ThreadState, riprel_scratch)) as i64;
    }
    let code = entry.code;
    if code < recipe::PRO {
        // A body instruction is one entry; the signal can only have landed on
        // its boundary.
        debug_assert!(at_boundary, "preempted inside a body instruction");
        redirect(gregs, guest_pc);
        return true;
    }
    if code & 0xc0 == recipe::PRO {
        if code & recipe::PRO_RAX != 0 {
            gregs[REG_RAX as usize] = gs_load(rax_slot()) as i64;
        }
        if code & recipe::PRO_RDX != 0 {
            gregs[REG_RDX as usize] = gs_load(offset_of!(ThreadState, fp_scratch)) as i64;
        }
        if code & recipe::PRO_FLAGS != 0 {
            let parked = gs_load(offset_of!(ThreadState, fp_flags));
            let efl = gregs[REG_EFL as usize] as u64;
            let status = (parked >> 8) & LAHF_MASK;
            let of = if parked & 1 != 0 { OF } else { 0 };
            gregs[REG_EFL as usize] = ((efl & !(LAHF_MASK | OF)) | status | of) as i64;
        }
        redirect(gregs, meta.guest_pc);
        return true;
    }
    match code {
        recipe::FLOW => true,
        recipe::PRECISE_T => {
            redirect(gregs, meta.term_ip);
            true
        }
        recipe::PRECISE_TAKEN => {
            redirect(gregs, meta.taken);
            true
        }
        recipe::PRECISE_FALL => {
            redirect(gregs, meta.fall);
            true
        }
        recipe::RAXSLOT_T => {
            gregs[REG_RAX as usize] = gs_load(rax_slot()) as i64;
            redirect(gregs, meta.term_ip);
            true
        }
        recipe::RAXSLOT_TAKEN => {
            gregs[REG_RAX as usize] = gs_load(rax_slot()) as i64;
            redirect(gregs, meta.taken);
            true
        }
        recipe::RAXSLOT_RIP_RBXSLOT => {
            let target = gs_load(offset_of!(ThreadState, regs) + 8);
            gregs[REG_RAX as usize] = gs_load(rax_slot()) as i64;
            redirect(gregs, target);
            true
        }
        recipe::RAXSLOT_RIP_RAX | recipe::RAXSLOT_RIP_RAX_RSPADJ => {
            let target = gregs[REG_RAX as usize] as u64;
            gregs[REG_RAX as usize] = gs_load(rax_slot()) as i64;
            if code == recipe::RAXSLOT_RIP_RAX_RSPADJ {
                gregs[REG_RSP as usize] =
                    (gregs[REG_RSP as usize] as u64).wrapping_add(meta.rsp_adj as u64) as i64;
            }
            redirect(gregs, target);
            true
        }
        _ => {
            debug_assert!(false, "unknown preemption recipe {code:#x}");
            true
        }
    }
}

/// Publish `guest_pc` as the PC to resume at and aim the thread at
/// `exit_now`, whose `exit_block` tail saves the (now guest-precise) live
/// registers into `ThreadState` and returns to the run loop.
fn redirect(gregs: &mut [i64; 23], guest_pc: u64) {
    gs_store(offset_of!(ThreadState, rip), guest_pc);
    gregs[REG_RIP as usize] = trampoline::exit_now_addr() as i64;
    fix_fs();
}

/// Make the exit trampoline's FS restore match reality. A block prologue
/// installs the guest FS base with `wrfsbase` and only then sets
/// `fs_is_guest`; a thread caught between the two would exit with the guest's
/// base still in FS and the flag clear, and the run loop's Rust would run its
/// TLS against guest memory. The base itself is the truth: if it is not
/// Chimera's, the exit must restore Chimera's.
fn fix_fs() {
    let current: u64;
    unsafe {
        std::arch::asm!("rdfsbase {x}", x = out(reg) current, options(nomem, nostack, preserves_flags));
    }
    if current != gs_load(offset_of!(ThreadState, chimera_fs_base)) {
        gs_store(offset_of!(ThreadState, fs_is_guest), 1);
    }
}

fn rax_slot() -> usize {
    offset_of!(ThreadState, regs)
}

/// Read the qword at `off` in the calling thread's `ThreadState`, through GS.
fn gs_load(off: usize) -> u64 {
    let v: u64;
    unsafe {
        std::arch::asm!(
            "mov {v}, gs:[{off}]",
            v = out(reg) v,
            off = in(reg) off,
            options(nostack, readonly, preserves_flags),
        );
    }
    v
}

/// Write the qword at `off` in the calling thread's `ThreadState`, through GS.
fn gs_store(off: usize, v: u64) {
    unsafe {
        std::arch::asm!(
            "mov gs:[{off}], {v}",
            v = in(reg) v,
            off = in(reg) off,
            options(nostack, preserves_flags),
        );
    }
}
