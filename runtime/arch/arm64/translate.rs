//! Basic-block translator for AArch64. Decodes a single guest basic block,
//! copies the straight-line prefix into the code cache (with PC-relative
//! operands — ADR, ADRP, LDR-literal — fixed up to refer to the original
//! guest target), and rewrites the terminator into a sequence that:
//!
//! 1. saves the guest's x16 and x17 by pushing them onto the guest stack
//!    (these are the only GPRs the exit trampoline tail won't see in their
//!    original registers — everything else is loaded directly from live
//!    registers),
//! 2. computes the next guest PC into `ctx.pc`, performing any side effects
//!    the original terminator mandated (e.g., BL writing `next_ip` to x30),
//! 3. branches to the appropriate exit trampoline (`exit_block` for ordinary
//!    block boundaries, `exit_syscall` for SVC).
//!
//! Per-block prologue: every emitted block starts with a load that re-syncs
//! x16 from `ctx.regs[16]`, because `dispatch` and every exit stub clobber x16
//! to carry their branch target. No GPR is reserved across translated code; the
//! context pointer is reached per-thread through the host thread's pthread TSD
//! slot, read inline via `TPIDRRO_EL0` (see [`emit_load_ctx`]).

use std::ptr;

use crate::Error;

use super::cache::{CodeCache, IB_BITS, IB_HASH_MULT};
use super::dispatch::{CHIMERA_CTX_TSD_SLOT, ThreadState};

const MAX_BLOCK_GUEST_INSNS: usize = 1024;

/// Guest page size on Apple Silicon; the granularity at which the translator
/// probes guest code for mappedness.
const GUEST_PAGE: u64 = 16 * 1024;

/// Whether the guest page at `page` is mapped. `mincore` is the cheapest
/// question the kernel answers here, and — unlike the Mach-based
/// `copy_from_guest` — it is a plain BSD call with no task port behind it, so
/// it stays correct in a fork child whose cached Mach state has not been
/// repaired yet (the translator runs before the guest's atfork handlers do).
fn page_mapped(page: u64) -> bool {
    let mut resident = 0u8;
    unsafe {
        libc::mincore(
            page as *mut libc::c_void,
            GUEST_PAGE as usize,
            &mut resident as *mut u8 as *mut libc::c_char,
        ) == 0
    }
}

/// Emit the sequence that loads this host thread's `ThreadState` pointer into
/// `reg` from its reserved pthread TSD slot, reached through `TPIDRRO_EL0`.
/// Per-thread and signal-safe — unlike a reserved register (libSystem and the
/// kernel clobber `x18`) or a process-wide global (concurrent guest threads
/// race on it). The guest cannot interfere: `TPIDRRO_EL0` is read-only from EL0
/// and `thread_set_tsd_base` is intercepted, and the slot is one libpthread
/// reserves and never touches (see [`CHIMERA_CTX_TSD_SLOT`]). The slot offset
/// (`slot * 8`) is a compile-time constant baked into the load.
fn emit_load_ctx(out: &mut Vec<u32>, reg: u8) {
    out.push(enc_mrs_tpidrro(reg)); // reg = TPIDRRO_EL0
    out.push(enc_and_not7(reg, reg)); // reg = TSD base (mask the CPU-number bits)
    out.push(enc_ldr_imm(reg, reg, (CHIMERA_CTX_TSD_SLOT * 8) as u32)); // reg = tsd[slot] = ctx
}

/// `mrs <reg>, TPIDRRO_EL0`.
fn enc_mrs_tpidrro(rt: u8) -> u32 {
    0xD53BD060 | (rt as u32)
}

/// `and <rd>, <rn>, #0xFFFF_FFFF_FFFF_FFF8` — clear the low 3 bits.
fn enc_and_not7(rd: u8, rn: u8) -> u32 {
    0x927DF000 | (rd as u32) | ((rn as u32) << 5)
}

/// The result of translating one guest basic block: where its host code
/// begins and ends, the guest address just past the terminator (so the caller
/// can arm the page(s) the block covers for self-modifying-code detection),
/// and the host-offset → guest-PC map for fault attribution.
pub struct Translation {
    pub host_pc: u64,
    pub host_end: u64,
    pub guest_end: u64,
    /// Entry point for a *linked* predecessor, a few words below `host_pc`:
    /// it pops the guest x16/x17 the predecessor's stub pushed and falls into
    /// the block body, skipping the ctx-based prologue `host_pc` starts with.
    /// This is the address a link slot holds.
    pub linked_pc: u64,
    /// One entry per guest instruction: the block-relative host word offset
    /// where its expansion begins, paired with its guest PC. Offsets ascend,
    /// so a faulting host PC maps to a guest PC by binary search; host words
    /// before the first entry (the block prologue) belong to the block start.
    pub pc_map: Vec<(u32, u64)>,
    /// This block's outgoing direct edges: for each, the guest PC it targets
    /// and the address of the link slot its exit stub reads. The cache fills
    /// the slot with the target block's `linked_pc` once that block exists.
    pub edges: Vec<OutEdge>,
}

/// One statically known outgoing edge of a translated block: the guest PC it
/// branches to, and the link slot the stub loads its host destination from.
pub struct OutEdge {
    pub target_guest: u64,
    pub slot: u64,
}

/// Translate one basic block starting at `guest_pc`: copy the straight-line
/// prefix into the code cache (fixing up ADR/ADRP/LDR-literal to their absolute
/// guest targets) and rewrite the terminator into an exit stub that computes
/// the next guest PC into `ctx.pc` and branches to `exit_tramp` (an ordinary
/// boundary), `syscall_tramp` (an SVC), or `trap_exit` (a `BRK`).
pub fn translate(
    cache: &mut CodeCache,
    guest_pc: u64,
    exit_tramp: u64,
    syscall_tramp: u64,
    trap_exit: u64,
) -> Result<Translation, Error> {
    let ib_table = cache.ib_table_addr();
    let mut out: Vec<u32> = Vec::new();

    // Linked entry (the address a predecessor's link slot holds): a linked
    // predecessor left the guest's x16/x17 on the guest stack rather than in
    // ctx, so pop them and skip the ctx-based prologue below. Two words, so
    // the dispatcher's entry point is `host_pc = block start + 8`.
    out.push(enc_ldp_post_index(16, 17, 31, 16));
    out.push(enc_b_imm(5)); // over the 4-word prologue, into the body

    // Block prologue: re-sync x16 from ctx.regs[16]. An entry from `dispatch`
    // or from an exit trampoline arrives with x16 clobbered. Load this
    // thread's ctx from its TSD slot, then read regs[16].
    emit_load_ctx(&mut out, 16);
    out.push(enc_ldr_imm(16, 16, regs_off(16))); // x16 = ctx.regs[16]

    let mut pc = guest_pc;
    let mut count = 0;
    let mut pc_map: Vec<(u32, u64)> = Vec::new();
    let mut edges: Vec<OutEdge> = Vec::new();
    // Reserve a link slot for a statically known edge, recording it for the
    // cache to fill in. `None` when the arena is exhausted, which leaves the
    // edge unlinked — correct, just slower.
    let no_link = crate::trace::no_link();
    macro_rules! link {
        ($target:expr) => {{
            let slot = if no_link {
                None
            } else {
                cache.alloc_link_slot()
            };
            if let Some(slot) = slot {
                edges.push(OutEdge {
                    target_guest: $target,
                    slot,
                });
            }
            slot
        }};
    }
    // The same, packaged with the edge's poll decision, for the two-way
    // terminators whose arms each carry their own exit tail.
    macro_rules! edge {
        ($target:expr, $from:expr) => {
            Edge {
                slot: link!($target),
                poll: is_back_edge($from, $target),
            }
        };
    }
    // Both arms of a two-way terminator at `from`.
    macro_rules! ways {
        ($target:expr, $fallthrough:expr, $from:expr) => {
            TwoWay {
                target: $target,
                fallthrough: $fallthrough,
                taken: edge!($target, $from),
                not_taken: edge!($fallthrough, $from),
            }
        };
    }
    // Guest code is fetched with a plain load, one per instruction, so the
    // page it lives on must be known mapped first — a fault here would land
    // inside the translator, which holds the address-space lock and so could
    // not even be reported cleanly. Probe once per page (at entry, and again
    // when decoding crosses into the next one) and report an unmapped one as
    // `BadAccess`, which the run loop reflects as the guest SIGSEGV the
    // branch would have taken natively.
    let mut probed_page = !0u64;
    macro_rules! ensure_mapped {
        ($pc:expr) => {{
            // arm64 instructions are 4-byte aligned; a misaligned pc is a
            // branch through a corrupted pointer, which faults natively
            // before any fetch. Reject it here rather than fetching from it —
            // the page can well be mapped, and then the only signal that
            // anything is wrong is a nonsense instruction stream.
            if $pc % 4 != 0 {
                return Err(Error::BadAccess($pc));
            }
            let page = $pc & !(GUEST_PAGE - 1);
            if page != probed_page {
                if !page_mapped(page) {
                    return Err(Error::BadAccess($pc));
                }
                probed_page = page;
            }
        }};
    }
    loop {
        ensure_mapped!(pc);
        pc_map.push((out.len() as u32, pc));
        if count >= MAX_BLOCK_GUEST_INSNS {
            // A straight-line run with no branch (a long `.rept`, a jump-table
            // initializer) can outlast the fixed decode window. Rather than fail
            // translation, split the block here: fall through to `pc` with a
            // synthetic unconditional branch and let the run loop resolve the
            // continuation block. `pc` is the next not-yet-decoded instruction,
            // so the split neither drops nor duplicates an instruction.
            let slot = link!(pc);
            emit_terminator_direct(&mut out, pc, None, exit_tramp, slot, false);
            break;
        }
        let insn = unsafe { ptr::read_unaligned(pc as *const u32) };
        match classify_at(insn, pc) {
            InsnKind::Other => {
                out.push(insn);
                pc += 4;
                count += 1;
            }
            InsnKind::Adr { rd, imm } => {
                emit_imm64_compact(&mut out, rd, pc.wrapping_add(imm as u64));
                pc += 4;
                count += 1;
            }
            InsnKind::Adrp { rd, imm } => {
                let base = pc & !0xFFF;
                let target = base.wrapping_add((imm << 12) as u64);
                emit_imm64_compact(&mut out, rd, target);
                pc += 4;
                count += 1;
            }
            InsnKind::LdrLiteral {
                rt,
                byte_offset,
                is_64,
            } => {
                let addr = pc.wrapping_add(byte_offset as u64);
                emit_imm64_compact(&mut out, rt, addr);
                if is_64 {
                    out.push(enc_ldr_imm(rt, rt, 0));
                } else {
                    out.push(enc_ldr32_imm(rt, rt, 0));
                }
                pc += 4;
                count += 1;
            }
            InsnKind::LdrLiteralFp {
                rt,
                byte_offset,
                kind,
            } => {
                // Replace `ldr <fp>t, label` with:
                //   str x16, [sp, #-16]!     ; spill a scratch GPR
                //   movz/movk x16, target    ; materialize literal addr
                //   ldr <fp>t, [x16, #0]     ; load from there
                //   ldr x16, [sp], #16        ; restore
                let addr = pc.wrapping_add(byte_offset as u64);
                out.push(enc_str_pre_index(16, 31, -16));
                emit_imm64_compact(&mut out, 16, addr);
                out.push(enc_ldr_fp_imm0(rt, 16, kind));
                out.push(enc_ldr_post_index(16, 31, 16));
                pc += 4;
                count += 1;
            }
            InsnKind::Branch { target } => {
                let slot = link!(target);
                emit_terminator_direct(
                    &mut out,
                    target,
                    None,
                    exit_tramp,
                    slot,
                    is_back_edge(pc, target),
                );
                break;
            }
            InsnKind::BranchLink { target, next_ip } => {
                let slot = link!(target);
                emit_terminator_direct(
                    &mut out,
                    target,
                    Some(next_ip),
                    exit_tramp,
                    slot,
                    is_back_edge(pc, target),
                );
                break;
            }
            InsnKind::CondBranch {
                target,
                next_ip,
                cond,
            } => {
                emit_terminator_cond(&mut out, cond, ways!(target, next_ip, pc), exit_tramp);
                break;
            }
            InsnKind::Cbz {
                rt,
                sf,
                target,
                next_ip,
                nonzero,
            } => {
                emit_terminator_cbz(
                    &mut out,
                    rt,
                    sf,
                    nonzero,
                    ways!(target, next_ip, pc),
                    exit_tramp,
                );
                break;
            }
            InsnKind::Tbz {
                rt,
                bit,
                target,
                next_ip,
                nonzero,
            } => {
                emit_terminator_tbz(
                    &mut out,
                    rt,
                    bit,
                    nonzero,
                    ways!(target, next_ip, pc),
                    exit_tramp,
                );
                break;
            }
            InsnKind::BranchReg { rn } => {
                emit_terminator_indirect(&mut out, rn, None, exit_tramp, ib_table);
                break;
            }
            InsnKind::BranchLinkReg { rn, next_ip } => {
                emit_terminator_indirect(&mut out, rn, Some(next_ip), exit_tramp, ib_table);
                break;
            }
            InsnKind::Ret { rn } => {
                emit_terminator_indirect(&mut out, rn, None, exit_tramp, ib_table);
                break;
            }
            InsnKind::PacRet { use_b_key } => {
                let _ = use_b_key;
                out.push(enc_xpaci(30));
                emit_terminator_indirect(&mut out, 30, None, exit_tramp, ib_table);
                break;
            }
            InsnKind::PacBranchReg { rn } => {
                out.push(enc_xpaci(rn));
                emit_terminator_indirect(&mut out, rn, None, exit_tramp, ib_table);
                break;
            }
            InsnKind::PacBranchLinkReg { rn, next_ip } => {
                out.push(enc_xpaci(rn));
                emit_terminator_indirect(&mut out, rn, Some(next_ip), exit_tramp, ib_table);
                break;
            }
            InsnKind::PacNop => {
                // Emit nothing: the pointer keeps its (unsigned) value.
                pc += 4;
                count += 1;
            }
            InsnKind::MrsTpidrro { rt } => {
                // rt = ctx.guest_tsd, the guest's virtualized TSD base. An xzr
                // destination discards the read; emitting the ctx load would
                // turn xzr into a zero base register, so emit nothing.
                if rt != 31 {
                    emit_load_ctx(&mut out, rt);
                    out.push(enc_ldr_imm(rt, rt, guest_tsd_off()));
                }
                pc += 4;
                count += 1;
            }
            InsnKind::Brk => {
                emit_terminator_brk(&mut out, pc, trap_exit);
                break;
            }
            InsnKind::Svc { next_ip } => {
                emit_terminator_svc(&mut out, next_ip, syscall_tramp);
                break;
            }
            InsnKind::Unsupported => {
                return Err(Error::Translate(format!(
                    "unsupported instruction {:#010x} at {:#x}",
                    insn, pc
                )));
            }
        }
    }

    if crate::trace::trace_emit() {
        let words: Vec<String> = out.iter().map(|w| format!("{w:08x}")).collect();
        eprintln!("chimera: emit {guest_pc:#x} [{}]", words.join(" "));
    }
    let linked_pc = cache.emit(&out)?;
    Ok(Translation {
        // The dispatcher (and any exit trampoline) enters past the two-word
        // linked entry, which only a linked predecessor may run.
        host_pc: linked_pc + 8,
        host_end: linked_pc + (out.len() as u64) * 4,
        guest_end: pc + 4,
        linked_pc,
        pc_map,
        edges,
    })
}

// === Terminator emission ===

/// Push the guest x16/x17 to the guest stack, then load the ctx pointer
/// into x17. After this sequence, x16 and x17 are scratch and `[sp, #0]`
/// / `[sp, #8]` hold the guest values of x16 / x17.
///
/// The push writes below the guest's sp. AAPCS64 reserves nothing there and
/// XNU itself pushes signal frames into it, so this breaks no guarantee — but
/// a guest whose sp sits within 16 bytes of a guard page would not have taken
/// the fault this can. Freeing a register without first having one is not
/// possible on arm64, so the spill stays; `abi/red-zone.c` tracks the
/// divergence and records what was ruled out.
fn emit_save_x16_x17_and_load_ctx(out: &mut Vec<u32>) {
    // stp x16, x17, [sp, #-16]!
    out.push(enc_stp_pre_index(16, 17, 31, -16));
    emit_load_ctx(out, 17); // x17 = ctx (this thread's TSD slot)
}

/// Final tail of every exit stub: store the new pc into ctx.pc using x17
/// as the ctx base, materialize the exit trampoline into x16, and branch.
fn emit_exit_tail(out: &mut Vec<u32>, exit_tramp: u64) {
    out.push(enc_str_imm(16, 17, pc_off())); // ctx.pc = x16
    emit_imm64_fixed(out, 16, exit_tramp);
    out.push(enc_br(16));
}

/// Exit tail for an edge whose guest target is statically known: try the
/// direct link first, fall back to the trampoline. `slot` is a word of
/// ordinary read-write memory the cache fills with the successor block's
/// linked entry once that block is translated (and re-zeroes when it is
/// invalidated), so linking never rewrites code — no JIT write-protect
/// toggling, no instruction-cache maintenance, and a lock-free reader sees
/// either the old value or the new one.
///
/// `poll` arms the asynchronous safepoint check, which every loop-closing
/// edge needs: a chain of linked blocks stays inside one `dispatch` call, so
/// the run loop's boundary work — signal delivery, a sibling's `exit_group`
/// or committed `execve` — would otherwise never get a turn. When the flag is
/// set the edge takes the cold path, returning to the run loop with `ctx.pc`
/// already pointing at the successor, so delivery happens at a clean
/// boundary.
///
/// Entry state matches [`emit_exit_tail`]: `x16` = next guest PC, `x17` =
/// ctx, and the guest's own x16/x17 saved on the guest stack — which is why
/// the link targets the successor's `linked_pc`, the entry that pops them.
fn emit_exit_tail_linked(out: &mut Vec<u32>, exit_tramp: u64, slot: u64, poll: bool) {
    out.push(enc_str_imm(16, 17, pc_off())); // ctx.pc = x16
    if poll {
        out.push(enc_ldr32_imm(16, 17, exit_requested_off()));
        // Skip the 7-word link sequence to the cold tail.
        out.push(enc_cbz32(16, true, 8));
    }
    emit_imm64_fixed(out, 16, slot);
    out.push(enc_ldr_imm(16, 16, 0)); // x16 = *slot
    out.push(enc_cbz64(16, false, 2)); // unlinked (slot still zero): cold tail
    out.push(enc_br(16));
    // Cold tail: the trampoline, with ctx.pc already set above.
    emit_imm64_fixed(out, 16, exit_tramp);
    out.push(enc_br(16));
}

/// Whether an edge from `from` to `target` closes a loop. A backward branch
/// is the only shape that can keep control inside the cache indefinitely, so
/// it is the only one that needs the safepoint poll; forward chains are
/// finite and end at an indirect branch or a back edge.
fn is_back_edge(from: u64, target: u64) -> bool {
    target <= from
}

fn emit_terminator_direct(
    out: &mut Vec<u32>,
    target: u64,
    next_ip_for_lr: Option<u64>,
    exit_tramp: u64,
    slot: Option<u64>,
    poll: bool,
) {
    emit_save_x16_x17_and_load_ctx(out);
    if let Some(next_ip) = next_ip_for_lr {
        // x30 = next_ip (BL semantics: discard guest x30 and write next_ip)
        emit_imm64_compact(out, 16, next_ip);
        out.push(enc_mov_reg(30, 16));
    }
    emit_imm64_compact(out, 16, target);
    match slot {
        Some(slot) => emit_exit_tail_linked(out, exit_tramp, slot, poll),
        None => emit_exit_tail(out, exit_tramp),
    }
}

/// Build one arm of a two-way terminator: materialize its next guest PC into
/// x16 and emit its exit tail (linked or plain). `reload_ctx` re-derives x17
/// for the arms of a test whose operand forced x17 to be borrowed.
fn cond_arm(next_pc: u64, edge: Edge, exit_tramp: u64, reload_ctx: bool) -> Vec<u32> {
    let mut arm = Vec::new();
    if reload_ctx {
        emit_load_ctx(&mut arm, 17);
    }
    emit_imm64_compact(&mut arm, 16, next_pc);
    match edge.slot {
        Some(slot) => emit_exit_tail_linked(&mut arm, exit_tramp, slot, edge.poll),
        None => emit_exit_tail(&mut arm, exit_tramp),
    }
    arm
}

/// One outgoing edge of a two-way terminator: the link slot its stub reads
/// (`None` leaves it unlinked) and whether it needs the safepoint poll.
#[derive(Clone, Copy)]
pub struct Edge {
    pub slot: Option<u64>,
    pub poll: bool,
}

/// Both destinations of a two-way terminator, each with its own link.
#[derive(Clone, Copy)]
pub struct TwoWay {
    pub target: u64,
    pub fallthrough: u64,
    pub taken: Edge,
    pub not_taken: Edge,
}

fn emit_terminator_cond(out: &mut Vec<u32>, cond: u8, ways: TwoWay, exit_tramp: u64) {
    // The guest's NZCV is still live here: pushing x16/x17 and loading the ctx
    // (stp/mrs/and/ldr) leaves the flags untouched, so the condition can be
    // tested directly rather than materialized through `csel`. Each arm then
    // ends in its own exit tail, which is what makes both edges linkable.
    emit_save_x16_x17_and_load_ctx(out);
    let not_taken_arm = cond_arm(ways.fallthrough, ways.not_taken, exit_tramp, false);
    let taken_arm = cond_arm(ways.target, ways.taken, exit_tramp, false);
    out.push(enc_b_cond(cond, not_taken_arm.len() as i32 + 1));
    out.extend_from_slice(&not_taken_arm);
    out.extend_from_slice(&taken_arm);
}

/// Save the guest x16/x17 and get the register-test terminators' operand into
/// a register that survives it. CBZ/TBZ read `rt` and touch no flags, but the
/// stub is about to clobber x16/x17 — so when `rt` is one of those, its guest
/// value is reloaded from the stack slot the push just wrote, into x17. The
/// arms then re-derive the ctx pointer for themselves.
fn emit_test_operand(out: &mut Vec<u32>, rt: u8) -> u8 {
    out.push(enc_stp_pre_index(16, 17, 31, -16));
    match rt {
        16 => {
            out.push(enc_ldr_imm(17, 31, 0)); // ldr x17, [sp]
            17
        }
        17 => {
            out.push(enc_ldr_imm(17, 31, 8)); // ldr x17, [sp, #8]
            17
        }
        _ => rt,
    }
}

fn emit_terminator_cbz(
    out: &mut Vec<u32>,
    rt: u8,
    sf: bool,
    nonzero: bool,
    ways: TwoWay,
    exit_tramp: u64,
) {
    let test_reg = emit_test_operand(out, rt);
    let not_taken_arm = cond_arm(ways.fallthrough, ways.not_taken, exit_tramp, true);
    let taken_arm = cond_arm(ways.target, ways.taken, exit_tramp, true);
    let over = not_taken_arm.len() as i32 + 1;
    out.push(if sf {
        enc_cbz64(test_reg, nonzero, over)
    } else {
        enc_cbz32(test_reg, nonzero, over)
    });
    out.extend_from_slice(&not_taken_arm);
    out.extend_from_slice(&taken_arm);
}

fn emit_terminator_tbz(
    out: &mut Vec<u32>,
    rt: u8,
    bit: u8,
    nonzero: bool,
    ways: TwoWay,
    exit_tramp: u64,
) {
    let test_reg = emit_test_operand(out, rt);
    let not_taken_arm = cond_arm(ways.fallthrough, ways.not_taken, exit_tramp, true);
    let taken_arm = cond_arm(ways.target, ways.taken, exit_tramp, true);
    // TBZ's imm14 is in instruction units — ample for an arm of this size.
    out.push(enc_tbz(
        test_reg,
        bit,
        nonzero,
        not_taken_arm.len() as i32 + 1,
    ));
    out.extend_from_slice(&not_taken_arm);
    out.extend_from_slice(&taken_arm);
}

fn emit_terminator_indirect(
    out: &mut Vec<u32>,
    rn: u8,
    next_ip_for_lr: Option<u64>,
    exit_tramp: u64,
    ib_table: u64,
) {
    // The source register `rn` could be x16 or x17, whose guest values
    // are about to be (or already have been) pushed onto the stack. We
    // capture rn FIRST into a temp on the stack, then proceed.
    //
    // Sequence:
    //   str rn, [sp, #-16]!         ; spill rn (if rn=16/17 still live)
    //   stp x16, x17, [sp, #-16]!   ; now save guest x16, x17 too
    //   (but that double-pushes for rn != 16/17)
    //
    // Easier: handle each case.
    //   rn != 16, 17, 30:  rn's live value is its guest value. Push x16/x17,
    //                      load ctx, build next_ip / pc / tail using rn
    //                      directly.
    //   rn == 16:          rn's guest value is in the live x16 BEFORE we
    //                      stp. Capture into a temp first.
    //   rn == 17:          similarly.
    //   rn == 30 + BL:     for BL we'll overwrite x30 with next_ip, so
    //                      capture x30 first.
    //
    // For BL: we read x30 BEFORE overwriting. The original x30 is the
    // *new* x30 = next_ip; the OLD x30 (return address of caller) is what
    // exit_block stores into ctx.regs[30]. Wait — BLR semantics: x30 =
    // next_ip; pc = xRn. The OLD x30 is discarded. So we just need to
    // overwrite x30 with next_ip; the next_ip ends up in ctx.regs[30]
    // when exit_block runs (because it reads live x30).

    // Step 1: save guest x16, x17 (we'll clobber them).
    out.push(enc_stp_pre_index(16, 17, 31, -16));

    // Step 2: load ctx into x17.
    emit_load_ctx(out, 17); // x17 = ctx (this thread's TSD slot)

    // Step 3: read the indirect target (xRn's guest value) into x16.
    if rn == 16 {
        // Guest x16 was just pushed at [sp, #0] (16 bytes below current sp,
        // but sp was pre-decremented, so it's at [sp, #0]).
        out.push(enc_ldr_imm(16, 31, 0));
    } else if rn == 17 {
        out.push(enc_ldr_imm(16, 31, 8));
    } else {
        // Plain mov x16, xRn.
        out.push(enc_mov_reg(16, rn));
    }

    // Step 4: if this is a call (BL/BLR), set x30 = next_ip. We use
    // a small scratch dance: spill x16 to ctx.pc, build next_ip in x16,
    // mov it to x30, then rebuild target in x16.
    if let Some(next_ip) = next_ip_for_lr {
        // ctx.pc = x16 (the target we just read)
        out.push(enc_str_imm(16, 17, pc_off()));
        emit_imm64_compact(out, 16, next_ip);
        out.push(enc_mov_reg(30, 16));
        // Re-read ctx.pc back into x16 to set up the tail.
        out.push(enc_ldr_imm(16, 17, pc_off()));
    }

    if crate::trace::no_link() {
        emit_exit_tail(out, exit_tramp);
    } else {
        emit_ib_probe(out, exit_tramp, ib_table);
    }
}

/// Exit tail for an indirect branch (`BR`/`BLR`/`RET`): probe the
/// indirect-branch table for the runtime target and, on a hit, jump straight
/// into the successor. A return or a computed call is the one edge whose
/// destination is unknown at translation time, so it cannot be linked ahead
/// of time the way [`emit_exit_tail_linked`] links a static branch — without
/// this probe, every function return leaves the cache.
///
/// Entry state matches [`emit_exit_tail`]: `x16` = target guest PC, `x17` =
/// ctx, guest x16/x17 on the guest stack. The probe borrows one more
/// register (x0) through a second stack slot, popped on both paths.
///
/// The table is read with a single aligned `ldp`, which is single-copy atomic
/// on this backend's hardware (see `CodeCache::ib_insert`), so the key and
/// host PC always come from the same generation. A null host means an empty
/// slot — how invalidation drops an entry — so it is rejected before the
/// branch. The safepoint poll comes first and unconditionally: an indirect
/// edge closes loops too (a `while` over a function-pointer table, a
/// recursive call's returns), and unlike a static branch there is no target
/// to compare against to tell a back edge from a forward one.
///
/// Every instruction here leaves NZCV alone — the key is compared with `eor`
/// and `cbnz` rather than `cmp` and `b.ne`. The guest's flags are live across
/// this stub: a `br` into a jump-table arm may branch on them, and on the
/// cold path the trampoline records them as the guest's architectural state.
fn emit_ib_probe(out: &mut Vec<u32>, exit_tramp: u64, ib_table: u64) {
    out.push(enc_str_imm(16, 17, pc_off())); // ctx.pc = target

    // Cold tail, emitted last but sized first so the branches can reach it.
    let mut cold = Vec::new();
    emit_imm64_fixed(&mut cold, 16, exit_tramp);
    cold.push(enc_br(16));

    // The probe proper, with x0 borrowed as a third scratch register.
    let mut probe = Vec::new();
    probe.push(enc_str_pre_index(0, 31, -16)); // spill guest x0
    probe.push(enc_ldr_imm(16, 17, pc_off())); // x16 = target
    emit_imm64_fixed(&mut probe, 0, IB_HASH_MULT);
    probe.push(enc_mul(0, 16, 0)); // x0 = target * MULT
    probe.push(enc_lsr_imm(0, 0, 64 - IB_BITS)); // x0 = slot index
    emit_imm64_fixed(&mut probe, 17, ib_table); // x17 = table base (ctx dropped)
    probe.push(enc_add_lsl(0, 17, 0, 4)); // x0 = &entry (16 bytes each)
    probe.push(enc_ldp_imm0(17, 0, 0)); // x17 = key, x0 = host
    probe.push(enc_eor_reg(17, 17, 16)); // zero exactly when the key matches
    // A miss (wrong key, or an empty slot) skips the four hit words —
    // cbnz/mov/restore/br — landing on the restore-and-cold tail below.
    probe.push(enc_cbz64(17, true, 5));
    probe.push(enc_cbz64(0, false, 4)); // empty slot -> miss
    probe.push(enc_mov_reg(16, 0)); // x16 = successor linked entry
    probe.push(enc_ldr_post_index(0, 31, 16)); // restore guest x0
    probe.push(enc_br(16));
    // Miss: restore x0 and the ctx pointer, then fall into the cold tail.
    probe.push(enc_ldr_post_index(0, 31, 16));
    emit_load_ctx(&mut probe, 17);

    // Poll first: if a stop or a pending signal is armed, skip the probe.
    out.push(enc_ldr32_imm(16, 17, exit_requested_off()));
    out.push(enc_cbz32(16, true, probe.len() as i32 + 1));
    out.extend_from_slice(&probe);
    out.extend_from_slice(&cold);
}

/// `BRK`: leave the cache with `ctx.pc` at the `BRK` itself. AArch64 takes it
/// as a synchronous fault, so the architectural PC is the trapping
/// instruction — a guest handler that returns re-executes it, exactly as it
/// would natively.
fn emit_terminator_brk(out: &mut Vec<u32>, pc: u64, trap_exit: u64) {
    emit_save_x16_x17_and_load_ctx(out);
    emit_imm64_compact(out, 16, pc);
    emit_exit_tail(out, trap_exit);
}

fn emit_terminator_svc(out: &mut Vec<u32>, next_ip: u64, syscall_tramp: u64) {
    // SVC: real kernels switch to their own kernel stack and never touch
    // the user stack — `testing/conformance/abi/syscall-no-stack-touch.c`
    // exercises exactly this property. Unlike the generic `B`/`BL` exit
    // paths (which spill x16/x17 to the guest stack), the SVC stub stores
    // the syscall number directly into `ctx.regs[16]` and clobbers x17 as
    // the ctx scratch. Discarding x17 is fine: AArch64 PCS marks it as
    // IP1, an inter-procedure-call scratch that callees and the kernel
    // are free to clobber.
    //
    //   mrs  x17, tpidrro_el0 ; and ; ldr  ; x17 = ctx (this thread's TSD slot)
    //   str  x16, [x17, #regs_off(16)]    ; ctx.regs[16] = syscall number
    //   movz/movk x16, #next_ip
    //   str  x16, [x17, #pc_off]           ; ctx.pc = resume pc
    //   movz/movk x16, #syscall_tramp
    //   br   x16
    emit_load_ctx(out, 17); // x17 = ctx (this thread's TSD slot)
    out.push(enc_str_imm(16, 17, regs_off(16)));
    emit_imm64_compact(out, 16, next_ip);
    out.push(enc_str_imm(16, 17, pc_off()));
    emit_imm64_fixed(out, 16, syscall_tramp);
    out.push(enc_br(16));
}

// === Decoder ===

#[derive(Debug, Clone, Copy)]
enum FpLoadKind {
    Word32,  // S registers
    Word64,  // D registers
    Word128, // Q registers
}

#[derive(Debug, Clone, Copy)]
enum InsnKind {
    Other,
    Adr {
        rd: u8,
        imm: i64,
    },
    Adrp {
        rd: u8,
        imm: i64,
    },
    LdrLiteral {
        rt: u8,
        byte_offset: i64,
        is_64: bool,
    },
    LdrLiteralFp {
        rt: u8,
        byte_offset: i64,
        kind: FpLoadKind,
    },
    Branch {
        target: u64,
    },
    BranchLink {
        target: u64,
        next_ip: u64,
    },
    CondBranch {
        target: u64,
        next_ip: u64,
        cond: u8,
    },
    Cbz {
        rt: u8,
        sf: bool,
        target: u64,
        next_ip: u64,
        nonzero: bool,
    },
    Tbz {
        rt: u8,
        bit: u8,
        target: u64,
        next_ip: u64,
        nonzero: bool,
    },
    BranchReg {
        rn: u8,
    },
    BranchLinkReg {
        rn: u8,
        next_ip: u64,
    },
    Ret {
        rn: u8,
    },
    PacRet {
        use_b_key: bool,
    },
    PacBranchReg {
        rn: u8,
    },
    PacBranchLinkReg {
        rn: u8,
        next_ip: u64,
    },
    Svc {
        next_ip: u64,
    },
    /// `BRK #imm` — a software breakpoint. It must not run from the cache:
    /// the host would take the exception against Chimera's own execution
    /// state, with no guest handler consulted. Exiting instead lets the run
    /// loop raise `SIGTRAP` for the guest. Terminating the block also stops
    /// the decoder walking through the trap padding compilers place between
    /// functions and merging the next one into this block.
    Brk,
    /// A standalone PAC sign/authenticate op (`pacia`, `autda`, `paciasp`, …).
    /// Chimera is PAC-oblivious, so these translate to nothing.
    PacNop,
    /// `mrs <rt>, TPIDRRO_EL0` — the guest asking for its TSD base. The real
    /// register is Chimera's (it reaches the ctx), so this reads
    /// `ctx.guest_tsd` instead.
    MrsTpidrro {
        rt: u8,
    },
    Unsupported,
}

fn classify_at(insn: u32, pc: u64) -> InsnKind {
    if (insn & 0xFC000000) == 0x14000000 {
        let imm26 = sign_extend((insn & 0x03FFFFFF) as i64, 26);
        let target = pc.wrapping_add((imm26 * 4) as u64);
        return InsnKind::Branch { target };
    }
    if (insn & 0xFC000000) == 0x94000000 {
        let imm26 = sign_extend((insn & 0x03FFFFFF) as i64, 26);
        let target = pc.wrapping_add((imm26 * 4) as u64);
        return InsnKind::BranchLink {
            target,
            next_ip: pc + 4,
        };
    }
    if (insn & 0xFF000010) == 0x54000000 {
        let imm19 = sign_extend(((insn >> 5) & 0x7FFFF) as i64, 19);
        let target = pc.wrapping_add((imm19 * 4) as u64);
        let cond = (insn & 0xF) as u8;
        return InsnKind::CondBranch {
            target,
            next_ip: pc + 4,
            cond,
        };
    }
    if (insn & 0x7E000000) == 0x34000000 {
        let sf = (insn >> 31) != 0;
        let nonzero = ((insn >> 24) & 1) != 0;
        let rt = (insn & 0x1F) as u8;
        let imm19 = sign_extend(((insn >> 5) & 0x7FFFF) as i64, 19);
        let target = pc.wrapping_add((imm19 * 4) as u64);
        return InsnKind::Cbz {
            rt,
            sf,
            nonzero,
            target,
            next_ip: pc + 4,
        };
    }
    if (insn & 0xFE1FFC1F) == 0xD61F0000 {
        let op = (insn >> 21) & 0x3;
        let rn = ((insn >> 5) & 0x1F) as u8;
        return match op {
            0b00 => InsnKind::BranchReg { rn },
            0b01 => InsnKind::BranchLinkReg {
                rn,
                next_ip: pc + 4,
            },
            0b10 => InsnKind::Ret { rn },
            _ => InsnKind::Unsupported,
        };
    }
    if insn == 0xD65F0BFF {
        return InsnKind::PacRet { use_b_key: false };
    }
    if insn == 0xD65F0FFF {
        return InsnKind::PacRet { use_b_key: true };
    }
    if (insn & 0xFFFFFC1F) == 0xD61F081F || (insn & 0xFFFFFC1F) == 0xD61F0C1F {
        let rn = ((insn >> 5) & 0x1F) as u8;
        return InsnKind::PacBranchReg { rn };
    }
    if (insn & 0xFFFFFC1F) == 0xD63F081F || (insn & 0xFFFFFC1F) == 0xD63F0C1F {
        let rn = ((insn >> 5) & 0x1F) as u8;
        return InsnKind::PacBranchLinkReg {
            rn,
            next_ip: pc + 4,
        };
    }
    // `BRAA  Xn, Xm` / `BRAB  Xn, Xm`  — PAC indirect branch with an
    // explicit modifier register. Bits[20:16] are fixed at `11111` (the
    // signature of the unconditional-branch-register encoding family) and
    // must be matched; only Rn (bits 9:5) and Rm (bits 4:0) vary. The
    // earlier mask `0xFFE0FC00` accidentally left bits[20:16] *un*-checked,
    // so a real `BRAA x16, x17` (used by dyld_shared_cache stubs as the
    // final hop into libSystem) slipped through as `Other` and the
    // translator slid right past it into the next stub, eventually hitting
    // the per-block instruction cap.
    if (insn & 0xFFFFFC00) == 0xD71F0800 || (insn & 0xFFFFFC00) == 0xD71F0C00 {
        let rn = ((insn >> 5) & 0x1F) as u8;
        return InsnKind::PacBranchReg { rn };
    }
    if (insn & 0xFFFFFC00) == 0xD73F0800 || (insn & 0xFFFFFC00) == 0xD73F0C00 {
        let rn = ((insn >> 5) & 0x1F) as u8;
        return InsnKind::PacBranchLinkReg {
            rn,
            next_ip: pc + 4,
        };
    }
    if (insn & 0xFFE0001F) == 0xD4000001 {
        return InsnKind::Svc { next_ip: pc + 4 };
    }
    if (insn & 0xFFE0001F) == 0xD4200000 {
        return InsnKind::Brk;
    }
    if (insn & 0x1F000000) == 0x10000000 {
        let immlo = ((insn >> 29) & 0x3) as i64;
        let immhi = ((insn >> 5) & 0x7FFFF) as i64;
        let imm = sign_extend((immhi << 2) | immlo, 21);
        let rd = (insn & 0x1F) as u8;
        if (insn >> 31) == 0 {
            return InsnKind::Adr { rd, imm };
        } else {
            return InsnKind::Adrp { rd, imm };
        }
    }
    if (insn & 0x3B000000) == 0x18000000 {
        let opc = (insn >> 30) & 0x3;
        let v = (insn >> 26) & 0x1;
        let imm19 = sign_extend(((insn >> 5) & 0x7FFFF) as i64, 19);
        let rt = (insn & 0x1F) as u8;
        if v == 0 {
            return match opc {
                0b00 => InsnKind::LdrLiteral {
                    rt,
                    byte_offset: imm19 * 4,
                    is_64: false,
                },
                0b01 => InsnKind::LdrLiteral {
                    rt,
                    byte_offset: imm19 * 4,
                    is_64: true,
                },
                _ => InsnKind::Unsupported,
            };
        } else {
            return match opc {
                0b00 => InsnKind::LdrLiteralFp {
                    rt,
                    byte_offset: imm19 * 4,
                    kind: FpLoadKind::Word32,
                },
                0b01 => InsnKind::LdrLiteralFp {
                    rt,
                    byte_offset: imm19 * 4,
                    kind: FpLoadKind::Word64,
                },
                0b10 => InsnKind::LdrLiteralFp {
                    rt,
                    byte_offset: imm19 * 4,
                    kind: FpLoadKind::Word128,
                },
                _ => InsnKind::Unsupported,
            };
        }
    }
    if (insn & 0x7E000000) == 0x36000000 {
        let nonzero = ((insn >> 24) & 1) != 0;
        let b5 = (insn >> 31) & 1;
        let b40 = (insn >> 19) & 0x1F;
        let bit = ((b5 << 5) | b40) as u8;
        let rt = (insn & 0x1F) as u8;
        let imm14 = sign_extend(((insn >> 5) & 0x3FFF) as i64, 14);
        let target = pc.wrapping_add((imm14 * 4) as u64);
        return InsnKind::Tbz {
            rt,
            bit,
            nonzero,
            target,
            next_ip: pc + 4,
        };
    }
    // Standalone PAC sign/authenticate. Chimera keeps every pointer unsigned
    // (fixups write plain pointers; runtime signing is elided here) and strips
    // PAC on branch, so these must be no-ops — a verbatim `autia`/`autda` on an
    // unsigned pointer would fail authentication and poison it.
    //   - register/zero form pacia..autdzb: DAC1_0000..DAC1_3FFF (xpaci/xpacd at
    //     DAC1_43E0/47E0 are excluded and left verbatim — stripping is harmless);
    //   - SP/LR hint form paciasp/autiasp/paciaz/…: D5032_31F..D5032_3FF.
    if (insn & 0xFFFF_C000) == 0xDAC1_0000 || (insn & 0xFFFF_FF1F) == 0xD503_231F {
        return InsnKind::PacNop;
    }
    if (insn & 0xFFFF_FFE0) == 0xD53B_D060 {
        return InsnKind::MrsTpidrro {
            rt: (insn & 0x1F) as u8,
        };
    }
    InsnKind::Other
}

// === Encoders ===

const fn regs_off(idx: usize) -> u32 {
    (std::mem::offset_of!(ThreadState, regs) + idx * 8) as u32
}

const fn pc_off() -> u32 {
    std::mem::offset_of!(ThreadState, pc) as u32
}

const fn guest_tsd_off() -> u32 {
    std::mem::offset_of!(ThreadState, guest_tsd) as u32
}

const fn exit_requested_off() -> u32 {
    std::mem::offset_of!(ThreadState, exit_requested) as u32
}

/// `mul <rd>, <rn>, <rm>`.
fn enc_mul(rd: u8, rn: u8, rm: u8) -> u32 {
    0x9B007C00 | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
}

/// `lsr <rd>, <rn>, #shift` (the `ubfm` alias).
fn enc_lsr_imm(rd: u8, rn: u8, shift: u32) -> u32 {
    0xD340_0000
        | ((shift & 0x3F) << 16)
        | (0x3F << 10)
        | ((rn as u32 & 0x1F) << 5)
        | (rd as u32 & 0x1F)
}

/// `add <rd>, <rn>, <rm>, lsl #shift`.
fn enc_add_lsl(rd: u8, rn: u8, rm: u8, shift: u32) -> u32 {
    0x8B000000
        | ((rm as u32 & 0x1F) << 16)
        | ((shift & 0x3F) << 10)
        | ((rn as u32 & 0x1F) << 5)
        | (rd as u32 & 0x1F)
}

/// `eor <rd>, <rn>, <rm>` — flag-preserving, unlike `cmp`.
fn enc_eor_reg(rd: u8, rn: u8, rm: u8) -> u32 {
    0xCA000000 | ((rm as u32 & 0x1F) << 16) | ((rn as u32 & 0x1F) << 5) | (rd as u32 & 0x1F)
}

/// `ldp <rt1>, <rt2>, [<rn>]` — 64-bit signed-offset load pair at offset 0.
fn enc_ldp_imm0(rt1: u8, rt2: u8, rn: u8) -> u32 {
    0xA9400000 | ((rt2 as u32 & 0x1F) << 10) | ((rn as u32 & 0x1F) << 5) | (rt1 as u32 & 0x1F)
}

/// `b.<cond> <insn_offset>` — conditional branch, offset in instructions.
fn enc_b_cond(cond: u8, insn_offset: i32) -> u32 {
    let imm19 = (insn_offset as u32) & 0x7FFFF;
    0x54000000 | (imm19 << 5) | (cond as u32 & 0xF)
}

/// `ldp <rt1>, <rt2>, [<rn>], #imm` — 64-bit post-index load pair.
fn enc_ldp_post_index(rt1: u8, rt2: u8, rn: u8, byte_offset: i32) -> u32 {
    let imm7 = ((byte_offset / 8) as u32) & 0x7F;
    0xA8C00000
        | (imm7 << 15)
        | ((rt2 as u32 & 0x1F) << 10)
        | ((rn as u32 & 0x1F) << 5)
        | (rt1 as u32 & 0x1F)
}

fn enc_movz(rd: u8, imm: u16, shift: u8) -> u32 {
    let hw = ((shift / 16) & 0x3) as u32;
    0xD2800000 | (hw << 21) | ((imm as u32) << 5) | (rd as u32 & 0x1F)
}

fn enc_movk(rd: u8, imm: u16, shift: u8) -> u32 {
    let hw = ((shift / 16) & 0x3) as u32;
    0xF2800000 | (hw << 21) | ((imm as u32) << 5) | (rd as u32 & 0x1F)
}

fn enc_mov_reg(rd: u8, rm: u8) -> u32 {
    0xAA0003E0 | ((rm as u32 & 0x1F) << 16) | (rd as u32 & 0x1F)
}

fn enc_ldr_imm(rt: u8, rn: u8, byte_offset: u32) -> u32 {
    let imm12 = (byte_offset / 8) & 0xFFF;
    0xF9400000 | (imm12 << 10) | ((rn as u32 & 0x1F) << 5) | (rt as u32 & 0x1F)
}

fn enc_ldr32_imm(rt: u8, rn: u8, byte_offset: u32) -> u32 {
    let imm12 = (byte_offset / 4) & 0xFFF;
    0xB9400000 | (imm12 << 10) | ((rn as u32 & 0x1F) << 5) | (rt as u32 & 0x1F)
}

fn enc_str_imm(rt: u8, rn: u8, byte_offset: u32) -> u32 {
    let imm12 = (byte_offset / 8) & 0xFFF;
    0xF9000000 | (imm12 << 10) | ((rn as u32 & 0x1F) << 5) | (rt as u32 & 0x1F)
}

/// `str xt, [sp, #imm]!` — pre-indexed store with writeback. `rn=31`
/// is SP. imm is signed 9-bit.
fn enc_str_pre_index(rt: u8, rn: u8, imm: i32) -> u32 {
    let imm9 = ((imm as u32) & 0x1FF) << 12;
    0xF8000C00 | imm9 | ((rn as u32 & 0x1F) << 5) | (rt as u32 & 0x1F)
}

/// `ldr xt, [sp], #imm` — post-indexed load with writeback.
fn enc_ldr_post_index(rt: u8, rn: u8, imm: i32) -> u32 {
    let imm9 = ((imm as u32) & 0x1FF) << 12;
    0xF8400400 | imm9 | ((rn as u32 & 0x1F) << 5) | (rt as u32 & 0x1F)
}

/// `stp xt1, xt2, [rn, #imm]!` — pre-indexed STP with writeback. imm is
/// signed 7-bit *8 (so byte_offset must be a multiple of 8).
fn enc_stp_pre_index(rt1: u8, rt2: u8, rn: u8, byte_offset: i32) -> u32 {
    let imm7 = ((byte_offset / 8) & 0x7F) as u32;
    0xA9800000
        | (imm7 << 15)
        | ((rt2 as u32 & 0x1F) << 10)
        | ((rn as u32 & 0x1F) << 5)
        | (rt1 as u32 & 0x1F)
}

fn enc_br(rn: u8) -> u32 {
    0xD61F0000 | ((rn as u32 & 0x1F) << 5)
}

fn enc_b_imm(insn_offset: i32) -> u32 {
    let imm26 = (insn_offset as u32) & 0x03FFFFFF;
    0x14000000 | imm26
}

fn enc_cbz32(rt: u8, nonzero: bool, insn_offset: i32) -> u32 {
    let imm19 = (insn_offset as u32) & 0x7FFFF;
    let op = if nonzero { 1 } else { 0 };
    0x34000000 | (op << 24) | (imm19 << 5) | (rt as u32 & 0x1F)
}

fn enc_cbz64(rt: u8, nonzero: bool, insn_offset: i32) -> u32 {
    let imm19 = (insn_offset as u32) & 0x7FFFF;
    let op = if nonzero { 1 } else { 0 };
    0xB4000000 | (op << 24) | (imm19 << 5) | (rt as u32 & 0x1F)
}

fn enc_tbz(rt: u8, bit: u8, nonzero: bool, insn_offset: i32) -> u32 {
    let b5 = ((bit >> 5) & 1) as u32;
    let b40 = (bit as u32) & 0x1F;
    let op = if nonzero { 1 } else { 0 };
    let imm14 = (insn_offset as u32) & 0x3FFF;
    0x36000000 | (b5 << 31) | (op << 24) | (b40 << 19) | (imm14 << 5) | (rt as u32 & 0x1F)
}

fn enc_xpaci(rd: u8) -> u32 {
    0xDAC143E0 | (rd as u32 & 0x1F)
}

/// `ldr <fp>t, [Xn]` — unsigned-offset SIMD/FP load with zero immediate.
fn enc_ldr_fp_imm0(rt: u8, rn: u8, kind: FpLoadKind) -> u32 {
    // SIMD/FP "Load/store register (unsigned immediate)" encoding:
    //   size(2) 111 1 01 opc(2) imm12 Rn Rt
    // For loads (opc bit 0 = 1):
    //   32-bit (S):  size=10, opc=01 → 0xBD400000
    //   64-bit (D):  size=11, opc=01 → 0xFD400000
    //   128-bit (Q): size=00, opc=11 → 0x3DC00000
    let base = match kind {
        FpLoadKind::Word32 => 0xBD400000u32,
        FpLoadKind::Word64 => 0xFD400000u32,
        FpLoadKind::Word128 => 0x3DC00000u32,
    };
    base | ((rn as u32 & 0x1F) << 5) | (rt as u32 & 0x1F)
}

// === Immediate materialization ===

/// Emit exactly 4 instructions to materialize a 64-bit immediate into Rd.
/// Always-fixed length is required when emitting code into a diamond
/// where both legs must be the same size for control-flow correctness.
fn emit_imm64_padded(out: &mut Vec<u32>, rd: u8, val: u64, count: usize) {
    debug_assert_eq!(count, 4, "only 4-wide padding is supported");
    out.push(enc_movz(rd, (val & 0xFFFF) as u16, 0));
    out.push(enc_movk(rd, ((val >> 16) & 0xFFFF) as u16, 16));
    out.push(enc_movk(rd, ((val >> 32) & 0xFFFF) as u16, 32));
    out.push(enc_movk(rd, ((val >> 48) & 0xFFFF) as u16, 48));
}

/// Always emit 4 instructions, even for small immediates. Used in places
/// where the surrounding sequence relies on a fixed layout.
fn emit_imm64_fixed(out: &mut Vec<u32>, rd: u8, val: u64) {
    emit_imm64_padded(out, rd, val, 4);
}

/// Emit the shortest sequence that materializes `val` into Rd. Useful in
/// places where length doesn't matter.
fn emit_imm64_compact(out: &mut Vec<u32>, rd: u8, val: u64) {
    let parts = [
        (val & 0xFFFF) as u16,
        ((val >> 16) & 0xFFFF) as u16,
        ((val >> 32) & 0xFFFF) as u16,
        ((val >> 48) & 0xFFFF) as u16,
    ];
    out.push(enc_movz(rd, parts[0], 0));
    if parts[1] != 0 {
        out.push(enc_movk(rd, parts[1], 16));
    }
    if parts[2] != 0 {
        out.push(enc_movk(rd, parts[2], 32));
    }
    if parts[3] != 0 {
        out.push(enc_movk(rd, parts[3], 48));
    }
}

// === Helpers ===

fn sign_extend(value: i64, bits: u32) -> i64 {
    let shift = 64 - bits;
    (value << shift) >> shift
}
