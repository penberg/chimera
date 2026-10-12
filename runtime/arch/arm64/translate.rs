//! Basic-block translator for AArch64. Decodes a single guest basic block,
//! copies the straight-line prefix into the code cache (with PC-relative
//! operands — ADR, ADRP, LDR-literal — fixed up to refer to the original
//! guest target), and rewrites the terminator into a sequence that:
//!
//! 1. saves the guest's x16 into `ctx.regs[16]`, freeing it as scratch (every
//!    other GPR reaches the exit trampoline in its own register),
//! 2. computes the next guest PC into `ctx.pc`, performing any side effects
//!    the original terminator mandated (e.g., BL writing `next_ip` to x30),
//! 3. branches to the appropriate exit trampoline (`exit_block` for ordinary
//!    block boundaries, `exit_syscall` for SVC, `exit_trap` for BRK).
//!
//! One register is Chimera's inside translated code: `x17` holds the context
//! pointer at every guest-instruction boundary, and the guest's own x17 lives
//! in `ctx.regs[17]`. That is what lets a terminator free a scratch register
//! without first having one — arm64 has no register-free store except
//! relative to `sp`, and writing below the guest's `sp` is exactly what a
//! translator must not do (`testing/conformance/abi/red-zone.c`). A guest
//! instruction that names x17 is wrapped: the guest value is loaded in, the
//! instruction runs, and if it may have written x17 the result is stored back
//! through a register the instruction leaves alone, with the context
//! re-derived from the host thread's TSD slot (see [`emit_load_ctx`]). The
//! `dispatch` trampoline enters every block with `x17 = ctx`, and a linked
//! predecessor arrives the same way, so a block has one entry point.
//!
//! Per-block prologue: every emitted block starts with a load that re-syncs
//! x16 from `ctx.regs[16]`, because `dispatch` and every exit stub clobber x16
//! to carry their branch target.

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
/// `reg` from its reserved pthread TSD slot, reached through `TPIDRRO_EL0` —
/// how the context pointer is re-derived once a guest instruction that names
/// x17 has had the register. Per-thread and signal-safe, and beyond the
/// guest's reach: `TPIDRRO_EL0` is read-only from EL0 and `thread_set_tsd_base`
/// is intercepted, and the slot is one libpthread reserves and never touches
/// (see [`CHIMERA_CTX_TSD_SLOT`]). The slot offset (`slot * 8`) is a
/// compile-time constant baked into the load. (`x18`, the platform's reserved
/// register, would have served without the slot — but libSystem and the kernel
/// clobber it across ordinary guest code.)
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
/// and the host-offset → guest-PC map for fault attribution. `host_pc` is the
/// block's one entry point, for the dispatcher and a linked predecessor alike:
/// both arrive with `x17 = ctx` and x16 clobbered.
pub struct Translation {
    pub host_pc: u64,
    pub host_end: u64,
    pub guest_end: u64,
    /// One entry per guest instruction: the block-relative host word offset
    /// where its expansion begins, paired with its guest PC. Offsets ascend,
    /// so a faulting host PC maps to a guest PC by binary search; host words
    /// before the first entry (the block prologue) belong to the block start.
    pub pc_map: Vec<(u32, u64)>,
    /// This block's outgoing direct edges: for each, the guest PC it targets
    /// and the address of the link slot its exit stub reads. The cache fills
    /// the slot with the target block's entry once that block exists.
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

    // Block prologue: re-sync x16 from ctx.regs[16]. Every entry — from
    // `dispatch`, from a linked predecessor's exit stub, from the inline
    // indirect-branch probe — arrives with x17 = ctx and x16 clobbered.
    out.push(enc_ldr_imm(16, 17, regs_off(16))); // x16 = ctx.regs[16]

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
                match x17_use(insn) {
                    None => out.push(insn),
                    Some(use_) => emit_x17_wrapped(&mut out, insn, use_),
                }
                pc += 4;
                count += 1;
            }
            InsnKind::Adr { rd, imm } => {
                let target = pc.wrapping_add(imm as u64);
                emit_into_guest_reg(&mut out, rd, |out, rd| emit_imm64_compact(out, rd, target));
                pc += 4;
                count += 1;
            }
            InsnKind::Adrp { rd, imm } => {
                let base = pc & !0xFFF;
                let target = base.wrapping_add((imm << 12) as u64);
                emit_into_guest_reg(&mut out, rd, |out, rd| emit_imm64_compact(out, rd, target));
                pc += 4;
                count += 1;
            }
            InsnKind::LdrLiteral {
                rt,
                byte_offset,
                is_64,
            } => {
                let addr = pc.wrapping_add(byte_offset as u64);
                emit_into_guest_reg(&mut out, rt, |out, rt| {
                    emit_imm64_compact(out, rt, addr);
                    if is_64 {
                        out.push(enc_ldr_imm(rt, rt, 0));
                    } else {
                        out.push(enc_ldr32_imm(rt, rt, 0));
                    }
                });
                pc += 4;
                count += 1;
            }
            InsnKind::LdrLiteralFp {
                rt,
                byte_offset,
                kind,
            } => {
                // Replace `ldr <fp>t, label` with the absolute address
                // materialized in x16, which is parked in ctx around it.
                let addr = pc.wrapping_add(byte_offset as u64);
                out.push(enc_str_imm(16, 17, regs_off(16)));
                emit_imm64_compact(&mut out, 16, addr);
                out.push(enc_ldr_fp_imm0(rt, 16, kind));
                out.push(enc_ldr_imm(16, 17, regs_off(16)));
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
                emit_terminator_indirect(&mut out, rn, None, false, exit_tramp, ib_table);
                break;
            }
            InsnKind::BranchLinkReg { rn, next_ip } => {
                emit_terminator_indirect(&mut out, rn, Some(next_ip), false, exit_tramp, ib_table);
                break;
            }
            InsnKind::Ret { rn } => {
                emit_terminator_indirect(&mut out, rn, None, false, exit_tramp, ib_table);
                break;
            }
            InsnKind::PacRet { use_b_key } => {
                let _ = use_b_key;
                emit_terminator_indirect(&mut out, 30, None, true, exit_tramp, ib_table);
                break;
            }
            InsnKind::PacBranchReg { rn } => {
                emit_terminator_indirect(&mut out, rn, None, true, exit_tramp, ib_table);
                break;
            }
            InsnKind::PacBranchLinkReg { rn, next_ip } => {
                emit_terminator_indirect(&mut out, rn, Some(next_ip), true, exit_tramp, ib_table);
                break;
            }
            InsnKind::PacNop => {
                // Emit nothing: the pointer keeps its (unsigned) value.
                pc += 4;
                count += 1;
            }
            InsnKind::MrsTpidrro { rt } => {
                // rt = ctx.guest_tsd, the guest's virtualized TSD base. An xzr
                // destination discards the read; emitting the load would turn
                // xzr into a zero base register, so emit nothing.
                if rt != 31 {
                    emit_into_guest_reg(&mut out, rt, |out, rt| {
                        out.push(enc_ldr_imm(rt, 17, guest_tsd_off()))
                    });
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
    let host_pc = cache.emit(&out)?;
    Ok(Translation {
        host_pc,
        host_end: host_pc + (out.len() as u64) * 4,
        guest_end: pc + 4,
        pc_map,
        edges,
    })
}

// === The reserved register ===

/// How a guest instruction may touch x17, read off its encoding's register
/// fields, or `None` when it cannot name the register at all. The answer errs
/// toward naming it: a field that is a register in one encoding and an
/// immediate in a neighbour is taken as a register, and a destination field
/// that is sometimes flags (`ccmp`) as a write. Either costs a wrapper the
/// instruction did not need, never a value it did: the wrapper loads the
/// guest's x17 in before the instruction regardless, so storing it back after
/// one that did not write it stores what was there.
#[derive(Clone, Copy)]
struct X17Use {
    /// Whether the instruction may write x17, which the wrapper must then
    /// store to `ctx.regs[17]`.
    writes: bool,
}

/// The register fields a load/store encoding may carry: `Rt` (bits 0-4), `Rn`
/// (5-9), `Rt2` (10-14), and `Rm`/`Rs` (16-20).
fn x17_use(insn: u32) -> Option<X17Use> {
    const R: u32 = 17;
    let field = |lo: u32| (insn >> lo) & 0x1F;
    let (rd, rn, ra, rm) = (field(0), field(5), field(10), field(16));
    let (named, writes) = match (insn >> 25) & 0xF {
        // Data-processing immediate: Rd is written, Rn read; extract reads
        // Rm as well. Move-wide and bitfield insert read Rd too, which the
        // unconditional pre-load covers.
        0b1000 | 0b1001 => match (insn >> 23) & 0x7 {
            0b000 | 0b001 | 0b101 => (rd == R, rd == R),
            0b111 => (rd == R || rn == R || rm == R, rd == R),
            _ => (rd == R || rn == R, rd == R),
        },
        // Branches never reach the plain copy (they are terminators); what is
        // left is the system class, whose `mrs`/`msr`/`sys` name Rt.
        0b1010 | 0b1011 => (rd == R, rd == R),
        // Loads and stores.
        0b0100 | 0b0110 | 0b1100 | 0b1110 => {
            // `V` (bit 26) makes Rt (and Rt2) SIMD&FP registers; the base and
            // an index register stay general.
            let v = (insn >> 26) & 1 == 1;
            match (insn >> 27) & 0x7 {
                // Advanced SIMD load/store structures: a GPR base, written
                // back by the post-index forms, which may also index by Rm.
                0b001 if v => {
                    let post = (insn >> 23) & 1 == 1;
                    (rn == R || (post && rm == R), post && rn == R)
                }
                // Exclusives, compare-and-swap, ordered loads and stores:
                // Rt, Rt2 and the Rs status register may all be written.
                0b001 => (
                    rd == R || rn == R || ra == R || rm == R,
                    rd == R || ra == R || rm == R,
                ),
                // Literal loads are rewritten before this; LDAPR/STLR and the
                // memory-copy instructions remain, the latter writing every
                // register they name.
                0b011 => (
                    (!v && rd == R) || rn == R || rm == R,
                    (!v && rd == R) || rn == R || rm == R,
                ),
                // Register pairs: Rt and Rt2, and the base, written back by
                // the pre- and post-index forms (bit 23).
                0b101 => {
                    let writeback = (insn >> 23) & 1 == 1;
                    (
                        (!v && (rd == R || ra == R)) || rn == R,
                        (!v && (rd == R || ra == R)) || (writeback && rn == R),
                    )
                }
                // Single registers: unsigned offset (bit 24), the immediate
                // forms with bits 11:10 selecting unscaled/post/unprivileged/
                // pre (writeback for post and pre), and the bit-21 family —
                // register offset, atomics with an Rs source, PAC-authenticated
                // loads with a writeback option.
                0b111 => {
                    if (insn >> 24) & 1 == 1 {
                        ((!v && rd == R) || rn == R, !v && rd == R)
                    } else if (insn >> 21) & 1 == 0 {
                        let writeback = (insn >> 10) & 1 == 1;
                        (
                            (!v && rd == R) || rn == R,
                            (!v && rd == R) || (writeback && rn == R),
                        )
                    } else {
                        (
                            (!v && rd == R) || rn == R || rm == R,
                            (!v && rd == R) || rn == R,
                        )
                    }
                }
                _ => (
                    rd == R || rn == R || ra == R || rm == R,
                    rd == R || rn == R || ra == R || rm == R,
                ),
            }
        }
        // Data-processing register: Rd written, Rn and Rm read, Ra too for
        // the three-source group (bits 28:24 = 11011).
        0b0101 | 0b1101 => {
            let three_source = (insn >> 24) & 0x1F == 0b11011;
            (
                rd == R || rn == R || rm == R || (three_source && ra == R),
                rd == R,
            )
        }
        // SIMD and floating point: general registers appear only in the
        // conversions and moves between a GPR and a vector register
        // (`sf 0 S 11110 type 1 rmode opcode 000000 Rn Rd`) and in the SIMD
        // copy group (`0 Q op 01110000 imm5 0 imm4 1 Rn Rd`: `dup`, `ins`,
        // `umov`, `smov` with a general operand).
        0b0111 | 0b1111 => {
            let fp_int = insn & 0x5F20_FC00 == 0x1E20_0000;
            let simd_copy = insn & 0x9FE0_8400 == 0x0E00_0400;
            if fp_int || simd_copy {
                (rd == R || rn == R, rd == R)
            } else {
                (false, false)
            }
        }
        // Reserved, SVE and SME: not emitted for this host, so take every
        // field as a register.
        _ => (
            rd == R || rn == R || ra == R || rm == R,
            rd == R || rn == R || ra == R || rm == R,
        ),
    };
    named.then_some(X17Use { writes })
}

/// A register the instruction names in none of its fields, to park across
/// it: the wrapper stores it to its own `ctx` slot before the instruction,
/// uses it as the base that saves the instruction's x17 afterwards, and
/// restores it. Any of the 28 other general registers will do; the lowest is
/// taken.
fn park_register(insn: u32) -> u8 {
    let fields = [
        insn & 0x1F,
        (insn >> 5) & 0x1F,
        (insn >> 10) & 0x1F,
        (insn >> 16) & 0x1F,
    ];
    (0u32..31)
        .filter(|r| !matches!(r, 16 | 17))
        .find(|r| !fields.contains(r))
        .expect("an instruction names at most four registers") as u8
}

/// Emit a guest instruction that names x17, with the guest's x17 in the
/// register while it runs. Before: the guest value replaces the context
/// pointer. After: the context is re-derived from the TSD slot — into x17
/// itself when the instruction only read it, or, when it may have written it,
/// into a parked register that then serves as the base to store the result,
/// so no value is lost for want of a free register.
fn emit_x17_wrapped(out: &mut Vec<u32>, insn: u32, use_: X17Use) {
    if !use_.writes {
        out.push(enc_ldr_imm(17, 17, regs_off(17))); // x17 = guest x17
        out.push(insn);
        emit_load_ctx(out, 17);
        return;
    }
    let park = park_register(insn);
    out.push(enc_str_imm(park, 17, regs_off(park as usize))); // park
    out.push(enc_ldr_imm(17, 17, regs_off(17))); // x17 = guest x17
    out.push(insn);
    emit_load_ctx(out, park); // park = ctx
    out.push(enc_str_imm(17, park, regs_off(17))); // ctx.regs[17] = guest x17
    out.push(enc_mov_reg(17, park)); // x17 = ctx
    out.push(enc_ldr_imm(park, 17, regs_off(park as usize))); // unpark
}

/// Emit a rewrite whose result lands in guest register `rd`. For any register
/// but x17 the rewrite writes it directly. For x17 the value is produced in
/// x16 — parked in its ctx slot around the rewrite — and stored to
/// `ctx.regs[17]`, where the guest's x17 lives.
fn emit_into_guest_reg(out: &mut Vec<u32>, rd: u8, produce: impl FnOnce(&mut Vec<u32>, u8)) {
    if rd != 17 {
        produce(out, rd);
        return;
    }
    out.push(enc_str_imm(16, 17, regs_off(16)));
    produce(out, 16);
    out.push(enc_str_imm(16, 17, regs_off(17)));
    out.push(enc_ldr_imm(16, 17, regs_off(16)));
}

// === Terminator emission ===

/// Save the guest's x16 into `ctx.regs[16]`, freeing it as the stub's scratch.
/// x17 is already the context pointer, so this is the whole bootstrap: one
/// store, nothing on the guest stack.
fn emit_save_x16(out: &mut Vec<u32>) {
    out.push(enc_str_imm(16, 17, regs_off(16)));
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
/// entry once that block is translated (and re-zeroes when it is
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
/// ctx, the guest's x16 saved in ctx — which is exactly what the successor's
/// prologue expects, so the link targets the block entry itself.
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
    emit_save_x16(out);
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
/// x16 and emit its exit tail (linked or plain).
fn cond_arm(next_pc: u64, edge: Edge, exit_tramp: u64) -> Vec<u32> {
    let mut arm = Vec::new();
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
    // The guest's NZCV is still live here: saving x16 leaves the flags
    // untouched, so the condition can be tested directly rather than
    // materialized through `csel`. Each arm then ends in its own exit tail,
    // which is what makes both edges linkable.
    emit_save_x16(out);
    let not_taken_arm = cond_arm(ways.fallthrough, ways.not_taken, exit_tramp);
    let taken_arm = cond_arm(ways.target, ways.taken, exit_tramp);
    out.push(enc_b_cond(cond, not_taken_arm.len() as i32 + 1));
    out.extend_from_slice(&not_taken_arm);
    out.extend_from_slice(&taken_arm);
}

/// Save the guest x16 and name the register the register-test terminators
/// should test. CBZ/TBZ read `rt` and touch no flags. x16 still holds its
/// guest value after the save, so it is tested in place; the guest's x17
/// lives in ctx and is loaded into the freed x16 for the test.
fn emit_test_operand(out: &mut Vec<u32>, rt: u8) -> u8 {
    emit_save_x16(out);
    match rt {
        17 => {
            out.push(enc_ldr_imm(16, 17, regs_off(17)));
            16
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
    let not_taken_arm = cond_arm(ways.fallthrough, ways.not_taken, exit_tramp);
    let taken_arm = cond_arm(ways.target, ways.taken, exit_tramp);
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
    let not_taken_arm = cond_arm(ways.fallthrough, ways.not_taken, exit_tramp);
    let taken_arm = cond_arm(ways.target, ways.taken, exit_tramp);
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

/// `BR`/`BLR`/`RET` and their PAC-authenticating forms (`strip_pac`, which
/// lowers the authentication to `xpaci` on the target — Chimera keeps every
/// pointer unsigned). The target is read from `rn` before anything is
/// clobbered: x16 is its own guest value until the save, and the guest's x17
/// is in ctx.
fn emit_terminator_indirect(
    out: &mut Vec<u32>,
    rn: u8,
    next_ip_for_lr: Option<u64>,
    strip_pac: bool,
    exit_tramp: u64,
    ib_table: u64,
) {
    emit_save_x16(out);
    match rn {
        16 => {}
        17 => out.push(enc_ldr_imm(16, 17, regs_off(17))),
        _ => out.push(enc_mov_reg(16, rn)),
    }
    if strip_pac {
        out.push(enc_xpaci(16));
    }
    // A call sets x30 = next_ip after the target has been read (BLR x30
    // reads the old value). The target is parked in ctx.pc while x16
    // materializes the return address.
    if let Some(next_ip) = next_ip_for_lr {
        out.push(enc_str_imm(16, 17, pc_off()));
        emit_imm64_compact(out, 16, next_ip);
        out.push(enc_mov_reg(30, 16));
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
/// ctx, guest x16 saved in ctx. The probe borrows x0 and x1, parked in their
/// own ctx slots and restored on both paths, and keeps x17 as the context
/// throughout — the successor's prologue expects it.
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

    // The probe proper, with x0 and x1 borrowed as scratch.
    let mut probe = Vec::new();
    probe.push(enc_stp_imm(0, 1, 17, regs_off(0))); // park guest x0, x1
    probe.push(enc_ldr_imm(16, 17, pc_off())); // x16 = target (the poll used x16)
    emit_imm64_fixed(&mut probe, 0, IB_HASH_MULT);
    probe.push(enc_mul(0, 16, 0)); // x0 = target * MULT
    probe.push(enc_lsr_imm(0, 0, 64 - IB_BITS)); // x0 = slot index
    emit_imm64_fixed(&mut probe, 1, ib_table); // x1 = table base
    probe.push(enc_add_lsl(0, 1, 0, 4)); // x0 = &entry (16 bytes each)
    probe.push(enc_ldp_imm(1, 0, 0, 0)); // x1 = key, x0 = host
    probe.push(enc_eor_reg(1, 1, 16)); // zero exactly when the key matches
    // A miss (wrong key, or an empty slot) skips the four hit words —
    // cbz/mov/restore/br — landing on the restore-and-cold tail below.
    probe.push(enc_cbz64(1, true, 5));
    probe.push(enc_cbz64(0, false, 4)); // empty slot -> miss
    probe.push(enc_mov_reg(16, 0)); // x16 = successor entry
    probe.push(enc_ldp_imm(0, 1, 17, regs_off(0))); // restore guest x0, x1
    probe.push(enc_br(16));
    // Miss: restore x0 and x1, then fall into the cold tail.
    probe.push(enc_ldp_imm(0, 1, 17, regs_off(0)));

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
    emit_save_x16(out);
    emit_imm64_compact(out, 16, pc);
    emit_exit_tail(out, trap_exit);
}

/// `SVC`: the syscall number is in the guest's x16, saved like any other
/// terminator's; the resume pc is the next instruction. Nothing touches the
/// guest stack, which is the kernel's own contract for a syscall
/// (`testing/conformance/abi/syscall-no-stack-touch.c`).
fn emit_terminator_svc(out: &mut Vec<u32>, next_ip: u64, syscall_tramp: u64) {
    emit_save_x16(out);
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

/// `ldp <rt1>, <rt2>, [<rn>, #imm]` — 64-bit signed-offset load pair.
fn enc_ldp_imm(rt1: u8, rt2: u8, rn: u8, byte_offset: u32) -> u32 {
    let imm7 = (byte_offset / 8) & 0x7F;
    0xA9400000
        | (imm7 << 15)
        | ((rt2 as u32 & 0x1F) << 10)
        | ((rn as u32 & 0x1F) << 5)
        | (rt1 as u32 & 0x1F)
}

/// `stp <rt1>, <rt2>, [<rn>, #imm]` — 64-bit signed-offset store pair.
fn enc_stp_imm(rt1: u8, rt2: u8, rn: u8, byte_offset: u32) -> u32 {
    let imm7 = (byte_offset / 8) & 0x7F;
    0xA9000000
        | (imm7 << 15)
        | ((rt2 as u32 & 0x1F) << 10)
        | ((rn as u32 & 0x1F) << 5)
        | (rt1 as u32 & 0x1F)
}

/// `b.<cond> <insn_offset>` — conditional branch, offset in instructions.
fn enc_b_cond(cond: u8, insn_offset: i32) -> u32 {
    let imm19 = (insn_offset as u32) & 0x7FFFF;
    0x54000000 | (imm19 << 5) | (cond as u32 & 0xF)
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

fn enc_br(rn: u8) -> u32 {
    0xD61F0000 | ((rn as u32 & 0x1F) << 5)
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
