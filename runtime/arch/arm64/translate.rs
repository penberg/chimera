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
//! Per-block prologue: every emitted block starts with a load that
//! re-syncs x16 from `ctx.regs[16]`, because `dispatch` and every exit stub
//! clobber x16 to carry their branch target. No GPR is reserved across
//! translated code; the context pointer is reached through the
//! `CHIMERA_CTX_PTR` global instead, which translated code dereferences
//! through a short movz/movk + ldr sequence.

use std::{ptr, sync::atomic::Ordering};

use crate::Error;

use super::dispatch::{CHIMERA_CTX_PTR, ThreadState};

const CACHE_SIZE: usize = 16 * 1024 * 1024;
const MAX_BLOCK_GUEST_INSNS: usize = 1024;

/// A bump-allocated JIT region into which `translate()` emits blocks.
pub struct CodeCache {
    base: *mut u8,
    size: usize,
    used: usize,
}

impl CodeCache {
    pub fn new() -> Result<Self, Error> {
        let p = unsafe {
            libc::mmap(
                ptr::null_mut(),
                CACHE_SIZE,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_JIT,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(Error::last_os_error("code cache mmap (MAP_JIT)"));
        }
        Ok(Self {
            base: p as *mut u8,
            size: CACHE_SIZE,
            used: 0,
        })
    }

    fn next_pc(&self) -> u64 {
        (self.base as u64) + self.used as u64
    }

    fn emit_words(&mut self, words: &[u32]) -> Result<(), Error> {
        let bytes = words.len() * 4;
        if self.used + bytes > self.size {
            return Err(Error::CodeCacheExhausted);
        }
        unsafe {
            jit_write_protect(false);
            let dst = self.base.add(self.used) as *mut u32;
            for (i, w) in words.iter().enumerate() {
                ptr::write_unaligned(dst.add(i), *w);
            }
            jit_write_protect(true);
            let start = self.base.add(self.used) as usize;
            let end = start + bytes;
            invalidate_icache(start, end);
        }
        self.used += bytes;
        Ok(())
    }
}

unsafe extern "C" {
    fn pthread_jit_write_protect_np(enabled: libc::c_int);
    fn sys_icache_invalidate(start: *mut libc::c_void, len: libc::size_t);
}

unsafe fn jit_write_protect(enabled: bool) {
    unsafe { pthread_jit_write_protect_np(if enabled { 1 } else { 0 }) };
}

unsafe fn invalidate_icache(start: usize, end: usize) {
    unsafe {
        sys_icache_invalidate(start as *mut libc::c_void, end - start);
    }
}

/// Address of the `CHIMERA_CTX_PTR` global. Captured once per translation
/// so the translator does not have to materialize a Rust-side symbol at
/// every emit site.
fn ctx_global_addr() -> u64 {
    &CHIMERA_CTX_PTR as *const _ as u64
}

/// Translate one basic block starting at `guest_pc`. Returns the host PC at
/// which the translated block begins.
pub fn translate(
    cache: &mut CodeCache,
    guest_pc: u64,
    exit_tramp: u64,
    syscall_tramp: u64,
) -> Result<u64, Error> {
    let host_pc = cache.next_pc();
    let mut out: Vec<u32> = Vec::new();
    let ctx_global = ctx_global_addr();

    // Block prologue: re-sync x16 from ctx.regs[16]. Every entry into a
    // block (from `dispatch` or from an exit stub of a previous block)
    // arrives with x16 clobbered. We load x16 = &CHIMERA_CTX_PTR, then
    // ldr the ctx pointer, then read regs[16] — all using x16 as the
    // base, so no other register is disturbed.
    emit_imm64_fixed(&mut out, 16, ctx_global);
    out.push(enc_ldr_imm(16, 16, 0)); // x16 = ctx pointer
    out.push(enc_ldr_imm(16, 16, regs_off(16))); // x16 = guest x16

    let mut pc = guest_pc;
    let mut count = 0;
    loop {
        if count >= MAX_BLOCK_GUEST_INSNS {
            return Err(Error::Translate(format!(
                "basic block at {:#x} exceeds {} instructions",
                guest_pc, MAX_BLOCK_GUEST_INSNS
            )));
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
                emit_terminator_direct(&mut out, target, None, exit_tramp, ctx_global);
                break;
            }
            InsnKind::BranchLink { target, next_ip } => {
                emit_terminator_direct(&mut out, target, Some(next_ip), exit_tramp, ctx_global);
                break;
            }
            InsnKind::CondBranch {
                target,
                next_ip,
                cond,
            } => {
                emit_terminator_cond(&mut out, cond, target, next_ip, exit_tramp, ctx_global);
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
                    &mut out, rt, sf, nonzero, (target, next_ip), exit_tramp, ctx_global,
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
                    &mut out, rt, bit, nonzero, (target, next_ip), exit_tramp, ctx_global,
                );
                break;
            }
            InsnKind::BranchReg { rn } => {
                emit_terminator_indirect(&mut out, rn, None, exit_tramp, ctx_global);
                break;
            }
            InsnKind::BranchLinkReg { rn, next_ip } => {
                emit_terminator_indirect(&mut out, rn, Some(next_ip), exit_tramp, ctx_global);
                break;
            }
            InsnKind::Ret { rn } => {
                emit_terminator_indirect(&mut out, rn, None, exit_tramp, ctx_global);
                break;
            }
            InsnKind::PacRet { use_b_key } => {
                let _ = use_b_key;
                out.push(enc_xpaci(30));
                emit_terminator_indirect(&mut out, 30, None, exit_tramp, ctx_global);
                break;
            }
            InsnKind::PacBranchReg { rn } => {
                out.push(enc_xpaci(rn));
                emit_terminator_indirect(&mut out, rn, None, exit_tramp, ctx_global);
                break;
            }
            InsnKind::PacBranchLinkReg { rn, next_ip } => {
                out.push(enc_xpaci(rn));
                emit_terminator_indirect(&mut out, rn, Some(next_ip), exit_tramp, ctx_global);
                break;
            }
            InsnKind::Svc { next_ip } => {
                emit_terminator_svc(&mut out, next_ip, syscall_tramp, ctx_global);
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

    cache.emit_words(&out)?;
    Ok(host_pc)
}

// === Terminator emission ===

/// Push the guest x16/x17 to the guest stack, then load the ctx pointer
/// into x17. After this sequence, x16 and x17 are scratch and `[sp, #0]`
/// / `[sp, #8]` hold the guest values of x16 / x17.
fn emit_save_x16_x17_and_load_ctx(out: &mut Vec<u32>, ctx_global: u64) {
    // stp x16, x17, [sp, #-16]!
    out.push(enc_stp_pre_index(16, 17, 31, -16));
    // x17 = ctx pointer (load through &CHIMERA_CTX_PTR).
    emit_imm64_fixed(out, 17, ctx_global);
    out.push(enc_ldr_imm(17, 17, 0));
}

/// Final tail of every exit stub: store the new pc into ctx.pc using x17
/// as the ctx base, materialize the exit trampoline into x16, and branch.
fn emit_exit_tail(out: &mut Vec<u32>, exit_tramp: u64) {
    out.push(enc_str_imm(16, 17, pc_off())); // ctx.pc = x16
    emit_imm64_fixed(out, 16, exit_tramp);
    out.push(enc_br(16));
}

fn emit_terminator_direct(
    out: &mut Vec<u32>,
    target: u64,
    next_ip_for_lr: Option<u64>,
    exit_tramp: u64,
    ctx_global: u64,
) {
    emit_save_x16_x17_and_load_ctx(out, ctx_global);
    if let Some(next_ip) = next_ip_for_lr {
        // x30 = next_ip (BL semantics: discard guest x30 and write next_ip)
        emit_imm64_compact(out, 16, next_ip);
        out.push(enc_mov_reg(30, 16));
    }
    emit_imm64_compact(out, 16, target);
    emit_exit_tail(out, exit_tramp);
}

fn emit_terminator_cond(
    out: &mut Vec<u32>,
    cond: u8,
    target: u64,
    fallthrough: u64,
    exit_tramp: u64,
    ctx_global: u64,
) {
    emit_save_x16_x17_and_load_ctx(out, ctx_global);
    // Build the two candidate next-PCs in x16, x17. We just clobbered x17
    // with the ctx pointer, but we don't need it again until the tail —
    // and the tail recomputes it from `ctx` already-stored values, so we
    // can repurpose x17 here. Actually we still need x17 = ctx for the
    // str at the end. So use a different scratch arrangement: build
    // target/fallthrough into x16 with csel from a stack-stored alternative.
    //
    // Easier: emit_imm64_fixed each into a single register sequentially.
    // We use x16 for one, and a temp on the guest stack for the other.
    // But that's clumsy. A cleaner option: do the comparison FIRST (NZCV
    // already reflects the guest's flags) by computing both candidates
    // into the pair (x16, scratch on stack), then csel.
    //
    // Cleanest: build target into x16, then push it, build fallthrough
    // into x16, csel between [sp] and x16 — no, csel doesn't take memory.
    //
    // Fall back to using x0 as a transient scratch: save it onto the
    // guest stack, build the second candidate into it, csel, restore.
    // That keeps x17 free for ctx.
    //
    // Layout:
    //   str x0, [sp, #-16]!         ; spill guest x0 (16-byte aligned)
    //   movz/movk x16, target
    //   movz/movk x0,  fallthrough
    //   csel x16, x16, x0, cond     ; reads NZCV
    //   ldr x0, [sp], #16           ; restore guest x0
    //   (now ctx in x17, target in x16, tail proceeds)
    out.push(enc_str_pre_index(0, 31, -16));
    emit_imm64_compact_or_pad(out, 16, target);
    emit_imm64_compact_or_pad(out, 0, fallthrough);
    out.push(enc_csel(16, 16, 0, cond));
    out.push(enc_ldr_post_index(0, 31, 16));
    emit_exit_tail(out, exit_tramp);
}

fn emit_terminator_cbz(
    out: &mut Vec<u32>,
    rt: u8,
    sf: bool,
    nonzero: bool,
    (target, fallthrough): (u64, u64),
    exit_tramp: u64,
    ctx_global: u64,
) {
    // CBZ/CBNZ don't modify NZCV. We must read `rt` BEFORE clobbering it
    // with our scratch usage. The cleanest sequence:
    //
    //   stp x16, x17, [sp, #-16]!     ; save guest x16, x17 (which we'll
    //                                    clobber). rt's value is still
    //                                    live in its host register, even
    //                                    if rt is 16 or 17 — stp only
    //                                    stores, it doesn't modify.
    //   cb(n)z xRt, +<to taken>       ; branch within the stub
    //   ... build x16 = fallthrough ...
    //   b +<past taken>
    //   ... build x16 = target ...
    //   ... load ctx and tail ...
    //
    // The CB(N)Z is encoded with imm19 in instruction units; we use a
    // fixed-width emit_imm64 (4 movz/movk) so both branches predict to
    // the same offsets.
    out.push(enc_stp_pre_index(16, 17, 31, -16));

    // If rt is 16 or 17, the guest value we want to test is now on the
    // stack — we must reload it into a scratch register first. Use x17
    // as scratch (it's already saved on the stack at [sp, #8]).
    let test_reg = if rt == 16 {
        out.push(enc_ldr_imm(17, 31, 0)); // ldr x17, [sp]
        17u8
    } else if rt == 17 {
        out.push(enc_ldr_imm(17, 31, 8)); // ldr x17, [sp, #8]
        17u8
    } else {
        rt
    };

    // After this point, the fixed-width emit_imm64 ensures both branches
    // of the diamond are the same size (4 instructions for each operand
    // path: 4 movz/movk).
    //
    //   I0: cb(n)z xRt, +6        ; if taken, jump 6 instructions ahead
    //   I1..I4: movz/movk x16, fallthrough   (4 instrs)
    //   I5: b +5                  ; skip the taken path
    //   I6..I9: movz/movk x16, target        (4 instrs)
    //   I10..: tail
    //
    // CB(N)Z imm19 is in instruction units. From I0 to I6 is +6.
    // From I5 to "after I9" is +5 (5 instructions later = I10).
    let cbz = if sf {
        enc_cbz64(test_reg, nonzero, 6)
    } else {
        enc_cbz32(test_reg, nonzero, 6)
    };
    out.push(cbz);
    emit_imm64_padded(out, 16, fallthrough, 4);
    out.push(enc_b_imm(5));
    emit_imm64_padded(out, 16, target, 4);

    // Reload ctx into x17 (we may have clobbered it above when rt == 16/17).
    emit_imm64_fixed(out, 17, ctx_global);
    out.push(enc_ldr_imm(17, 17, 0));
    emit_exit_tail(out, exit_tramp);
}

fn emit_terminator_tbz(
    out: &mut Vec<u32>,
    rt: u8,
    bit: u8,
    nonzero: bool,
    (target, fallthrough): (u64, u64),
    exit_tramp: u64,
    ctx_global: u64,
) {
    // Same pattern as CBZ but with TBZ/TBNZ. TBZ's imm14 is in
    // instruction units (signed 14-bit) — plenty of range for our stub.
    out.push(enc_stp_pre_index(16, 17, 31, -16));

    let test_reg = if rt == 16 {
        out.push(enc_ldr_imm(17, 31, 0));
        17u8
    } else if rt == 17 {
        out.push(enc_ldr_imm(17, 31, 8));
        17u8
    } else {
        rt
    };

    out.push(enc_tbz(test_reg, bit, nonzero, 6));
    emit_imm64_padded(out, 16, fallthrough, 4);
    out.push(enc_b_imm(5));
    emit_imm64_padded(out, 16, target, 4);

    emit_imm64_fixed(out, 17, ctx_global);
    out.push(enc_ldr_imm(17, 17, 0));
    emit_exit_tail(out, exit_tramp);
}

fn emit_terminator_indirect(
    out: &mut Vec<u32>,
    rn: u8,
    next_ip_for_lr: Option<u64>,
    exit_tramp: u64,
    ctx_global: u64,
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
    emit_imm64_fixed(out, 17, ctx_global);
    out.push(enc_ldr_imm(17, 17, 0));

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

    emit_exit_tail(out, exit_tramp);
}

fn emit_terminator_svc(out: &mut Vec<u32>, next_ip: u64, syscall_tramp: u64, ctx_global: u64) {
    // SVC: real kernels switch to their own kernel stack and never touch
    // the user stack — `testing/conformance/abi/syscall-no-stack-touch.c`
    // exercises exactly this property. Unlike the generic `B`/`BL` exit
    // paths (which spill x16/x17 to the guest stack), the SVC stub stores
    // the syscall number directly into `ctx.regs[16]` and clobbers x17 as
    // the ctx scratch. Discarding x17 is fine: AArch64 PCS marks it as
    // IP1, an inter-procedure-call scratch that callees and the kernel
    // are free to clobber.
    //
    //   movz/movk x17, &CHIMERA_CTX_PTR   ; 4 instrs to materialise the
    //                                       global's address
    //   ldr  x17, [x17, #0]                ; x17 = ctx pointer
    //   str  x16, [x17, #regs_off(16)]    ; ctx.regs[16] = syscall number
    //   movz/movk x16, #next_ip
    //   str  x16, [x17, #pc_off]           ; ctx.pc = resume pc
    //   movz/movk x16, #syscall_tramp
    //   br   x16
    emit_imm64_fixed(out, 17, ctx_global);
    out.push(enc_ldr_imm(17, 17, 0));
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
    InsnKind::Other
}

// === Encoders ===

const fn regs_off(idx: usize) -> u32 {
    (std::mem::offset_of!(ThreadState, regs) + idx * 8) as u32
}

const fn pc_off() -> u32 {
    std::mem::offset_of!(ThreadState, pc) as u32
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

fn enc_csel(rd: u8, rn: u8, rm: u8, cond: u8) -> u32 {
    0x9A800000
        | ((rm as u32 & 0x1F) << 16)
        | ((cond as u32 & 0xF) << 12)
        | ((rn as u32 & 0x1F) << 5)
        | (rd as u32 & 0x1F)
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

/// Wrapper that picks the right variant based on need; currently this
/// just defers to `emit_imm64_compact`, but a future revision may
/// optimise differently.
fn emit_imm64_compact_or_pad(out: &mut Vec<u32>, rd: u8, val: u64) {
    emit_imm64_compact(out, rd, val);
}

// === Helpers ===

fn sign_extend(value: i64, bits: u32) -> i64 {
    let shift = 64 - bits;
    (value << shift) >> shift
}

// Silence dead-code warnings for the import that is only used in
// `Ordering::Relaxed` typing — the static is exported for the asm side.
#[allow(dead_code)]
fn _force_ordering_used() {
    let _ = Ordering::Relaxed;
}
