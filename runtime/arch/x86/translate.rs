//! Basic-block translator. Decodes a single guest basic block, copies the
//! straight-line prefix into the code cache (with RIP-relative operands
//! fixed up by `BlockEncoder`), and rewrites the terminator into a
//! "compute next guest PC, then exit to the dispatcher" sequence.

use std::{mem::offset_of, ptr};

use iced_x86::{
    BlockEncoder, BlockEncoderOptions, Code, Decoder, DecoderOptions, FlowControl, Instruction,
    InstructionBlock, MemoryOperand, OpKind, Register,
};

use crate::Error;

use super::dispatch::ThreadState;

const MAX_BLOCK_GUEST_BYTES: usize = 4096;

/// Inline indirect-branch lookup table: a direct-mapped, guest-readable mirror
/// of the guest-PC -> host-PC map, probed from the code cache so that `ret`,
/// indirect `call`, and indirect `jmp` can stay in the cache on a hit instead
/// of round-tripping through the dispatcher. Each slot is `{guest_pc, host_pc}`
/// (two `u64`s). It is a prediction cache, not authoritative: a miss or a
/// collision simply falls back to the dispatcher, which re-inserts the entry.
const IB_SLOTS: usize = 1 << 17;
const IB_SLOT_BYTES: usize = 16;
const IB_TABLE_BYTES: usize = IB_SLOTS * IB_SLOT_BYTES;
const IB_BITS: u32 = IB_SLOTS.trailing_zeros();

/// Multiplier for the Fibonacci hash that maps a guest PC to a table slot. The
/// slot index is the top [`IB_BITS`] bits of `guest_pc * IB_HASH_MULT`. A plain
/// `(guest_pc >> k) & mask` would be cheaper, but any fixed right-shift folds
/// every PC within one 2^k-byte window onto the same slot — and indirect
/// targets routinely land that close (a polymorphic `ret` returns just past
/// each of several adjacent call sites). Those neighbours would then evict each
/// other on every branch, so the table would never hit. The multiply mixes the
/// low bits into the index, so PCs a few bytes apart take different slots.
const IB_HASH_MULT: u64 = 0x9e37_79b9_7f4a_7c15;
/// Empty-slot fill byte. "Not a real guest PC" is not enough to make a slot
/// unmatchable — a `ret` or indirect operand can present any 64-bit value,
/// canonical or not — so [`CodeCache::clear_ib_table`] additionally repairs
/// the one slot the all-`0xff` key hashes to (see [`ib_unmatchable`]).
const IB_EMPTY: u8 = 0xff;

/// `gs:[]` displacement of the guest's rbx slot (`regs[1]`). Terminators that
/// need a scratch memory slot borrow it: `exit_block` re-saves the live rbx
/// over the slot on the way out, so the guest's rbx register is preserved.
const RBX_SLOT: i64 = 8;

/// A bump-allocated RWX region into which `translate()` emits blocks, paired
/// with the inline indirect-branch lookup table and the shared lookup routine.
pub struct CodeCache {
    base: *mut u8,
    size: usize,
    used: usize,
    /// Direct-mapped indirect-branch table (see [`IB_SLOTS`]). A separate RW
    /// mapping at a fixed address, baked into the lookup routine as an
    /// immediate.
    ib_table: *mut u8,
    /// Host address of the shared lookup routine, emitted lazily into the code
    /// region on first use and re-emitted after [`CodeCache::reset`].
    ib_lookup: Option<u64>,
}

impl CodeCache {
    /// Create a code cache backed by a `size`-byte RWX region. The region is
    /// `mmap`'d lazily, so unused capacity costs virtual address space, not
    /// resident memory. `size` must stay under 2 GiB so every intra-cache
    /// `rel32` branch displacement fits in an `i32` (see `patch_site` in the
    /// block cache); [`Sandbox::run`](crate::Sandbox::run) rejects an
    /// out-of-range size before any guest state exists, so by the time a
    /// cache is built the bound is an invariant.
    pub fn new(size: usize) -> Result<Self, Error> {
        assert!(
            size > 0 && size <= crate::MAX_CODE_CACHE_SIZE,
            "code cache size {size} out of range"
        );
        let p = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(Error::last_os_error("code cache mmap"));
        }
        let t = unsafe {
            libc::mmap(
                ptr::null_mut(),
                IB_TABLE_BYTES,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if t == libc::MAP_FAILED {
            let err = Error::last_os_error("ib table mmap");
            unsafe { libc::munmap(p, size) };
            return Err(err);
        }
        let cache = Self {
            base: p as *mut u8,
            size,
            used: 0,
            ib_table: t as *mut u8,
            ib_lookup: None,
        };
        cache.clear_ib_table();
        Ok(cache)
    }

    fn next_pc(&self) -> u64 {
        (self.base as u64) + self.used as u64
    }

    fn emit(&mut self, bytes: &[u8]) -> Result<(), Error> {
        if self.used + bytes.len() > self.size {
            return Err(Error::CodeCacheExhausted);
        }
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(self.used), bytes.len());
        }
        self.used += bytes.len();
        Ok(())
    }

    /// Record a guest-PC -> host-PC mapping in the inline lookup table. Direct
    /// mapped, so this overwrites any prior occupant of the slot; the evicted
    /// entry just falls back to the dispatcher (and is re-inserted) next time.
    pub fn ib_insert(&mut self, guest_pc: u64, host_pc: u64) {
        let slot = ib_slot(guest_pc) * IB_SLOT_BYTES;
        unsafe {
            let key = self.ib_table.add(slot) as *mut u64;
            let host = key.add(1);
            // Torn-read-safe publish for the lock-free inline lookup:
            // invalidate the key, write the host PC, then write the real key
            // last. A reader that matches the key — and re-checks it after
            // loading the host (see `ensure_ib_lookup`) — therefore always pairs
            // the key with its own host PC, never one from a different
            // generation. Inserts are serialized under the address-space lock and
            // x86 keeps stores ordered, so the three writes publish in order.
            // `write_volatile` keeps the compiler from coalescing or reordering
            // them. The transient key must be unmatchable by structure, not by
            // value — a guest can synthesize any 64-bit branch target, so no
            // reserved constant is safe (see `ib_unmatchable`).
            key.write_volatile(ib_unmatchable(guest_pc));
            host.write_volatile(host_pc);
            key.write_volatile(guest_pc);
        }
    }

    fn clear_ib_table(&self) {
        unsafe {
            ptr::write_bytes(self.ib_table, IB_EMPTY, IB_TABLE_BYTES);
            // The all-0xff key is matchable in the single slot that u64::MAX
            // hashes to: a lookup for target 0xffff_ffff_ffff_ffff would "hit"
            // there and jump to the all-0xff host word. Repair that slot's key
            // so every empty slot fails the compare.
            let slot = ib_slot(u64::MAX) * IB_SLOT_BYTES;
            (self.ib_table.add(slot) as *mut u64).write(ib_unmatchable(u64::MAX));
        }
    }

    /// Emit the shared inline indirect-branch lookup routine (once), returning
    /// its host address. Translated indirect branches reach it via
    /// `jmp gs:[ib_lookup]` with the guest target in `rax` and the guest's rax
    /// already saved at `gs:[0]`; the guest's flags and all other registers are
    /// still live.
    ///
    /// The routine preserves the guest's flags around its own arithmetic
    /// (`lahf`/`seto` on entry, `add al,0x7f`/`sahf` on exit, via the `ib_flags`
    /// slot) and borrows rcx/rdx through the `ib_rcx`/`ib_rdx` slots. It hashes
    /// the target, reads the one direct-mapped table slot, and on a guest-PC
    /// match restores every register and jumps straight to the cached host PC.
    /// On a mismatch it restores state, writes the target into the `rip` slot,
    /// and falls through to the normal block-exit trampoline, exactly as the
    /// pre-lookup terminator did — so a miss is transparent and re-inserts the
    /// entry on the way back through the dispatcher.
    pub fn ensure_ib_lookup(&mut self, exit_tramp: u64) -> Result<u64, Error> {
        if let Some(addr) = self.ib_lookup {
            return Ok(addr);
        }
        let d_rax = 0i32;
        let d_rip = offset_of!(ThreadState, rip) as i32;
        let d_flags = offset_of!(ThreadState, ib_flags) as i32;
        let d_target = offset_of!(ThreadState, ib_target) as i32;
        let d_rcx = offset_of!(ThreadState, ib_rcx) as i32;
        let d_rdx = offset_of!(ThreadState, ib_rdx) as i32;
        let d_host = offset_of!(ThreadState, ib_host) as i32;
        let table = self.ib_table as u64;

        let mut out = Vec::new();
        // Stash the target and save the guest's status flags before any
        // flag-clobbering arithmetic. lahf/seto write into rax (AH/AL), so the
        // target is moved out to its slot first.
        gs_store(&mut out, MODRM_RAX, d_target); // mov gs:[target], rax
        out.push(0x9f); // lahf
        out.extend_from_slice(&[0x0f, 0x90, 0xc0]); // seto al
        gs_store(&mut out, MODRM_RAX, d_flags); // mov gs:[flags], rax
        // Borrow rcx (target) and rdx (slot index) as scratch.
        gs_store(&mut out, MODRM_RCX, d_rcx); // mov gs:[rcx], rcx
        gs_store(&mut out, MODRM_RDX, d_rdx); // mov gs:[rdx], rdx
        gs_load(&mut out, MODRM_RCX, d_target); // mov rcx, gs:[target]
        // Fibonacci hash: slot = (target * IB_HASH_MULT) >> (64 - IB_BITS),
        // matching `ib_slot`. The multiply mixes the low PC bits in, so nearby
        // indirect targets do not alias onto the same slot (see IB_HASH_MULT).
        out.extend_from_slice(&[0x48, 0x89, 0xca]); // mov rdx, rcx
        movabs_rax(&mut out, IB_HASH_MULT); // movabs rax, IB_HASH_MULT
        out.extend_from_slice(&[0x48, 0x0f, 0xaf, 0xd0]); // imul rdx, rax
        out.extend_from_slice(&[0x48, 0xc1, 0xea, (64 - IB_BITS) as u8]); // shr rdx, 64-IB_BITS
        out.extend_from_slice(&[0x48, 0xc1, 0xe2, 0x04]); // shl rdx, 4  (*16)
        movabs_rax(&mut out, table); // movabs rax, table_base
        out.extend_from_slice(&[0x48, 0x01, 0xd0]); // add rax, rdx -> &slot
        out.extend_from_slice(&[0x48, 0x3b, 0x08]); // cmp rcx, [rax]
        out.extend_from_slice(&[0x0f, 0x85]); // jne miss
        let jne_rel = take_rel32(&mut out);

        // Asynchronous-signal safepoint poll. Every guest loop that stays in the
        // cache via an indirect branch (a computed-goto or function-pointer
        // dispatch loop hitting this table) closes here, so without a poll a
        // pending signal would never reach a block boundary. If the exit flag is
        // set, divert to the miss path, which already publishes the resolved target
        // (gs:[target]) as the next guest PC and returns to the dispatcher —
        // delivering at a clean boundary with the right guest PC. The guest's flags
        // are saved in gs:[ib_flags] here (restored on both the hit and miss
        // paths), so this `cmp`'s flag clobber is harmless.
        out.extend_from_slice(&[0x65, 0x83, 0x3c, 0x25]); // cmp dword ptr gs:[exit_requested], 0
        emit_u32(&mut out, offset_of!(ThreadState, exit_requested) as u32);
        out.push(0x00);
        out.extend_from_slice(&[0x0f, 0x85]); // jne miss
        let poll_rel = take_rel32(&mut out);

        // Hit: load the host PC, restore flags and the borrowed registers, and
        // jump into the successor block with the full guest register file live.
        out.extend_from_slice(&[0x48, 0x8b, 0x50, 0x08]); // mov rdx, [rax+8]
        gs_store(&mut out, MODRM_RDX, d_host); // mov gs:[host], rdx
        // Re-check the key after reading the host: if the slot was
        // republished out from under us (a colliding insert landing between the
        // first key compare and the host load), the host could be from a
        // different generation — treat it as a miss. rcx still holds the target
        // and rax still points at the slot; rdx is already saved to gs:[host].
        out.extend_from_slice(&[0x48, 0x3b, 0x08]); // cmp rcx, [rax]
        out.extend_from_slice(&[0x0f, 0x85]); // jne miss
        let jne2_rel = take_rel32(&mut out);
        emit_restore_flags(&mut out, d_flags);
        gs_load(&mut out, MODRM_RCX, d_rcx); // mov rcx, gs:[rcx]
        gs_load(&mut out, MODRM_RDX, d_rdx); // mov rdx, gs:[rdx]
        gs_load(&mut out, MODRM_RAX, d_rax); // mov rax, gs:[0]  (guest rax)
        out.extend_from_slice(&[0x65, 0xff, 0x24, 0x25]); // jmp gs:[host]
        emit_u32(&mut out, d_host as u32);

        // Miss: restore flags and registers, publish the target as the next
        // guest PC, and exit to the dispatcher exactly as before. Both the
        // key-compare and the post-host recheck branch here.
        let miss = out.len();
        write_rel32(&mut out, jne_rel, miss);
        write_rel32(&mut out, poll_rel, miss);
        write_rel32(&mut out, jne2_rel, miss);
        emit_restore_flags(&mut out, d_flags);
        gs_load(&mut out, MODRM_RCX, d_rcx); // mov rcx, gs:[rcx]
        gs_load(&mut out, MODRM_RDX, d_rdx); // mov rdx, gs:[rdx]
        gs_load(&mut out, MODRM_RAX, d_target); // mov rax, gs:[target]
        gs_store(&mut out, MODRM_RAX, d_rip); // mov gs:[rip], rax
        movabs_rax(&mut out, exit_tramp); // movabs rax, exit_block
        out.extend_from_slice(&[0xff, 0xe0]); // jmp rax

        let addr = self.next_pc();
        self.emit(&out)?;
        self.ib_lookup = Some(addr);
        Ok(addr)
    }

    pub fn reset(&mut self) {
        self.used = 0;
        self.ib_lookup = None;
        self.clear_ib_table();
    }
}

impl Drop for CodeCache {
    fn drop(&mut self) {
        let ret = unsafe { libc::munmap(self.base.cast(), self.size) };
        debug_assert_eq!(ret, 0, "code cache munmap failed");
        let ret = unsafe { libc::munmap(self.ib_table.cast(), IB_TABLE_BYTES) };
        debug_assert_eq!(ret, 0, "ib table munmap failed");
    }
}

/// Map a guest PC to its direct-mapped slot index in the indirect-branch table.
/// Must stay in lockstep with the hash the inline lookup routine computes in
/// [`CodeCache::ensure_ib_lookup`].
fn ib_slot(guest_pc: u64) -> usize {
    (guest_pc.wrapping_mul(IB_HASH_MULT) >> (64 - IB_BITS)) as usize
}

/// A key value the inline lookup can never match in `key`'s own slot. A reader
/// only compares its target against the one slot the target hashes to, so a
/// marker is unmatchable iff it does not hash to the slot it is stored in.
/// Flipping bit 63 guarantees that: `x ^ 2^63 = x ± 2^63`, and since
/// [`IB_HASH_MULT`] is odd, `±2^63 * IB_HASH_MULT ≡ 2^63 (mod 2^64)` — the
/// hash product's top bit flips while the lower bits are untouched, and with
/// it the top bit of the [`IB_BITS`]-bit slot index. This holds for every
/// 64-bit value, unlike any reserved constant, which a guest could present as
/// a branch target.
fn ib_unmatchable(key: u64) -> u64 {
    key ^ (1 << 63)
}

/// `gs:[disp]` with a 32-bit displacement, qword-sized.
fn gs_qword(disp: i64) -> MemoryOperand {
    MemoryOperand::new(
        Register::None,
        Register::None,
        1,
        disp,
        4,
        false,
        Register::GS,
    )
}

/// A patchable outgoing edge of a translated block: the guest PC of a
/// statically known successor, and the address of the `rel32` displacement
/// field of the direct branch that currently targets the block's cold exit
/// stub. Once the successor is translated, the dispatcher rewrites the
/// displacement so the branch jumps straight into the successor's host code,
/// keeping the guest register file live across the edge instead of
/// round-tripping through the dispatcher. See [`super::super::sys::mmap`].
pub struct OutEdge {
    pub target_guest: u64,
    pub site: usize,
}

/// Translate one basic block starting at `guest_pc`. Returns the host PC at
/// which the translated block begins, together with the block's statically
/// known outgoing edges (empty for blocks ending in an indirect branch,
/// return, or syscall) for the dispatcher to link.
pub fn translate(
    cache: &mut CodeCache,
    guest_pc: u64,
    exit_tramp: u64,
    syscall_tramp: u64,
) -> Result<(u64, Vec<OutEdge>), Error> {
    let host_pc = cache.next_pc();
    let guest_bytes =
        unsafe { std::slice::from_raw_parts(guest_pc as *const u8, MAX_BLOCK_GUEST_BYTES) };
    let mut decoder = Decoder::with_ip(64, guest_bytes, guest_pc, DecoderOptions::NONE);
    let mut instrs = Vec::new();
    let mut instr = Instruction::default();

    let term = loop {
        if !decoder.can_decode() {
            return Err(Error::Translate(format!(
                "decoder ran out of bytes at {:#x}",
                guest_pc
            )));
        }
        decoder.decode_out(&mut instr);
        if matches!(instr.flow_control(), FlowControl::Next) {
            instrs.push(instr);
            continue;
        }
        break instr;
    };

    rewrite_rip_relative_leas(&mut instrs)?;

    // A terminator with statically known target(s) — a direct jmp, a direct
    // call, or a supported conditional branch — gets the linkable layout: the
    // straight-line body, then a fast-path direct branch (initially aimed at a
    // cold exit stub) that the dispatcher later back-patches to the successor.
    // Everything else (indirect branches/calls, returns, syscalls, and the few
    // unsupported conditional forms) keeps the original "compute next guest PC,
    // exit to dispatcher" terminator and contributes no links.
    if let Some(link) = classify_terminator(&term) {
        emit_body(cache, &instrs, host_pc, guest_pc)?;
        let term_pc = cache.next_pc() as usize;
        let (bytes, rel_edges) = build_linked_terminator(&link, exit_tramp, guest_pc, term_pc);
        cache.emit(&bytes)?;
        let edges = rel_edges
            .into_iter()
            .map(|(off, target_guest)| OutEdge {
                target_guest,
                site: term_pc + off,
            })
            .collect();
        Ok((host_pc, edges))
    } else {
        emit_terminator(&mut instrs, &term, syscall_tramp)?;
        emit_body(cache, &instrs, host_pc, guest_pc)?;
        Ok((host_pc, Vec::new()))
    }
}

/// Rewrite each `lea reg, [rip + disp]` in the block body into a `movabs reg,
/// <absolute guest target>` that materializes the same guest address.
///
/// `BlockEncoder` fixes up a RIP-relative operand by preserving its *effective
/// address* — except when that address falls inside the very block being encoded,
/// in which case it relocates the operand to the target's NEW location in the code
/// cache. A computed `goto` compiles to `lea &&label(%rip)` with the label in the
/// same basic block, so the encoder would hand back a code-cache address. Chimera
/// needs the guest address instead: the value flows into an indirect branch that
/// the dispatcher resolves as a guest PC. Materializing the absolute guest target
/// up front sidesteps the relocation entirely (and produces the identical result
/// for an ordinary out-of-block `lea`, so there is no behavior change there).
///
/// Only `lea r64, [rip+...]` is rewritten — the form compilers emit to take the
/// address of code or data (computed gotos, function pointers, PIC). Other
/// instructions reading RIP-relative *data* address memory outside the block, so
/// the encoder preserves their target correctly and they are left untouched.
fn rewrite_rip_relative_leas(instrs: &mut [Instruction]) -> Result<(), Error> {
    for instr in instrs.iter_mut() {
        if instr.code() == Code::Lea_r64_m && instr.is_ip_rel_memory_operand() {
            let dest = instr.op0_register();
            let target = instr.ip_rel_memory_address();
            *instr = mkinstr(Instruction::with2(Code::Mov_r64_imm64, dest, target))?;
        }
    }
    Ok(())
}

/// Encode the straight-line instruction list at `host_pc` (with RIP-relative
/// operands fixed up by `BlockEncoder`) and append it to the cache. A block
/// whose first instruction is the terminator has an empty body and emits
/// nothing here.
fn emit_body(
    cache: &mut CodeCache,
    instrs: &[Instruction],
    host_pc: u64,
    guest_pc: u64,
) -> Result<(), Error> {
    if instrs.is_empty() {
        return Ok(());
    }
    let block = InstructionBlock::new(instrs, host_pc);
    let result = BlockEncoder::encode(64, block, BlockEncoderOptions::NONE)
        .map_err(|e| Error::Translate(format!("encode block at {:#x}: {}", guest_pc, e)))?;
    cache.emit(&result.code_buffer)
}

/// `gs:[]` displacement of the guest's rax slot (`regs[0]`); cold exit stubs
/// save the live rax here for the dispatcher, matching `emit_save_rax`.
const RAX_SLOT: i32 = 0;
/// `gs:[]` displacement of the `rip` slot the exit trampoline resumes from.
const RIP_SLOT: i32 = 128;

/// A terminator whose successor PCs are known at translation time, and can
/// therefore be linked directly to its successor blocks once they exist.
enum LinkTerm {
    /// `jmp rel`: one direct successor.
    Uncond { target: u64 },
    /// `jcc rel`: the taken target and the fall-through, selected by the
    /// guest's live flags. `opcode` is the second byte of the `0F 8x` form.
    Cond {
        opcode: u8,
        taken: u64,
        fallthrough: u64,
    },
    /// `call rel`: pushes the return address (`ret`), then jumps to `target`.
    DirectCall { target: u64, ret: u64 },
}

/// Classify a terminator as linkable, or `None` if it ends the block with a
/// runtime-determined target (indirect branch/call, return), is a syscall, or
/// is a conditional form this translator does not lower (`loop`, `jrcxz`).
fn classify_terminator(t: &Instruction) -> Option<LinkTerm> {
    if t.code() == Code::Syscall {
        return None;
    }
    match t.flow_control() {
        FlowControl::UnconditionalBranch => Some(LinkTerm::Uncond {
            target: t.near_branch_target(),
        }),
        FlowControl::ConditionalBranch => Some(LinkTerm::Cond {
            opcode: jcc_opcode(t.code())?,
            taken: t.near_branch_target(),
            fallthrough: t.next_ip(),
        }),
        FlowControl::Call => Some(LinkTerm::DirectCall {
            target: t.near_branch_target(),
            ret: t.next_ip(),
        }),
        _ => None,
    }
}

/// Whether an edge to `target` from a block starting at `block_start` closes a
/// loop: its successor lies at or before the block's own start, so it is a
/// backward edge. Only such edges carry the asynchronous-signal safepoint poll;
/// forward (straight-line) edges stay poll-free so non-looping code is
/// unaffected. Every guest loop closes via a direct back-branch (handled here)
/// or an indirect branch (handled in the shared inline lookup routine).
fn is_back_edge(target: u64, block_start: u64) -> bool {
    target <= block_start
}

/// Emit the asynchronous-signal safepoint poll for a loop-closing edge. Returns
/// the offset of a reserved `rel32` the caller patches to the edge's cold-exit
/// stub: when `exit_requested` is set the poll branches there (the stub publishes
/// the successor guest PC and exits to the dispatcher, where the pending signal is
/// delivered); when clear it falls through to the caller's fast-path branch. On
/// both paths the borrowed register is restored, so the full guest register file
/// stays live across the edge.
///
/// The poll must not perturb the guest's arithmetic flags: a valid loop can carry
/// a flag across its back-edge — an `adc`/`sbb` reduction closed by `dec`/`jnz`
/// relies on `dec` preserving CF for the next iteration's `adc` — so a
/// flag-clobbering `cmp` would make such a loop misexecute the moment it is
/// linked. It therefore tests the flag with `jrcxz`, which neither reads nor
/// writes the arithmetic flags, borrowing rcx through the `ib_rcx` scratch slot.
/// That slot is otherwise owned only by the inline indirect-branch lookup routine,
/// which a linked terminator never runs, and a thread executes these strictly
/// sequentially, so the borrow cannot collide. The 32-bit load zero-extends the
/// `u32` flag into rcx, so `jrcxz` sees zero exactly when no signal is pending.
fn emit_exit_poll(out: &mut Vec<u8>) -> usize {
    let d_exit = offset_of!(ThreadState, exit_requested) as i32;
    let d_rcx = offset_of!(ThreadState, ib_rcx) as i32;
    // mov gs:[ib_rcx], rcx — stash guest rcx (no flags touched).
    gs_store(out, MODRM_RCX, d_rcx);
    // mov ecx, gs:[exit_requested] — rcx = flag, zero-extended (no flags touched).
    out.extend_from_slice(&[0x65, 0x8b, 0x0c, 0x25]);
    emit_u32(out, d_exit as u32);
    // jrcxz .cont — take the fast path if the flag is clear; preserves flags.
    out.push(0xe3);
    let jrcxz_at = out.len();
    out.push(0x00); // rel8, patched to .cont below
    // Flag set: restore guest rcx, then jump to the edge's cold-exit stub.
    gs_load(out, MODRM_RCX, d_rcx);
    out.push(0xe9);
    let stub_rel = take_rel32(out);
    // .cont: flag clear — restore guest rcx and continue on the fast path.
    let cont = out.len();
    let disp = cont as i64 - (jrcxz_at as i64 + 1);
    out[jrcxz_at] = i8::try_from(disp).expect("jrcxz poll displacement out of range") as u8;
    gs_load(out, MODRM_RCX, d_rcx);
    stub_rel
}

/// Build the raw machine code for a linkable terminator: a fast-path direct
/// branch followed by one cold exit stub per successor. Returns the encoded
/// bytes and, for each edge, the byte offset of its fast-path `rel32`
/// displacement paired with the successor's guest PC. The displacements are
/// initialized to target the stubs; the dispatcher rewrites them to the
/// successor blocks as those are translated.
///
/// A back-edge ([`is_back_edge`]) is preceded by a safepoint poll
/// ([`emit_exit_poll`]) that diverts to the edge's own cold-exit stub when an
/// asynchronous signal is pending, so a fully linked loop returns to the run loop
/// within one iteration. The stub already publishes the correct successor guest
/// PC, so delivery lands at a clean boundary.
///
/// `term_pc` is the host address at which these bytes will be emitted. It is
/// needed because every patchable `rel32` field is NOP-padded to a 4-byte
/// boundary, so the dispatcher can back-patch it with one aligned atomic store
/// (see [`super::cache::patch_site`]) — a sibling thread executing the branch
/// then reads the old or new target, never a torn displacement.
fn build_linked_terminator(
    link: &LinkTerm,
    exit_tramp: u64,
    block_start: u64,
    term_pc: usize,
) -> (Vec<u8>, Vec<(usize, u64)>) {
    let mut out = Vec::new();
    let mut edges = Vec::new();
    match *link {
        LinkTerm::Uncond { target } => {
            // An unconditional back-branch is the whole loop: poll before taking
            // it. The poll preserves the guest's flags and registers (see
            // emit_exit_poll), so a loop carrying flags across this edge is safe.
            let poll = is_back_edge(target, block_start).then(|| emit_exit_poll(&mut out));
            // jmp rel32 -> stub (later: -> target's host code)
            pad_rel32_alignment(&mut out, term_pc, 1);
            out.push(0xE9);
            let rel = take_rel32(&mut out);
            let stub = out.len();
            emit_stub(&mut out, target, exit_tramp);
            write_rel32(&mut out, rel, stub);
            if let Some(poll_rel) = poll {
                write_rel32(&mut out, poll_rel, stub);
            }
            edges.push((rel, target));
        }
        LinkTerm::Cond {
            opcode,
            taken,
            fallthrough,
        } => {
            // The fall-through is `next_ip`, always forward, so only the taken edge
            // can close a loop. When it does, route the taken path through a
            // flag-preserving poll once the guest's `jcc` has read the live flags;
            // the poll keeps them intact for the loop head on the fast path. Every
            // patchable `rel32` (the ones pushed to `edges`) is NOP-padded to a
            // 4-byte boundary so the dispatcher can back-patch it atomically.
            if is_back_edge(taken, block_start) {
                // jcc rel32 -> taken_check ; jmp rel32 -> fall stub.
                out.extend_from_slice(&[0x0F, opcode]);
                let jcc_rel = take_rel32(&mut out); // internal, not patched
                pad_rel32_alignment(&mut out, term_pc, 1);
                out.push(0xE9);
                let jmp_rel = take_rel32(&mut out);
                // taken_check: poll, then the linkable fast-path branch to taken.
                let taken_check = out.len();
                write_rel32(&mut out, jcc_rel, taken_check);
                let poll_rel = emit_exit_poll(&mut out);
                pad_rel32_alignment(&mut out, term_pc, 1);
                out.push(0xE9);
                let taken_jmp_rel = take_rel32(&mut out);
                let taken_stub = out.len();
                emit_stub(&mut out, taken, exit_tramp);
                let fall_stub = out.len();
                emit_stub(&mut out, fallthrough, exit_tramp);
                write_rel32(&mut out, jmp_rel, fall_stub);
                write_rel32(&mut out, taken_jmp_rel, taken_stub);
                write_rel32(&mut out, poll_rel, taken_stub);
                edges.push((taken_jmp_rel, taken));
                edges.push((jmp_rel, fallthrough));
            } else {
                // jcc rel32 -> taken stub ; jmp rel32 -> fall-through stub.
                // The native jcc reads the block's live guest flags directly. Each
                // branch is padded independently so both rel32 fields land aligned.
                pad_rel32_alignment(&mut out, term_pc, 2);
                out.extend_from_slice(&[0x0F, opcode]);
                let jcc_rel = take_rel32(&mut out);
                pad_rel32_alignment(&mut out, term_pc, 1);
                out.push(0xE9);
                let jmp_rel = take_rel32(&mut out);
                let taken_stub = out.len();
                emit_stub(&mut out, taken, exit_tramp);
                let fall_stub = out.len();
                emit_stub(&mut out, fallthrough, exit_tramp);
                write_rel32(&mut out, jcc_rel, taken_stub);
                write_rel32(&mut out, jmp_rel, fall_stub);
                edges.push((jcc_rel, taken));
                edges.push((jmp_rel, fallthrough));
            }
        }
        LinkTerm::DirectCall { target, ret } => {
            // Push the 64-bit return address with a single 8-byte store so the
            // callee's `ret` (an 8-byte `pop`) can store-to-load forward from
            // it. The obvious `push imm32` + `mov [rsp+4], imm32` split would
            // write the slot as two stores of different sizes; the pop then
            // straddles both and the load stalls (a store-to-load-forward
            // failure) on every call/ret pair. There is no `push imm64`, so
            // route the address through rax: borrow the rax slot, materialize
            // the full address, `push rax`, and restore rax — leaving every
            // guest register live for the linked successor. No flags are
            // touched (mov/movabs/push only).
            mov_gs_rax(&mut out, RAX_SLOT); // mov gs:[rax_slot], rax
            movabs_rax(&mut out, ret); // movabs rax, ret
            out.push(0x50); // push rax
            gs_load(&mut out, MODRM_RAX, RAX_SLOT); // mov rax, gs:[rax_slot]
            // A direct call whose target is at or before this block can close a
            // loop with no back-branch and no `ret`: direct or mutual recursion
            // through fixed call targets, which otherwise stays entirely inside
            // linked call edges until the stack overflows. Poll after the return
            // address is pushed (so the call's effect is complete) but before the
            // branch, so a pending signal is delivered at the callee entry. Every
            // call cycle contains an edge into its lowest-address member, so this
            // catches the cycle even when the individual calls run forward.
            let poll = is_back_edge(target, block_start).then(|| emit_exit_poll(&mut out));
            pad_rel32_alignment(&mut out, term_pc, 1);
            out.push(0xE9);
            let rel = take_rel32(&mut out);
            let stub = out.len();
            emit_stub(&mut out, target, exit_tramp);
            write_rel32(&mut out, rel, stub);
            if let Some(poll_rel) = poll {
                write_rel32(&mut out, poll_rel, stub);
            }
            edges.push((rel, target));
        }
    }
    (out, edges)
}

/// Emit a cold exit stub: stash the live rax, materialize the successor's
/// guest PC into the `rip` slot, and jump to the exit trampoline. Identical in
/// effect to the original `emit_terminator` exit tail, established as raw
/// bytes so the fast-path branch above can be patched independently.
fn emit_stub(out: &mut Vec<u8>, target_guest: u64, exit_tramp: u64) {
    mov_gs_rax(out, RAX_SLOT); // mov gs:[rax_slot], rax
    movabs_rax(out, target_guest); // movabs rax, target_guest
    mov_gs_rax(out, RIP_SLOT); // mov gs:[rip_slot], rax
    movabs_rax(out, exit_tramp); // movabs rax, exit_tramp
    out.extend_from_slice(&[0xFF, 0xE0]); // jmp rax
}

/// `mov gs:[disp32], rax` — `65 48 89 04 25 <disp32>`.
fn mov_gs_rax(out: &mut Vec<u8>, disp: i32) {
    out.extend_from_slice(&[0x65, 0x48, 0x89, 0x04, 0x25]);
    emit_u32(out, disp as u32);
}

/// `movabs rax, imm64` — `48 B8 <imm64>`.
fn movabs_rax(out: &mut Vec<u8>, imm: u64) {
    out.extend_from_slice(&[0x48, 0xB8]);
    out.extend_from_slice(&imm.to_le_bytes());
}

fn emit_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// ModRM bytes for `mov gs:[disp32], <reg>` / `mov <reg>, gs:[disp32]`: the
/// reg field selects the register and rm=100 pulls in a SIB byte (`0x25`) that
/// encodes a bare disp32. rax=`0x04`, rcx=`0x0c`, rdx=`0x14`.
const MODRM_RAX: u8 = 0x04;
const MODRM_RCX: u8 = 0x0c;
const MODRM_RDX: u8 = 0x14;

/// `mov gs:[disp32], <reg>` — `65 48 89 <modrm> 25 <disp32>`.
fn gs_store(out: &mut Vec<u8>, modrm: u8, disp: i32) {
    out.extend_from_slice(&[0x65, 0x48, 0x89, modrm, 0x25]);
    emit_u32(out, disp as u32);
}

/// `mov <reg>, gs:[disp32]` — `65 48 8b <modrm> 25 <disp32>`.
fn gs_load(out: &mut Vec<u8>, modrm: u8, disp: i32) {
    out.extend_from_slice(&[0x65, 0x48, 0x8b, modrm, 0x25]);
    emit_u32(out, disp as u32);
}

/// Restore the guest status flags previously saved by `lahf`/`seto` in the
/// `ib_flags` slot: reload them into rax, rebuild OF from AL with `add al,0x7f`,
/// then load SF/ZF/AF/PF/CF with `sahf`. Clobbers rax (callers reload guest rax
/// afterward), and is itself flag-setting only through the restore it performs.
fn emit_restore_flags(out: &mut Vec<u8>, d_flags: i32) {
    gs_load(out, MODRM_RAX, d_flags); // mov rax, gs:[flags]
    out.extend_from_slice(&[0x04, 0x7f]); // add al, 0x7f
    out.push(0x9e); // sahf
}

/// Reserve four bytes for a `rel32` displacement and return their offset.
fn take_rel32(out: &mut Vec<u8>) -> usize {
    let at = out.len();
    out.extend_from_slice(&[0; 4]);
    at
}

/// Pad `out` with single-byte NOPs until the `rel32` field that will follow
/// `opcode_len` opcode bytes lands on a 4-byte boundary in the cache, given the
/// terminator will be emitted at `term_pc`. A naturally aligned `rel32` can be
/// back-patched with a single atomic 32-bit store, so a sibling thread executing
/// the branch sees the old displacement or the new one but never a torn mix.
fn pad_rel32_alignment(out: &mut Vec<u8>, term_pc: usize, opcode_len: usize) {
    while !(term_pc + out.len() + opcode_len).is_multiple_of(4) {
        out.push(0x90); // nop
    }
}

/// Write the `rel32` at `rel` so the branch reaches `target` (both byte
/// offsets within `out`). A `rel32` is measured from the end of its 4 bytes.
fn write_rel32(out: &mut [u8], rel: usize, target: usize) {
    let disp = target as i64 - (rel as i64 + 4);
    out[rel..rel + 4].copy_from_slice(&(disp as i32).to_le_bytes());
}

/// The second byte of the `0F 8x` near-conditional-branch encoding for a
/// `Jcc` instruction, or `None` for conditional forms the linker does not
/// lower (`loop`, `loopcc`, `jrcxz`/`jecxz`).
fn jcc_opcode(code: Code) -> Option<u8> {
    Some(match code {
        Code::Jo_rel8_64 | Code::Jo_rel32_64 => 0x80,
        Code::Jno_rel8_64 | Code::Jno_rel32_64 => 0x81,
        Code::Jb_rel8_64 | Code::Jb_rel32_64 => 0x82,
        Code::Jae_rel8_64 | Code::Jae_rel32_64 => 0x83,
        Code::Je_rel8_64 | Code::Je_rel32_64 => 0x84,
        Code::Jne_rel8_64 | Code::Jne_rel32_64 => 0x85,
        Code::Jbe_rel8_64 | Code::Jbe_rel32_64 => 0x86,
        Code::Ja_rel8_64 | Code::Ja_rel32_64 => 0x87,
        Code::Js_rel8_64 | Code::Js_rel32_64 => 0x88,
        Code::Jns_rel8_64 | Code::Jns_rel32_64 => 0x89,
        Code::Jp_rel8_64 | Code::Jp_rel32_64 => 0x8A,
        Code::Jnp_rel8_64 | Code::Jnp_rel32_64 => 0x8B,
        Code::Jl_rel8_64 | Code::Jl_rel32_64 => 0x8C,
        Code::Jge_rel8_64 | Code::Jge_rel32_64 => 0x8D,
        Code::Jle_rel8_64 | Code::Jle_rel32_64 => 0x8E,
        Code::Jg_rel8_64 | Code::Jg_rel32_64 => 0x8F,
        _ => return None,
    })
}

/// Emit the exit sequence for a terminator the linker does not lower: a
/// syscall, an indirect branch/call, or a return.
///
/// In every case the guest's original `rax` is first saved at `gs:[0]` and the
/// next guest PC is computed into `rax`. A syscall then takes the common exit
/// tail (`mov gs:[128], rax; movabs rax, syscall_tramp; jmp rax`); the indirect
/// branch, indirect call, and return hand off to the shared inline lookup
/// routine via [`emit_jmp_ib_lookup`], which resolves the target in `rax`.
fn emit_terminator(
    instrs: &mut Vec<Instruction>,
    t: &Instruction,
    syscall_tramp: u64,
) -> Result<(), Error> {
    let next_ip = t.next_ip();
    // `syscall` is special: the instruction itself does *not* run. We save
    // the guest state (rax holds the syscall number, the args are in their
    // usual registers — exit_trampoline captures them all) and exit through
    // `syscall_exit_trampoline`, which signals the dispatcher to invoke the
    // embedder's `SystemCalls` handler before resuming. Checked by opcode
    // because iced reports `syscall`'s `flow_control()` as `Call`.
    //
    // x86-64 `syscall` architecturally clobbers `rcx` (with the address of
    // the instruction following `syscall`) and `r11` (with the caller's
    // `rflags`). Since the instruction never actually runs in the cache,
    // the translator has to synthesize these side effects into the guest
    // register slots so the guest sees them on resume. The companion
    // `syscall_exit_trampoline` deliberately leaves `gs:[16]`, `gs:[88]`,
    // and `gs:[136]` untouched, trusting these writes.
    //
    // Real `syscall` does not touch user memory; neither does this
    // emulation. To capture rflags we briefly switch `rsp` to Chimera's
    // own stack (`gs:[144]`), `pushfq`/`pop` there, and switch back to
    // the guest's rsp. The guest never sees a write below its own stack
    // pointer, so a guest with `rsp` parked on a guard page does not
    // spuriously fault on a translated syscall.
    if t.code() == Code::Syscall {
        emit_save_rax(instrs)?;

        // rcx <- next_ip
        emit_load_rax_imm(instrs, next_ip)?;
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_rm64_r64,
            gs_qword(16),
            Register::RAX,
        ))?);

        // Stash guest rsp (the trampoline later overwrites this slot with
        // the same value), switch to Chimera's stack, capture rflags into
        // rax via pushfq/pop, then restore guest rsp.
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_rm64_r64,
            gs_qword(56),
            Register::RSP,
        ))?);
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_r64_rm64,
            Register::RSP,
            gs_qword(144),
        ))?);
        instrs.push(Instruction::with(Code::Pushfq));
        emit_pop_rax(instrs)?;
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_r64_rm64,
            Register::RSP,
            gs_qword(56),
        ))?);

        // r11 <- rflags
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_rm64_r64,
            gs_qword(88),
            Register::RAX,
        ))?);
        // rflags slot <- rflags too, so the trampoline does not need to
        // pushfq on the guest stack to populate it.
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_rm64_r64,
            gs_qword(136),
            Register::RAX,
        ))?);

        // Reload next_ip for the exit tail, which stores it as the resumed
        // guest rip in `gs:[128]`.
        emit_load_rax_imm(instrs, next_ip)?;
        return emit_exit_tail(instrs, syscall_tramp);
    }
    match t.flow_control() {
        FlowControl::UnconditionalBranch => {
            let target = t.near_branch_target();
            emit_save_rax(instrs)?;
            emit_load_rax_imm(instrs, target)?;
        }
        FlowControl::ConditionalBranch => {
            let taken = t.near_branch_target();
            emit_cond_select(instrs, t.code(), taken, next_ip)?;
        }
        FlowControl::Call => {
            let target = t.near_branch_target();
            emit_save_rax(instrs)?;
            emit_load_rax_imm(instrs, next_ip)?;
            emit_push_rax(instrs)?;
            emit_load_rax_imm(instrs, target)?;
        }
        FlowControl::IndirectBranch => {
            emit_save_rax(instrs)?;
            emit_load_rax_from_op0(instrs, t)?;
        }
        FlowControl::IndirectCall => {
            // Read the target with the *original* rsp, before pushing the
            // return address: hardware computes `CALL m`'s target, then
            // pushes. Pushing first would evaluate an rsp-relative operand
            // (e.g. `call [rsp+0x58]`) against the post-push rsp, reading 8
            // bytes off. rax still holds the guest's rax at this point, so an
            // operand that uses rax (e.g. `call [rax+8]`) also reads correctly.
            emit_save_rax(instrs)?;
            emit_load_rax_from_op0(instrs, t)?;
            // Stash the target in the rbx slot, push the return address, then
            // reload the target. The rbx register is untouched, so the guest's
            // rbx survives via exit_block's save.
            instrs.push(mkinstr(Instruction::with2(
                Code::Mov_rm64_r64,
                gs_qword(RBX_SLOT),
                Register::RAX,
            ))?);
            emit_load_rax_imm(instrs, next_ip)?;
            emit_push_rax(instrs)?;
            emit_load_rax_from_gs(instrs, RBX_SLOT)?;
        }
        FlowControl::Return => {
            emit_save_rax(instrs)?;
            emit_pop_rax(instrs)?;
        }
        other => {
            return Err(Error::Translate(format!(
                "unhandled terminator at {:#x}: flow_control={:?} code={:?}",
                t.ip(),
                other,
                t.code(),
            )));
        }
    }
    // The reachable arms here — indirect branch, indirect call, return — have
    // left the runtime-computed guest target in rax. Hand off to the shared
    // inline lookup routine, which jumps straight to the cached translation on
    // a hit and otherwise falls back to the dispatcher.
    emit_jmp_ib_lookup(instrs)
}

/// Emit `jmp gs:[ib_lookup]`, transferring to the shared inline
/// indirect-branch lookup routine with the resolved guest target in rax and
/// the guest's rax already saved at `gs:[0]`.
fn emit_jmp_ib_lookup(instrs: &mut Vec<Instruction>) -> Result<(), Error> {
    let disp = offset_of!(ThreadState, ib_lookup) as i64;
    instrs.push(mkinstr(Instruction::with1(Code::Jmp_rm64, gs_qword(disp)))?);
    Ok(())
}

fn emit_save_rax(instrs: &mut Vec<Instruction>) -> Result<(), Error> {
    instrs.push(mkinstr(Instruction::with2(
        Code::Mov_rm64_r64,
        gs_qword(0),
        Register::RAX,
    ))?);
    Ok(())
}

fn emit_load_rax_imm(instrs: &mut Vec<Instruction>, imm: u64) -> Result<(), Error> {
    instrs.push(mkinstr(Instruction::with2(
        Code::Mov_r64_imm64,
        Register::RAX,
        imm,
    ))?);
    Ok(())
}

fn emit_load_rax_from_gs(instrs: &mut Vec<Instruction>, disp: i64) -> Result<(), Error> {
    instrs.push(mkinstr(Instruction::with2(
        Code::Mov_r64_rm64,
        Register::RAX,
        gs_qword(disp),
    ))?);
    Ok(())
}

fn emit_push_rax(instrs: &mut Vec<Instruction>) -> Result<(), Error> {
    instrs.push(mkinstr(Instruction::with1(Code::Push_r64, Register::RAX))?);
    Ok(())
}

fn emit_pop_rax(instrs: &mut Vec<Instruction>) -> Result<(), Error> {
    instrs.push(mkinstr(Instruction::with1(Code::Pop_r64, Register::RAX))?);
    Ok(())
}

fn emit_load_rax_from_op0(instrs: &mut Vec<Instruction>, t: &Instruction) -> Result<(), Error> {
    match t.op0_kind() {
        OpKind::Register => {
            let src = t.op0_register();
            if src != Register::RAX {
                instrs.push(mkinstr(Instruction::with2(
                    Code::Mov_r64_rm64,
                    Register::RAX,
                    src,
                ))?);
            }
            // If src is rax, rax already holds the value (mov gs:[0], rax
            // didn't modify rax).
        }
        OpKind::Memory => {
            let memop = extract_memory_operand(t);
            instrs.push(mkinstr(Instruction::with2(
                Code::Mov_r64_rm64,
                Register::RAX,
                memop,
            ))?);
        }
        other => {
            return Err(Error::Translate(format!(
                "unexpected op0 kind in indirect branch: {:?}",
                other
            )));
        }
    }
    Ok(())
}

fn extract_memory_operand(t: &Instruction) -> MemoryOperand {
    MemoryOperand::new(
        t.memory_base(),
        t.memory_index(),
        t.memory_index_scale(),
        t.memory_displacement64() as i64,
        if t.memory_displ_size() == 0 { 0 } else { 8 },
        t.is_broadcast(),
        t.segment_prefix(),
    )
}

/// For a conditional branch, select between `taken` and `fallthrough` with a
/// `cmov`, without ever touching the guest stack. A `push`/`pop` here would
/// write `[rsp-8]` and clobber the guest's red zone — the 128 bytes below
/// `rsp` that the System V ABI reserves for a leaf function's own use.
///
/// Instead, `taken` is stashed in the context's rbx slot (`gs:[8]`) and the
/// `cmov` reads it straight from memory. The guest's rbx *register* is never
/// touched, so it stays live; `exit_block` then saves that live rbx over the
/// slot on the way out, overwriting the scratch value. The flags set by the
/// block's compare survive up to the `cmov` because every instruction emitted
/// here (`mov` to memory, `movabs reg, imm64`) leaves flags untouched.
fn emit_cond_select(
    instrs: &mut Vec<Instruction>,
    jcc_code: Code,
    taken: u64,
    fallthrough: u64,
) -> Result<(), Error> {
    let cmov = jcc_to_cmov(jcc_code)?;
    emit_save_rax(instrs)?;
    // gs:[rbx] <- taken, staged through rax (there is no `mov m64, imm64`).
    emit_load_rax_imm(instrs, taken)?;
    instrs.push(mkinstr(Instruction::with2(
        Code::Mov_rm64_r64,
        gs_qword(RBX_SLOT),
        Register::RAX,
    ))?);
    // rax <- fallthrough, then pull in `taken` from gs:[rbx] if the condition holds.
    emit_load_rax_imm(instrs, fallthrough)?;
    instrs.push(mkinstr(Instruction::with2(
        cmov,
        Register::RAX,
        gs_qword(RBX_SLOT),
    ))?);
    Ok(())
}

fn jcc_to_cmov(jcc: Code) -> Result<Code, Error> {
    Ok(match jcc {
        Code::Je_rel8_64 | Code::Je_rel32_64 => Code::Cmove_r64_rm64,
        Code::Jne_rel8_64 | Code::Jne_rel32_64 => Code::Cmovne_r64_rm64,
        Code::Ja_rel8_64 | Code::Ja_rel32_64 => Code::Cmova_r64_rm64,
        Code::Jae_rel8_64 | Code::Jae_rel32_64 => Code::Cmovae_r64_rm64,
        Code::Jb_rel8_64 | Code::Jb_rel32_64 => Code::Cmovb_r64_rm64,
        Code::Jbe_rel8_64 | Code::Jbe_rel32_64 => Code::Cmovbe_r64_rm64,
        Code::Jg_rel8_64 | Code::Jg_rel32_64 => Code::Cmovg_r64_rm64,
        Code::Jge_rel8_64 | Code::Jge_rel32_64 => Code::Cmovge_r64_rm64,
        Code::Jl_rel8_64 | Code::Jl_rel32_64 => Code::Cmovl_r64_rm64,
        Code::Jle_rel8_64 | Code::Jle_rel32_64 => Code::Cmovle_r64_rm64,
        Code::Jo_rel8_64 | Code::Jo_rel32_64 => Code::Cmovo_r64_rm64,
        Code::Jno_rel8_64 | Code::Jno_rel32_64 => Code::Cmovno_r64_rm64,
        Code::Js_rel8_64 | Code::Js_rel32_64 => Code::Cmovs_r64_rm64,
        Code::Jns_rel8_64 | Code::Jns_rel32_64 => Code::Cmovns_r64_rm64,
        Code::Jp_rel8_64 | Code::Jp_rel32_64 => Code::Cmovp_r64_rm64,
        Code::Jnp_rel8_64 | Code::Jnp_rel32_64 => Code::Cmovnp_r64_rm64,
        other => {
            return Err(Error::Translate(format!(
                "unsupported conditional branch: {:?}",
                other
            )));
        }
    })
}

fn emit_exit_tail(instrs: &mut Vec<Instruction>, exit_tramp: u64) -> Result<(), Error> {
    // mov gs:[128], rax
    instrs.push(mkinstr(Instruction::with2(
        Code::Mov_rm64_r64,
        gs_qword(128),
        Register::RAX,
    ))?);
    // movabs rax, exit_tramp
    emit_load_rax_imm(instrs, exit_tramp)?;
    // jmp rax
    instrs.push(mkinstr(Instruction::with1(Code::Jmp_rm64, Register::RAX))?);
    Ok(())
}

fn mkinstr(r: Result<Instruction, iced_x86::IcedError>) -> Result<Instruction, Error> {
    r.map_err(|e| Error::Translate(format!("build instruction: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every patchable `rel32` field of a linked terminator must land on a
    /// 4-byte boundary, whatever address the terminator is emitted at, so the
    /// dispatcher can back-patch it with one aligned atomic store. Check all
    /// eight residues of `term_pc` mod 4 so the padding is exercised for every
    /// starting alignment.
    fn assert_edges_aligned(link: &LinkTerm) {
        for term_pc in 0..8usize {
            let (bytes, edges) = build_linked_terminator(link, 0xdead_beef, 0, term_pc);
            assert!(!edges.is_empty(), "a linked terminator must expose an edge");
            for (off, _target) in edges {
                assert!(
                    off + 4 <= bytes.len(),
                    "rel32 field at off={off} runs past the {} terminator bytes",
                    bytes.len(),
                );
                assert_eq!(
                    (term_pc + off) % 4,
                    0,
                    "rel32 field at term_pc={term_pc}, off={off} is not 4-byte aligned",
                );
            }
        }
    }

    #[test]
    fn uncond_edge_is_aligned() {
        assert_edges_aligned(&LinkTerm::Uncond { target: 0x1000 });
    }

    #[test]
    fn cond_edges_are_aligned() {
        // 0x84 is `jz`; both the taken (jcc) and fall-through (jmp) rel32 fields
        // must be aligned even though they sit five bytes apart.
        assert_edges_aligned(&LinkTerm::Cond {
            opcode: 0x84,
            taken: 0x1000,
            fallthrough: 0x2000,
        });
    }

    #[test]
    fn direct_call_edge_is_aligned() {
        assert_edges_aligned(&LinkTerm::DirectCall {
            target: 0x1000,
            ret: 0x1005,
        });
    }

    /// The transient key `ib_insert` publishes mid-update must hash to a
    /// different slot than the entry it invalidates, for every possible key —
    /// otherwise a concurrent lookup whose target equals the marker could
    /// match the in-progress slot and misdispatch. Sweep structured values
    /// (canonical and non-canonical PCs, bit patterns near the extremes) and a
    /// deterministic pseudo-random set.
    #[test]
    fn transient_key_never_hashes_to_its_own_slot() {
        let check = |x: u64| {
            assert_ne!(
                ib_slot(ib_unmatchable(x)),
                ib_slot(x),
                "marker for {x:#x} hashes to its own slot"
            );
        };
        for i in 0..64 {
            check(1u64 << i);
            check((1u64 << i) - 1);
            check(!(1u64 << i));
        }
        check(0);
        check(u64::MAX);
        let mut x = 0x1234_5678_9abc_def0u64;
        for _ in 0..10_000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            check(x);
        }
    }

    /// A cleared table must contain no matchable slot: for every slot that any
    /// target could probe, the key must not equal a target hashing there. The
    /// only candidate is the slot `u64::MAX` hashes to (the fill pattern),
    /// which `clear_ib_table` repairs.
    #[test]
    fn empty_table_never_matches() {
        let cache = CodeCache::new(4096).unwrap();
        let slot = ib_slot(u64::MAX) * IB_SLOT_BYTES;
        let key = unsafe { (cache.ib_table.add(slot) as *const u64).read() };
        assert_ne!(key, u64::MAX);
        assert_ne!(ib_slot(key), ib_slot(u64::MAX));
    }
}
