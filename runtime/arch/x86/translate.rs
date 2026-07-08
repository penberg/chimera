//! Basic-block translator. Decodes a single guest basic block, copies the
//! straight-line prefix into the code cache (with RIP-relative operands
//! fixed up by `BlockEncoder`), and rewrites the terminator into a
//! "compute next guest PC, then exit to the dispatcher" sequence.

use std::{
    mem::offset_of,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

use iced_x86::{
    BlockEncoder, BlockEncoderOptions, Code, CpuidFeature, Decoder, DecoderError, DecoderOptions,
    FlowControl, Instruction, InstructionBlock, InstructionInfoFactory, MemoryOperand, OpKind,
    Register,
};

use crate::Error;

use super::{dispatch::ThreadState, trampoline::fetch_copy};

const MAX_BLOCK_GUEST_BYTES: usize = 4096;
/// Longest possible x86-64 instruction encoding. An instruction whose start
/// lies within this many bytes of the decode window's end could be truncated by
/// it, so the block is split there rather than risk decoding a partial encoding.
const MAX_INSTR_LEN: u64 = 15;

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

/// Host bounds `[lo, hi)` of the one translated-code buffer, published when it is
/// mapped and read by the synchronous fault handler to classify a fault.
static CODE_CACHE_LO: AtomicUsize = AtomicUsize::new(0);
static CODE_CACHE_HI: AtomicUsize = AtomicUsize::new(0);

/// Whether `addr` is a host PC inside the translated-code buffer. The fault
/// handler uses this to tell a self-modifying-code write — a guest store that
/// faulted while executing translated code — from a genuine fault taken in
/// Chimera's own Rust; it also means the faulting thread is not holding the
/// address-space lock (only ever held off the code cache), so the handler can
/// take it without self-deadlock.
pub fn code_cache_contains(addr: usize) -> bool {
    let lo = CODE_CACHE_LO.load(Ordering::Relaxed);
    let hi = CODE_CACHE_HI.load(Ordering::Relaxed);
    lo != 0 && addr >= lo && addr < hi
}

/// `PROT_NONE` guard reserved on each side of the RWX code buffer. The kernel
/// places guest worker-thread stacks in the same high mmap area as the code
/// cache, sometimes immediately adjacent to it; JavaScriptCore's stack scrubber
/// (a descending `mov [rdx], 0` loop) can run off the end of such a stack and,
/// without a guard, would silently zero translated code in the abutting cache —
/// surfacing later as a wild jump into a zeroed hole. The guard turns that
/// overrun into an immediate fault at the scrubbing store instead, so the fault
/// handler sees it at its source. It is virtual-address-space only (`PROT_NONE`,
/// never faulted in), so it costs no resident memory.
const CACHE_GUARD: usize = 64 * 1024 * 1024;

/// A bump-allocated RWX region into which `translate()` emits blocks, paired
/// with the inline indirect-branch lookup table and the shared lookup routine.
pub struct CodeCache {
    base: *mut u8,
    size: usize,
    /// Base and length of the whole reservation (`CACHE_GUARD` + `size` +
    /// `CACHE_GUARD`), unmapped on drop; `base` points `CACHE_GUARD` into it.
    map_base: *mut u8,
    map_size: usize,
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
        // Reserve guard + buffer + guard as one `PROT_NONE` mapping, then open
        // only the middle to RWX. The guards (see [`CACHE_GUARD`]) catch an
        // overrun from an adjacent guest mapping before it reaches the buffer.
        let map_size = CACHE_GUARD + size + CACHE_GUARD;
        let region = unsafe {
            libc::mmap(
                ptr::null_mut(),
                map_size,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if region == libc::MAP_FAILED {
            return Err(Error::last_os_error("code cache reservation"));
        }
        let p = unsafe { (region as *mut u8).add(CACHE_GUARD) as *mut libc::c_void };
        if unsafe {
            libc::mprotect(
                p,
                size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            )
        } != 0
        {
            let err = Error::last_os_error("code cache mprotect");
            unsafe { libc::munmap(region, map_size) };
            return Err(err);
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
            unsafe { libc::munmap(region, map_size) };
            return Err(err);
        }
        let cache = Self {
            base: p as *mut u8,
            size,
            map_base: region as *mut u8,
            map_size,
            used: 0,
            ib_table: t as *mut u8,
            ib_lookup: None,
        };
        cache.clear_ib_table();
        // Publish the buffer bounds for the fault handler's in-cache check. One
        // CodeCache backs the process (reset rewinds it rather than remapping),
        // so this is set once.
        CODE_CACHE_LO.store(p as usize, Ordering::Relaxed);
        CODE_CACHE_HI.store(p as usize + size, Ordering::Relaxed);
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

    /// Redirect a block's host entry to its deopt stub by overwriting the first
    /// 5 bytes at `host_pc` with `jmp rel32 -> deopt_pc`. Called when the guest
    /// modifies the page the block was translated from: the block is dropped
    /// from the map, and any direct branch still linked to it now lands on the
    /// stub, which exits to the dispatcher and re-translates from current guest
    /// memory. Relies on JIT discipline — the guest stops every thread before
    /// rewriting code it might be running — so no sibling executes this block
    /// while it is patched. The `jmp` opcode is published last, so a reader that
    /// somehow raced would see the original instruction until the redirect is
    /// fully formed rather than a spliced one.
    pub fn neutralize(&self, host_pc: u64, deopt_pc: u64) {
        let rel = deopt_pc as i64 - (host_pc as i64 + 5);
        debug_assert!(
            i32::try_from(rel).is_ok(),
            "deopt displacement {rel} out of rel32 range"
        );
        let rel = rel as i32 as u32;
        unsafe {
            let p = host_pc as *mut u8;
            ptr::write_volatile(p.add(1), rel as u8);
            ptr::write_volatile(p.add(2), (rel >> 8) as u8);
            ptr::write_volatile(p.add(3), (rel >> 16) as u8);
            ptr::write_volatile(p.add(4), (rel >> 24) as u8);
            ptr::write_volatile(p, 0xE9);
        }
    }

    /// Drop `guest_pc` from the inline indirect-branch table if it currently
    /// holds the slot, so a later indirect branch to it misses and re-resolves
    /// through the dispatcher (where the block has been dropped and re-translates
    /// from current guest memory).
    pub fn ib_remove(&self, guest_pc: u64) {
        let slot = ib_slot(guest_pc) * IB_SLOT_BYTES;
        unsafe {
            let key = self.ib_table.add(slot) as *mut u64;
            if key.read_volatile() == guest_pc {
                key.write_volatile(u64::MAX);
            }
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
        // Unmap the whole reservation (both guards plus the buffer).
        let ret = unsafe { libc::munmap(self.map_base.cast(), self.map_size) };
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

/// The result of translating one basic block: where its host code begins, the
/// guest PC one past its last decoded byte (so the cache knows which guest pages
/// the block covers, for self-modifying-code invalidation), the address of its
/// deopt stub (see [`CodeCache::neutralize`]), and its statically known outgoing
/// edges for the dispatcher to link.
pub struct Translation {
    pub host_pc: u64,
    pub guest_end: u64,
    pub deopt_pc: u64,
    pub edges: Vec<OutEdge>,
}

/// Copy up to one decode window of guest bytes at `guest_pc` into `buf`,
/// without trusting the address ([`fetch_copy`]). Returns how many bytes are
/// readable: the whole window, its prefix up to where the window runs into
/// unreadable memory, or 0 when `guest_pc` itself is unreadable. This runs
/// once per translated block — over a million times while a large program
/// starts — so it is a direct fault-guarded copy, not a `process_vm_readv`
/// probe: W^X guarantees every page the guest can execute is host-readable,
/// so the guarded fault path never runs except on a wild jump.
fn read_guest_window(guest_pc: u64, buf: &mut [u8]) -> usize {
    fetch_copy(guest_pc, buf)
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
    trap_tramp: u64,
) -> Result<Translation, Error> {
    let host_pc = cache.next_pc();
    // Read the decode window through a guarded copy rather than dereferencing
    // the guest PC: a wild guest jump (through a corrupted function pointer,
    // say) lands here with an arbitrary address, and a raw read would take the
    // fault in Chimera. An unreadable PC is the guest's fault to receive — the
    // run loop raises SIGSEGV at it — and a window cut short by the end of a
    // mapping decodes only its readable prefix.
    let mut window = [0u8; MAX_BLOCK_GUEST_BYTES];
    let avail = read_guest_window(guest_pc, &mut window);
    if avail == 0 {
        return Err(Error::BadAccess(guest_pc));
    }
    let guest_bytes = &window[..avail];
    let mut decoder = Decoder::with_ip(64, guest_bytes, guest_pc, DecoderOptions::NONE);
    let mut instrs = Vec::new();
    let mut instr = Instruction::default();
    let window_end = guest_pc.wrapping_add(avail as u64);

    // Guest PC one past the block's last decoded byte: the terminator's end, or
    // the split point when an over-long block is cut short. Used to record which
    // guest pages the block covers for SMC invalidation. Every break assigns it.
    let guest_end;
    let term = loop {
        // A basic block can be longer than the fixed decode window — most often
        // a long straight-line run such as a jump-table initializer. Decoding an
        // instruction that straddles the window's end would feed iced a
        // truncated encoding, which it reports as an invalid (Exception)
        // instruction. Split the block at this instruction boundary instead:
        // synthesize an unconditional jump to the next guest PC, which the
        // dispatcher translates (and links) as its own block, resuming decode
        // from a fresh window. Forward progress is guaranteed because the first
        // instruction always starts `MAX_INSTR_LEN` short of the window end.
        let next_pc = decoder.ip();
        // The `next_pc != guest_pc` guard keeps a window shorter than one
        // instruction (its readable prefix ends within `MAX_INSTR_LEN` of the
        // block start) from splitting into a jump-to-self: the first
        // instruction always decodes from whatever bytes are readable instead.
        if next_pc != guest_pc && next_pc.wrapping_add(MAX_INSTR_LEN) > window_end {
            guest_end = next_pc;
            break mkinstr(Instruction::with_branch(Code::Jmp_rel32_64, next_pc))?;
        }
        if !decoder.can_decode() {
            return Err(Error::Translate(format!(
                "decoder ran out of bytes at {:#x}",
                guest_pc
            )));
        }
        decoder.decode_out(&mut instr);
        // An instruction cut off by the end of the readable window: its
        // remaining bytes are unmapped, so executing it faults there. Only the
        // block's first instruction can get here — later ones split above.
        if instr.is_invalid() && decoder.last_error() == DecoderError::NoMoreBytes {
            return Err(Error::BadAccess(window_end));
        }
        if matches!(instr.flow_control(), FlowControl::Next) {
            instrs.push(instr);
            continue;
        }
        guest_end = instr.next_ip();
        break instr;
    };

    rewrite_rip_relative_leas(&mut instrs)?;

    // A 66 operand-size prefix on a near `ret` truncates the popped rip to 16
    // bits on Intel hardware, but iced decodes the prefixed form to the same
    // `Retnq`/`Retnq_imm16` code as the plain one, so the return lowering in
    // `emit_terminator` cannot see it. Catch it here, where the raw bytes are
    // still at hand: everything before the `C3`/`C2 imm16` opcode tail is a
    // prefix, and no legitimate 64-bit code emits a 16-bit return — reject it
    // rather than silently lower it as a 64-bit return.
    if matches!(term.code(), Code::Retnq | Code::Retnq_imm16) {
        let off = (term.ip() - guest_pc) as usize;
        let tail = if term.code() == Code::Retnq_imm16 {
            3
        } else {
            1
        };
        if guest_bytes[off..off + term.len() - tail].contains(&0x66) {
            return Err(Error::Translate(format!(
                "unhandled 16-bit near return at {:#x}",
                term.ip(),
            )));
        }
    }

    // A block that touches FP/SIMD state or reads guest TLS (`fs:`) opens with
    // a lazy-install prologue: `dispatch` installs neither the guest's FP state
    // nor its FS base on entry, so the first block of a residency that needs
    // either installs it (and sets the matching flag) before its body runs.
    // Blocks that need neither emit no prologue and `host_pc` is their first
    // body instruction; the prologue, when present, sits at `host_pc` so links
    // and the indirect-branch table reach it (and re-run its checks) like any
    // other entry.
    let needs = BlockNeeds {
        fp: block_uses_fp(&instrs, &term),
        fs: block_uses_fs(&instrs, &term),
    };
    if needs.fp || needs.fs {
        let prologue = build_prologue(needs, block_flags_live_in(&instrs, &term));
        cache.emit(&prologue)?;
    }
    let body_pc = cache.next_pc();

    // A terminator with statically known target(s) — a direct jmp, a direct
    // call, or a supported conditional branch — gets the linkable layout: the
    // straight-line body, then a fast-path direct branch (initially aimed at a
    // cold exit stub) that the dispatcher later back-patches to the successor.
    // Everything else (indirect branches/calls, returns, syscalls, and the few
    // unsupported conditional forms) keeps the original "compute next guest PC,
    // exit to dispatcher" terminator and contributes no links.
    let edges = if let Some(link) = classify_terminator(&term) {
        emit_body(cache, &instrs, body_pc, guest_pc)?;
        let term_pc = cache.next_pc() as usize;
        let (bytes, rel_edges) = build_linked_terminator(&link, exit_tramp, guest_pc, term_pc);
        cache.emit(&bytes)?;
        rel_edges
            .into_iter()
            .map(|(off, target_guest)| OutEdge {
                target_guest,
                site: term_pc + off,
            })
            .collect()
    } else {
        emit_terminator(&mut instrs, &term, syscall_tramp, trap_tramp)?;
        emit_body(cache, &instrs, body_pc, guest_pc)?;
        Vec::new()
    };

    // A per-block deopt stub: it saves rax, publishes this block's own guest PC
    // into the rip slot, and exits to the dispatcher. SMC invalidation overwrites
    // the block's host entry with a 5-byte jump to this stub, so any surviving
    // direct link into the dropped block falls back to the dispatcher and
    // re-translates from current guest memory (see [`CodeCache::neutralize`]).
    let deopt_pc = cache.next_pc();
    let mut stub = Vec::new();
    emit_stub(&mut stub, guest_pc, exit_tramp);
    cache.emit(&stub)?;

    Ok(Translation {
        host_pc,
        guest_end,
        deopt_pc,
        edges,
    })
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

/// What lazily-installed guest state a block needs in place before its body
/// runs: the FP/SIMD register file, the guest FS base, or neither. Drives
/// whether — and what — the block's prologue emits.
#[derive(Clone, Copy)]
struct BlockNeeds {
    fp: bool,
    fs: bool,
}

/// Whether any instruction in the block (body or terminator) reads or writes
/// FP/SIMD state — x87, MMX, SSE/AVX/AVX-512 vector registers, opmask (`k`) or
/// tile registers — and therefore needs the guest's `fpstate` live in the
/// physical registers. Register-naming instructions are caught by their used
/// registers; the few that touch state without naming a vector register
/// (`ldmxcsr`/`stmxcsr`, `vzeroupper`/`vzeroall`, the x87 control-word ops) are
/// caught by their CPUID feature. Over-approximating only costs an unnecessary
/// save/restore, never correctness; under-approximating would drop guest state,
/// so the two signals are combined.
fn block_uses_fp(body: &[Instruction], term: &Instruction) -> bool {
    let mut info = InstructionInfoFactory::new();
    body.iter()
        .chain(std::iter::once(term))
        .any(|i| instr_uses_fp(&mut info, i))
}

/// Whether any instruction in the block reads or writes through the FS segment
/// — a `fs:`-prefixed memory access (guest TLS) or an `rd/wrfsbase` — and so
/// needs the guest's FS base installed rather than Chimera's. A block that
/// never touches FS runs correctly with Chimera's base still in FS, sparing the
/// `wrfsbase` pair around the residency.
fn block_uses_fs(body: &[Instruction], term: &Instruction) -> bool {
    body.iter().chain(std::iter::once(term)).any(instr_uses_fs)
}

fn instr_uses_fs(instr: &Instruction) -> bool {
    if instr.segment_prefix() == Register::FS {
        return true;
    }
    matches!(
        instr.code(),
        Code::Rdfsbase_r32 | Code::Rdfsbase_r64 | Code::Wrfsbase_r32 | Code::Wrfsbase_r64
    )
}

fn instr_uses_fp(info: &mut InstructionInfoFactory, instr: &Instruction) -> bool {
    let touches_vector_reg = info.info(instr).used_registers().iter().any(|u| {
        let r = u.register();
        r.is_xmm() || r.is_ymm() || r.is_zmm() || r.is_mm() || r.is_st() || r.is_k() || r.is_tmm()
    });
    touches_vector_reg || instr.cpuid_features().iter().any(|&f| is_fp_feature(f))
}

/// Whether a CPUID feature implies the instruction touches FP/SIMD/x87 state.
/// Covers the operations `block_uses_fp`'s register scan misses because they
/// name no vector register: the control words (`ldmxcsr` is SSE, `vzeroupper`
/// is AVX, `fldcw`/`fnstcw` are x87/`FPU`) and, critically, the whole-state
/// save/restore family (`fxsave`, `xsave`/`xsavec`/`xsaves`/`xsaveopt`,
/// `fxrstor`/`xrstor`/`xrstors`) that glibc's lazy-PLT resolver
/// (`_dl_runtime_resolve_xsavec`) spills the caller's vector registers through.
/// Such a block needs the guest's `fpstate` live in the physical registers
/// before it runs, exactly like an arithmetic SIMD block. The arithmetic forms
/// of every vector extension already name a register and are caught there.
fn is_fp_feature(f: CpuidFeature) -> bool {
    use CpuidFeature::*;
    matches!(
        f,
        FPU | FPU287
            | FPU387
            | MMX
            | SSE
            | SSE2
            | SSE3
            | SSSE3
            | SSE4_1
            | SSE4_2
            | SSE4A
            | AVX
            | AVX2
            | FMA
            | F16C
            | XOP
            | FMA4
            | FXSR
            | XSAVE
            | XSAVEC
            | XSAVES
            | XSAVEOPT
    )
}

/// Whether the block reads any status flag a predecessor block left live before
/// defining it itself. The FP-block prologue's `fp_in_regs` test clobbers flags,
/// so when this holds the prologue must save and restore them (`lahf`/`seto` …
/// `sahf`); otherwise it can skip that and just branch.
fn block_flags_live_in(body: &[Instruction], term: &Instruction) -> bool {
    let mut defined = 0u32;
    for i in body.iter().chain(std::iter::once(term)) {
        if i.rflags_read() & !defined != 0 {
            return true;
        }
        defined |= i.rflags_modified();
    }
    false
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
    /// `jrcxz`/`jecxz`/`loop`/`loope`/`loopne`: a conditional branch that exists
    /// only in a `rel8` encoding, so its taken edge cannot be a patchable `rel32`
    /// directly — it is lowered as the native instruction hopping over a jump
    /// pair (see [`build_linked_terminator`]). `opcode` is the `rel8` opcode byte
    /// (`E0`..`E3`); `prefix67` selects the 32-bit (`ecx`) register width.
    ShortCond {
        prefix67: bool,
        opcode: u8,
        taken: u64,
        fallthrough: u64,
    },
}

/// Classify a terminator as linkable, or `None` if it ends the block with a
/// runtime-determined target (indirect branch/call, return), or is a syscall.
fn classify_terminator(t: &Instruction) -> Option<LinkTerm> {
    if t.code() == Code::Syscall {
        return None;
    }
    match t.flow_control() {
        FlowControl::UnconditionalBranch => Some(LinkTerm::Uncond {
            target: t.near_branch_target(),
        }),
        FlowControl::ConditionalBranch => {
            if let Some((prefix67, opcode)) = short_cond_opcode(t.code()) {
                Some(LinkTerm::ShortCond {
                    prefix67,
                    opcode,
                    taken: t.near_branch_target(),
                    fallthrough: t.next_ip(),
                })
            } else {
                Some(LinkTerm::Cond {
                    opcode: jcc_opcode(t.code())?,
                    taken: t.near_branch_target(),
                    fallthrough: t.next_ip(),
                })
            }
        }
        FlowControl::Call => Some(LinkTerm::DirectCall {
            target: t.near_branch_target(),
            ret: t.next_ip(),
        }),
        _ => None,
    }
}

/// The `rel8`-only conditional branches: `(needs 67 prefix, opcode byte)`. They
/// test `rcx`/`ecx` (the `loop` forms also decrement it) and touch no flags
/// except `loope`/`loopne` reading ZF — all behavior the native instruction
/// reproduces exactly, so the lowering re-emits it verbatim.
fn short_cond_opcode(code: Code) -> Option<(bool, u8)> {
    Some(match code {
        Code::Loopne_rel8_64_RCX => (false, 0xE0),
        Code::Loopne_rel8_64_ECX => (true, 0xE0),
        Code::Loope_rel8_64_RCX => (false, 0xE1),
        Code::Loope_rel8_64_ECX => (true, 0xE1),
        Code::Loop_rel8_64_RCX => (false, 0xE2),
        Code::Loop_rel8_64_ECX => (true, 0xE2),
        Code::Jrcxz_rel8_64 => (false, 0xE3),
        Code::Jecxz_rel8_64 => (true, 0xE3),
        _ => return None,
    })
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
        LinkTerm::ShortCond {
            prefix67,
            opcode,
            taken,
            fallthrough,
        } => {
            // No `rel32` form exists, so hop: the native `rel8` instruction skips
            // the short `jmp` below to land on the taken path; not-taken falls
            // into that short `jmp`, which reaches the fall-through path. The
            // native instruction supplies the exact architectural behavior
            // (`loop`'s rcx decrement, `loope`/`loopne`'s ZF read) and clobbers no
            // flags, so the two edges link like any other direct branch.
            if prefix67 {
                out.push(0x67);
            }
            // rel8 = +2: taken skips the 2-byte `EB` short jmp that follows.
            out.extend_from_slice(&[opcode, 0x02]);
            out.push(0xEB);
            let eb_at = out.len();
            out.push(0x00); // disp8 to fall-through path, patched below
            // Taken path. `loop`/`jrcxz` are loop primitives, so the taken edge
            // usually closes a loop: route it through the flag- and rcx-preserving
            // safepoint poll (the native instruction has already read/decremented
            // rcx and read ZF, so polling after is safe).
            let poll = is_back_edge(taken, block_start).then(|| emit_exit_poll(&mut out));
            pad_rel32_alignment(&mut out, term_pc, 1);
            out.push(0xE9);
            let taken_rel = take_rel32(&mut out);
            // Fall-through path — the `EB` above lands here.
            let fall_path = out.len();
            let disp = fall_path as i64 - (eb_at as i64 + 1);
            out[eb_at] = i8::try_from(disp).expect("shortcond jmp displacement out of range") as u8;
            pad_rel32_alignment(&mut out, term_pc, 1);
            out.push(0xE9);
            let fall_rel = take_rel32(&mut out);
            let taken_stub = out.len();
            emit_stub(&mut out, taken, exit_tramp);
            let fall_stub = out.len();
            emit_stub(&mut out, fallthrough, exit_tramp);
            write_rel32(&mut out, taken_rel, taken_stub);
            write_rel32(&mut out, fall_rel, fall_stub);
            if let Some(poll_rel) = poll {
                write_rel32(&mut out, poll_rel, taken_stub);
            }
            edges.push((taken_rel, taken));
            edges.push((fall_rel, fallthrough));
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

/// Build the lazy-install prologue emitted at the head of a block that needs
/// the guest's FP/SIMD state, FS base, or both. Each is guarded by its
/// `ts.fp_in_regs` / `ts.fs_is_guest` flag: if an earlier block this residency
/// already installed it the check falls straight through; otherwise this block
/// installs it (XRSTOR of `fpstate`; `wrfsbase` of `guest_fs_base`) and sets the
/// flag. `dispatch` clears both flags on every cache entry, so the first needing
/// block of a residency does the install and the rest skip it; the exit
/// trampolines mirror the flags to decide what to undo (see `trampoline.S`).
///
/// Each install borrows rax (and the FP path borrows rdx, which `xrstor64`'s
/// `edx` mask half clobbers), parked in gs slots and reloaded. When the block
/// reads a predecessor's flags (`flags_live_in`) the guard `cmp`s would destroy
/// them, so the whole prologue is wrapped in a `lahf`/`seto` … `add al,0x7f`/
/// `sahf` save/restore — the same dance the indirect-branch lookup uses — and
/// rax is parked once across it; otherwise the cheap path leaves flags dead and
/// saves rax only inside whichever install actually runs, so an already-
/// installed fast path is just `cmp`/`jne` per guard.
fn build_prologue(needs: BlockNeeds, flags_live_in: bool) -> Vec<u8> {
    let d_fp = offset_of!(ThreadState, fp_in_regs) as i32;
    let d_fps = offset_of!(ThreadState, fpstate) as i32;
    let d_flags = offset_of!(ThreadState, fp_flags) as i32;
    let d_scr = offset_of!(ThreadState, fp_scratch) as i32;
    let d_fs = offset_of!(ThreadState, fs_is_guest) as i32;
    let d_guest_fs = offset_of!(ThreadState, guest_fs_base) as i32;
    let mut out = Vec::new();

    if flags_live_in {
        // Park rax and the guest flags up front; the guards' cmp may then
        // clobber flags, and the installs may clobber rax, freely.
        gs_store(&mut out, MODRM_RAX, RAX_SLOT); // mov gs:[rax], rax
        out.push(0x9f); // lahf
        out.extend_from_slice(&[0x0f, 0x90, 0xc0]); // seto al
        gs_store(&mut out, MODRM_RAX, d_flags); // mov gs:[fp_flags], rax
        if needs.fp {
            emit_guarded(&mut out, d_fp, |o| emit_fp_restore(o, d_fps, d_scr, d_fp));
        }
        if needs.fs {
            emit_guarded(&mut out, d_fs, |o| emit_fs_install(o, d_guest_fs, d_fs));
        }
        emit_restore_flags(&mut out, d_flags); // rax<-flags; add al,0x7f; sahf
        gs_load(&mut out, MODRM_RAX, RAX_SLOT); // mov rax, gs:[rax]
    } else {
        // Flags are dead, so each guard's cmp clobbers them harmlessly and rax
        // is saved only in the install path that actually runs.
        if needs.fp {
            emit_guarded(&mut out, d_fp, |o| {
                gs_store(o, MODRM_RAX, RAX_SLOT); // mov gs:[rax], rax
                emit_fp_restore(o, d_fps, d_scr, d_fp);
                gs_load(o, MODRM_RAX, RAX_SLOT); // mov rax, gs:[rax]
            });
        }
        if needs.fs {
            emit_guarded(&mut out, d_fs, |o| {
                gs_store(o, MODRM_RAX, RAX_SLOT); // mov gs:[rax], rax
                emit_fs_install(o, d_guest_fs, d_fs);
                gs_load(o, MODRM_RAX, RAX_SLOT); // mov rax, gs:[rax]
            });
        }
    }
    out
}

/// Emit `cmp byte gs:[flag], 0; jne skip; <body>; skip:` — the install runs only
/// when the flag is clear. The `jne` is a short `rel8`; an install is well under
/// 128 bytes.
fn emit_guarded(out: &mut Vec<u8>, flag_disp: i32, body: impl FnOnce(&mut Vec<u8>)) {
    cmp_gs_byte_zero(out, flag_disp);
    out.push(0x75); // jne skip
    let jne = out.len();
    out.push(0);
    body(out);
    let skip = out.len();
    out[jne] = jcc_rel8(jne, skip);
}

/// Emit the FP restore proper: park rdx, restore the full extended state from
/// `gs:[fpstate]` via `xrstor64` (mask `0xe7` in edx:eax), reload rdx, and set
/// `fp_in_regs`. Assumes rax is already saved (it is loaded with the mask's low
/// half) and that the caller reloads rax after.
fn emit_fp_restore(out: &mut Vec<u8>, d_fps: i32, d_scr: i32, d_in: i32) {
    gs_store(out, MODRM_RDX, d_scr); // mov gs:[fp_scratch], rdx
    out.extend_from_slice(&[0xb8, 0xe7, 0x00, 0x00, 0x00]); // mov eax, 0xe7
    out.extend_from_slice(&[0x31, 0xd2]); // xor edx, edx
    out.extend_from_slice(&[0x65, 0x48, 0x0f, 0xae, 0x2c, 0x25]); // xrstor64 gs:[
    emit_u32(out, d_fps as u32); //   fpstate]
    gs_load(out, MODRM_RDX, d_scr); // mov rdx, gs:[fp_scratch]
    mov_gs_byte_one(out, d_in); // mov byte gs:[fp_in_regs], 1
}

/// Emit the FS-base install: load the guest base from `gs:[guest_fs_base]` into
/// rax, `wrfsbase` it, and set `fs_is_guest`. Assumes rax is already saved and
/// reloaded by the caller.
fn emit_fs_install(out: &mut Vec<u8>, d_guest_fs: i32, d_fs: i32) {
    gs_load(out, MODRM_RAX, d_guest_fs); // mov rax, gs:[guest_fs_base]
    out.extend_from_slice(&[0xf3, 0x48, 0x0f, 0xae, 0xd0]); // wrfsbase rax
    mov_gs_byte_one(out, d_fs); // mov byte gs:[fs_is_guest], 1
}

/// `cmp byte ptr gs:[disp32], 0` — `65 80 3c 25 <disp32> 00`.
fn cmp_gs_byte_zero(out: &mut Vec<u8>, disp: i32) {
    out.extend_from_slice(&[0x65, 0x80, 0x3c, 0x25]);
    emit_u32(out, disp as u32);
    out.push(0x00);
}

/// `mov byte ptr gs:[disp32], 1` — `65 c6 04 25 <disp32> 01`.
fn mov_gs_byte_one(out: &mut Vec<u8>, disp: i32) {
    out.extend_from_slice(&[0x65, 0xc6, 0x04, 0x25]);
    emit_u32(out, disp as u32);
    out.push(0x01);
}

/// The `rel8` byte for a short `jcc`/`jmp` whose displacement field is at
/// `from` (one byte) and whose target is `to`, both offsets within the buffer.
fn jcc_rel8(from: usize, to: usize) -> u8 {
    let disp = to as i64 - (from as i64 + 1);
    i8::try_from(disp).expect("fp prologue jump out of rel8 range") as u8
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
    trap_tramp: u64,
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
    // `int3` (and `int1`): a software breakpoint. The instruction does not run
    // in the cache; it exits through `exit_trap`, which sets `exit_kind = TRAP`
    // so the run loop raises SIGTRAP. As a trap (not a fault), the resumed rip is
    // the instruction *after* the breakpoint, exactly where a real `int3` leaves
    // it, so the saved next guest PC is `next_ip`.
    if matches!(t.code(), Code::Int3 | Code::Int1) {
        emit_save_rax(instrs)?;
        emit_load_rax_imm(instrs, next_ip)?;
        return emit_exit_tail(instrs, trap_tramp);
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
            // Far returns (`retf`) also pop CS; no 64-bit userspace code uses
            // them, so reject them rather than silently mis-execute. (The
            // 66-prefixed near forms decode to the same near-return codes and
            // are rejected from the raw bytes in `translate`.)
            if !matches!(t.code(), Code::Retnq | Code::Retnq_imm16) {
                return Err(Error::Translate(format!(
                    "unhandled return form at {:#x}: {:?}",
                    t.ip(),
                    t.code(),
                )));
            }
            emit_save_rax(instrs)?;
            emit_pop_rax(instrs)?;
            // `ret imm16` releases imm16 more bytes of stack after popping the
            // return address — callee-popped arguments (V8 builtins use this
            // form). Drop them with `lea`, which, like `ret`, leaves the
            // arithmetic flags untouched.
            if t.code() == Code::Retnq_imm16 {
                let imm = t.immediate16() as i64;
                instrs.push(mkinstr(Instruction::with2(
                    Code::Lea_r64_m,
                    Register::RSP,
                    MemoryOperand::with_base_displ(Register::RSP, imm),
                ))?);
            }
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
            // block_start 0: every target here is a forward edge (no safepoint
            // poll), exercising the bare linked-terminator alignment.
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
