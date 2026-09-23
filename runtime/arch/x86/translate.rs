//! Basic-block translator. Decodes a single guest basic block, copies the
//! straight-line prefix into the code cache (with RIP-relative operands
//! fixed up by `BlockEncoder`), and rewrites the terminator into a
//! "compute next guest PC, then exit to the dispatcher" sequence.

use std::{
    arch::asm,
    mem::offset_of,
    ptr,
    sync::atomic::{AtomicI32, AtomicPtr, AtomicUsize, Ordering},
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
const PKEY_DISABLE_ACCESS: u32 = 0x1;
const PKEY_DISABLE_WRITE: u32 = 0x2;
const PKEY_UNINIT: i32 = -1;
const PKEY_ALLOCATING: i32 = -2;
/// Probed and found absent: no CPU PKU, no `CONFIG_PKEYS`, or the key pool is
/// exhausted. The cache then runs unguarded rather than refusing to start.
const PKEY_UNSUPPORTED: i32 = -3;

/// `gs:[]` displacement of the guest's rbx slot (`regs[1]`). Terminators that
/// need a scratch memory slot borrow it: `exit_block` re-saves the live rbx
/// over the slot on the way out, so the guest's rbx register is preserved.
const RBX_SLOT: i64 = 8;

/// Host bounds `[lo, hi)` of the one translated-code buffer, published when it is
/// mapped and read by the synchronous fault handler to classify a fault.
static CODE_CACHE_LO: AtomicUsize = AtomicUsize::new(0);
static CODE_CACHE_HI: AtomicUsize = AtomicUsize::new(0);
/// A protection key is process-global and can label every code-cache mapping;
/// keeping one until process exit avoids both pkey-pool churn and cache Drop
/// racing a concurrent cache construction.
static CODE_CACHE_PKEY: AtomicI32 = AtomicI32::new(PKEY_UNINIT);

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

/// Host-rip layout of the shared inline indirect-branch lookup routine,
/// published when it is emitted so the preemption code can recover the guest
/// state of a thread interrupted inside it: the routine's bounds `[lo, hi)`,
/// the boundary past which every scratch slot it borrows has been written
/// (`stashed`), and its miss label, which restores the guest registers from
/// those slots and exits to the dispatcher with the branch target as the
/// next guest PC.
static IB_LOOKUP_LO: AtomicUsize = AtomicUsize::new(0);
static IB_LOOKUP_STASHED: AtomicUsize = AtomicUsize::new(0);
static IB_LOOKUP_MISS: AtomicUsize = AtomicUsize::new(0);
static IB_LOOKUP_HI: AtomicUsize = AtomicUsize::new(0);

/// See [`IB_LOOKUP_LO`]. `None` until the routine has been emitted.
pub struct IbLookupSpan {
    pub lo: usize,
    pub stashed: usize,
    pub miss: usize,
    pub hi: usize,
}

pub fn ib_lookup_span() -> Option<IbLookupSpan> {
    let lo = IB_LOOKUP_LO.load(Ordering::Acquire);
    (lo != 0).then(|| IbLookupSpan {
        lo,
        stashed: IB_LOOKUP_STASHED.load(Ordering::Relaxed),
        miss: IB_LOOKUP_MISS.load(Ordering::Relaxed),
        hi: IB_LOOKUP_HI.load(Ordering::Relaxed),
    })
}

/// The preemption metadata of the one code cache: a sorted, append-only
/// index of every translated block ([`IndexEntry`], one per block, in host
/// address order, which bump allocation makes the order of translation) and
/// the arena holding each block's [`BlockMeta`]. Both are read by the host
/// signal catcher, on a thread that is executing translated code while a
/// sibling may be translating, so they are published lock-free: a block's
/// metadata and index entry are written before the block is reachable, and
/// the index length is the release point. Entries are never reclaimed short
/// of a cache reset — a dropped block's code stays in place for a thread
/// already in flight through it, and so does its metadata.
static META_INDEX: AtomicPtr<IndexEntry> = AtomicPtr::new(ptr::null_mut());
static META_INDEX_LEN: AtomicUsize = AtomicUsize::new(0);
static META_ARENA: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());

/// One translated block in the preemption index: where its host code starts
/// (as an offset into the cache buffer) and where its [`BlockMeta`] lives (as
/// an offset into the metadata arena).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IndexEntry {
    host_off: u32,
    meta_off: u32,
}

/// Per-block metadata the preemption code needs to recover a precise guest
/// state from a host rip anywhere inside the block's host code, followed in
/// the arena by its encoded [`Entry`] list (see [`Entry::encode`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockMeta {
    /// Guest PC of the block's first instruction.
    pub guest_pc: u64,
    /// Guest PC of the terminator instruction.
    pub term_ip: u64,
    /// The terminator's taken successor (a branch target or call target), for
    /// the recipes that resume there.
    pub taken: u64,
    /// The terminator's fall-through successor, for the recipes that resume there.
    pub fall: u64,
    /// Total host bytes of the block: prologue, body, terminator, stubs.
    pub host_len: u32,
    /// Byte length of the encoded entry list following this header.
    pub entries_len: u32,
    /// The `imm16` of a `ret imm16` terminator: the stack bytes its `lea`
    /// releases after the pop.
    pub rsp_adj: u16,
}

/// Recovery recipe codes, one per [`Entry`]. A code names, for every
/// instruction boundary inside the entry's host span, how the precise guest
/// state relates to the interrupted host state. Every code below `PRO`
/// describes a body instruction: the guest registers are the host registers
/// (less the entry's `fix` register, if any, parked in `riprel_scratch`) and
/// the guest PC is the one accumulated over the preceding body entries.
pub mod recipe {
    /// A body instruction; the guest PC advances by `guest_len` past it.
    pub const BODY: u8 = 0x00;
    /// Inside a block prologue: the guest PC is the block's own, and the low
    /// bits name which registers are parked in their prologue slots.
    pub const PRO: u8 = 0x40;
    /// `PRO` flag: guest rax is in the rax slot (`regs[0]`).
    pub const PRO_RAX: u8 = 0x01;
    /// `PRO` flag: guest rdx is in `fp_scratch`.
    pub const PRO_RDX: u8 = 0x02;
    /// `PRO` flag: the guest status flags are in `fp_flags` (`lahf`/`seto` form).
    pub const PRO_FLAGS: u8 = 0x04;
    /// Already bound for the dispatcher (a cold exit stub, a syscall or `int3`
    /// exit sequence): leave the thread alone, it reaches the run loop on its own.
    pub const FLOW: u8 = 0x80;
    /// Guest registers are the host registers; resume at the terminator.
    pub const PRECISE_T: u8 = 0x81;
    /// Guest registers are the host registers; resume at `taken`.
    pub const PRECISE_TAKEN: u8 = 0x82;
    /// Guest registers are the host registers; resume at `fall`.
    pub const PRECISE_FALL: u8 = 0x83;
    /// Guest rax is in the rax slot; resume at the terminator.
    pub const RAXSLOT_T: u8 = 0x84;
    /// Guest rax is in the rax slot; resume at `taken`.
    pub const RAXSLOT_TAKEN: u8 = 0x85;
    /// Guest rax is in the rax slot; resume at the PC in the rbx slot
    /// (`regs[1]`, where an indirect call parks its target across the push).
    pub const RAXSLOT_RIP_RBXSLOT: u8 = 0x86;
    /// Guest rax is in the rax slot; resume at the PC in the host rax (a
    /// popped return address).
    pub const RAXSLOT_RIP_RAX: u8 = 0x87;
    /// [`RAXSLOT_RIP_RAX`], and the guest rsp is the host rsp plus `rsp_adj`
    /// (a `ret imm16` interrupted between its pop and its stack release).
    pub const RAXSLOT_RIP_RAX_RSPADJ: u8 = 0x88;
}

/// One span of a block's host code sharing a recovery recipe. A body
/// instruction is one entry; a run of non-body instructions with the same
/// recipe (a stub, a padded branch) may share one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub host_len: u8,
    /// Guest bytes the PC advances past this span (body entries only).
    pub guest_len: u8,
    pub code: u8,
    /// `1 + ThreadState register index` of a register parked in
    /// `riprel_scratch` across this span (a far-RIP-relative rewrite), or 0.
    pub fix: u8,
}

impl Entry {
    /// Append the encoded form: one byte (`1..=15`) for the common body
    /// instruction whose host and guest lengths agree and which parks nothing,
    /// otherwise a zero byte followed by the four fields.
    pub fn encode(&self, out: &mut Vec<u8>) {
        if self.code == recipe::BODY
            && self.fix == 0
            && self.host_len == self.guest_len
            && (1..=15).contains(&self.host_len)
        {
            out.push(self.host_len);
        } else {
            out.extend_from_slice(&[0, self.host_len, self.guest_len, self.code, self.fix]);
        }
    }

    /// Decode one entry off the front of `bytes`, returning it and the rest.
    pub fn decode(bytes: &[u8]) -> Option<(Entry, &[u8])> {
        let (&b, rest) = bytes.split_first()?;
        if b != 0 {
            return Some((
                Entry {
                    host_len: b,
                    guest_len: b,
                    code: recipe::BODY,
                    fix: 0,
                },
                rest,
            ));
        }
        if rest.len() < 4 {
            return None;
        }
        Some((
            Entry {
                host_len: rest[0],
                guest_len: rest[1],
                code: rest[2],
                fix: rest[3],
            },
            &rest[4..],
        ))
    }
}

/// Append a non-body entry covering `out[start..]`, merging it into the
/// previous entry when that one carries the same recipe and the merged span
/// still fits; every instruction boundary inside a non-body span recovers
/// identically, so a span's granularity is free.
fn mark(entries: &mut Vec<Entry>, out: &[u8], start: usize, code: u8) {
    debug_assert!(code >= recipe::PRO);
    let len = out.len() - start;
    if len == 0 {
        return;
    }
    if let Some(last) = entries.last_mut()
        && last.code == code
        && last.fix == 0
        && last.host_len as usize + len <= u8::MAX as usize
    {
        last.host_len += len as u8;
        return;
    }
    entries.push(Entry {
        host_len: u8::try_from(len).expect("non-body span exceeds 255 bytes"),
        guest_len: 0,
        code,
        fix: 0,
    });
}

/// Locate the entry of a block covering host offset `off`, given the block's
/// encoded entry list and its starting guest PC. Returns the entry, the guest
/// PC at the entry's first instruction boundary, and whether `off` is exactly
/// that boundary (a body entry is one instruction, so any other offset inside
/// it is not a boundary at all).
pub fn walk_entries(mut bytes: &[u8], off: usize, mut guest_pc: u64) -> Option<(Entry, u64, bool)> {
    let mut cur = 0usize;
    while !bytes.is_empty() {
        let (e, rest) = Entry::decode(bytes)?;
        bytes = rest;
        let end = cur + e.host_len as usize;
        if off < end {
            return Some((e, guest_pc, off == cur));
        }
        cur = end;
        if e.code < recipe::PRO {
            guest_pc = guest_pc.wrapping_add(e.guest_len as u64);
        }
    }
    None
}

/// Find the translated block whose host code contains `rip`: its metadata,
/// its encoded entry list, and `rip`'s offset from the block's host start.
/// Lock-free, for the host signal catcher: a binary search over the published
/// prefix of the index. `None` for a rip in the cache but inside no block (the
/// inline lookup routine, or bytes nothing has been aimed at).
pub fn lookup_block(rip: usize) -> Option<(&'static BlockMeta, &'static [u8], usize)> {
    let base = CODE_CACHE_LO.load(Ordering::Relaxed);
    let len = META_INDEX_LEN.load(Ordering::Acquire);
    let index = META_INDEX.load(Ordering::Relaxed);
    let arena = META_ARENA.load(Ordering::Relaxed);
    if len == 0 || index.is_null() || arena.is_null() || rip < base {
        return None;
    }
    let off = (rip - base) as u32;
    let entries = unsafe { std::slice::from_raw_parts(index, len) };
    // Last entry whose start is at or before `off`.
    let i = entries
        .partition_point(|e| e.host_off <= off)
        .checked_sub(1)?;
    let e = entries[i];
    let meta = unsafe { &*(arena.add(e.meta_off as usize) as *const BlockMeta) };
    let in_block = off - e.host_off;
    if in_block >= meta.host_len {
        return None;
    }
    let bytes = unsafe {
        let p = arena.add(e.meta_off as usize + std::mem::size_of::<BlockMeta>());
        std::slice::from_raw_parts(p, meta.entries_len as usize)
    };
    Some((meta, bytes, in_block as usize))
}

/// Index capacity in entries for a cache of `size` bytes: no translated block
/// is shorter than 16 bytes (the smallest is an empty-body indirect jump:
/// save rax, `jmp gs:[ib_lookup]`), so this bounds the block count.
fn meta_index_cap(size: usize) -> usize {
    size / 16 + 1
}

/// Metadata arena bytes for a cache of `size` bytes. A block's metadata is a
/// 40-byte header plus roughly one byte per host instruction, well under its
/// code size except for the pathological cache of minimal blocks; twice the
/// code size covers that, and the reservation is virtual (`MAP_NORESERVE`).
fn meta_arena_size(size: usize) -> usize {
    size * 2
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
    pkey: Option<i32>,
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
    /// The preemption index and metadata arena (see [`META_INDEX`]); both
    /// `mmap`'d lazily alongside the code buffer and bump-allocated with it.
    index: *mut IndexEntry,
    index_cap: usize,
    arena: *mut u8,
    arena_size: usize,
    arena_used: usize,
}

impl CodeCache {
    /// Create a code cache backed by a `size`-byte RWX region, with guest
    /// writes denied per thread through an x86 protection key. The region is
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
        let cache_prot = libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;
        if unsafe { libc::mprotect(p, size, cache_prot) } != 0 {
            let err = Error::last_os_error("code cache mprotect");
            unsafe { libc::munmap(region, map_size) };
            return Err(err);
        }
        // A host without protection-key support runs the cache RWX and unguarded
        // rather than refusing to start; the tag only matters when a key exists.
        let pkey = acquire_pkey();
        if let Some(pkey) = pkey
            && let Err(err) = pkey_mprotect(p, size, cache_prot, pkey)
        {
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
        let index_cap = meta_index_cap(size);
        let arena_size = meta_arena_size(size);
        let side = |bytes: usize, what: &str| -> Result<*mut u8, Error> {
            let m = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if m == libc::MAP_FAILED {
                return Err(Error::last_os_error(what));
            }
            Ok(m as *mut u8)
        };
        let index = match side(
            index_cap * std::mem::size_of::<IndexEntry>(),
            "block index mmap",
        ) {
            Ok(m) => m as *mut IndexEntry,
            Err(err) => {
                unsafe {
                    libc::munmap(t, IB_TABLE_BYTES);
                    libc::munmap(region, map_size);
                }
                return Err(err);
            }
        };
        let arena = match side(arena_size, "block metadata mmap") {
            Ok(m) => m,
            Err(err) => {
                unsafe {
                    libc::munmap(index.cast(), index_cap * std::mem::size_of::<IndexEntry>());
                    libc::munmap(t, IB_TABLE_BYTES);
                    libc::munmap(region, map_size);
                }
                return Err(err);
            }
        };
        let cache = Self {
            base: p as *mut u8,
            size,
            pkey,
            map_base: region as *mut u8,
            map_size,
            used: 0,
            ib_table: t as *mut u8,
            ib_lookup: None,
            index,
            index_cap,
            arena,
            arena_size,
            arena_used: 0,
        };
        cache.clear_ib_table();
        // Publish the buffer bounds for the fault handler's in-cache check and
        // the preemption tables for the signal catcher. One CodeCache backs the
        // process (reset rewinds it rather than remapping), so this is set once.
        META_INDEX_LEN.store(0, Ordering::Relaxed);
        META_INDEX.store(index, Ordering::Relaxed);
        META_ARENA.store(arena, Ordering::Relaxed);
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

    /// Publish the preemption metadata of the block just emitted at `host_pc`:
    /// its header and encoded entry list go into the arena, its index entry is
    /// appended, and the index length is released last, so a catcher that
    /// sees the entry sees everything it points at. Called before the block
    /// becomes reachable (before it is mapped, linked, or mirrored into the
    /// indirect-branch table), so no thread can be interrupted inside a block
    /// the index does not yet cover.
    fn record_block(
        &mut self,
        host_pc: u64,
        meta: &BlockMeta,
        entries: &[Entry],
    ) -> Result<(), Error> {
        let mut meta = *meta;
        let mut encoded = Vec::with_capacity(entries.len());
        for e in entries {
            e.encode(&mut encoded);
        }
        meta.entries_len = encoded.len() as u32;
        let header = std::mem::size_of::<BlockMeta>();
        let start = self
            .arena_used
            .next_multiple_of(std::mem::align_of::<BlockMeta>());
        let end = start + header + encoded.len();
        let len = META_INDEX_LEN.load(Ordering::Relaxed);
        if end > self.arena_size || len >= self.index_cap {
            return Err(Error::CodeCacheExhausted);
        }
        unsafe {
            ptr::write(self.arena.add(start) as *mut BlockMeta, meta);
            ptr::copy_nonoverlapping(
                encoded.as_ptr(),
                self.arena.add(start + header),
                encoded.len(),
            );
            ptr::write(
                self.index.add(len),
                IndexEntry {
                    host_off: (host_pc - self.base as u64) as u32,
                    meta_off: start as u32,
                },
            );
        }
        self.arena_used = end;
        META_INDEX_LEN.store(len + 1, Ordering::Release);
        Ok(())
    }

    pub fn contains_range(&self, start: usize, len: usize) -> bool {
        let end = start.saturating_add(len);
        let lo = self.base as usize;
        let hi = lo + self.size;
        start < hi && end > lo
    }

    pub fn allow_writes(&self) {
        if let Some(pkey) = self.pkey {
            set_pkey_write_disabled(pkey, false);
        }
    }

    pub fn deny_writes(&self) {
        if let Some(pkey) = self.pkey {
            set_pkey_write_disabled(pkey, true);
        }
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
        // Every slot the miss path restores from is written from here on, so
        // a thread interrupted at or past this boundary can be redirected to
        // `miss` (see `preempt`); before it, the target is still in rax (at the
        // routine's first instruction) or in its slot, and every other guest
        // register is live.
        let stashed = out.len();
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
        IB_LOOKUP_STASHED.store(addr as usize + stashed, Ordering::Relaxed);
        IB_LOOKUP_MISS.store(addr as usize + miss, Ordering::Relaxed);
        IB_LOOKUP_HI.store(addr as usize + out.len(), Ordering::Relaxed);
        IB_LOOKUP_LO.store(addr as usize, Ordering::Release);
        Ok(addr)
    }

    pub fn reset(&mut self) {
        self.used = 0;
        self.ib_lookup = None;
        self.clear_ib_table();
        self.arena_used = 0;
        META_INDEX_LEN.store(0, Ordering::Release);
        IB_LOOKUP_LO.store(0, Ordering::Release);
    }
}

impl Drop for CodeCache {
    fn drop(&mut self) {
        // Unmap the whole reservation (both guards plus the buffer).
        let ret = unsafe { libc::munmap(self.map_base.cast(), self.map_size) };
        debug_assert_eq!(ret, 0, "code cache munmap failed");
        let ret = unsafe { libc::munmap(self.ib_table.cast(), IB_TABLE_BYTES) };
        debug_assert_eq!(ret, 0, "ib table munmap failed");
        let ret = unsafe {
            libc::munmap(
                self.index.cast(),
                self.index_cap * std::mem::size_of::<IndexEntry>(),
            )
        };
        debug_assert_eq!(ret, 0, "block index munmap failed");
        let ret = unsafe { libc::munmap(self.arena.cast(), self.arena_size) };
        debug_assert_eq!(ret, 0, "block metadata munmap failed");
    }
}

/// Resolve the process-global code-cache protection key, probing kernel support
/// exactly once. `None` means protection keys are unavailable, and the caller
/// leaves the cache unguarded rather than failing to start.
fn acquire_pkey() -> Option<i32> {
    loop {
        match CODE_CACHE_PKEY.load(Ordering::Acquire) {
            pkey if pkey >= 0 => return Some(pkey),
            PKEY_UNSUPPORTED => return None,
            PKEY_UNINIT => {
                if CODE_CACHE_PKEY
                    .compare_exchange(
                        PKEY_UNINIT,
                        PKEY_ALLOCATING,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_err()
                {
                    continue;
                }
                let Some(pkey) = allocate_pkey() else {
                    CODE_CACHE_PKEY.store(PKEY_UNSUPPORTED, Ordering::Release);
                    return None;
                };
                CODE_CACHE_PKEY.store(pkey, Ordering::Release);
                return Some(pkey);
            }
            PKEY_ALLOCATING => std::hint::spin_loop(),
            _ => unreachable!("invalid code cache pkey state"),
        }
    }
}

fn allocate_pkey() -> Option<i32> {
    let raw = unsafe { libc::syscall(libc::SYS_pkey_alloc, 0, 0) };
    (raw >= 0).then_some(raw as i32)
}

pub fn mpk_enabled() -> bool {
    let Some(pkey) = allocate_pkey() else {
        return false;
    };
    unsafe { libc::syscall(libc::SYS_pkey_free, pkey) };
    true
}

fn pkey_mprotect(
    addr: *mut libc::c_void,
    len: usize,
    prot: libc::c_int,
    pkey: i32,
) -> Result<(), Error> {
    let ret = unsafe { libc::syscall(libc::SYS_pkey_mprotect, addr, len, prot, pkey) };
    if ret != 0 {
        return Err(Error::last_os_error("code cache pkey_mprotect"));
    }
    Ok(())
}

fn set_pkey_write_disabled(pkey: i32, disabled: bool) {
    let shift = (pkey as u32) * 2;
    let mask = (PKEY_DISABLE_ACCESS | PKEY_DISABLE_WRITE) << shift;
    let write = PKEY_DISABLE_WRITE << shift;
    let mut pkru = read_pkru() & !mask;
    if disabled {
        pkru |= write;
    }
    write_pkru(pkru);
}

fn read_pkru() -> u32 {
    let eax: u32;
    let edx: u32;
    unsafe {
        asm!(
            "rdpkru",
            in("ecx") 0_u32,
            lateout("eax") eax,
            lateout("edx") edx,
            options(nomem, nostack, preserves_flags),
        );
    }
    ((edx as u64) << 32 | eax as u64) as u32
}

fn write_pkru(pkru: u32) {
    unsafe {
        asm!(
            "wrpkru",
            in("eax") pkru,
            in("ecx") 0_u32,
            in("edx") 0_u32,
            options(nomem, nostack, preserves_flags),
        );
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
/// statically known successor, the address of the `rel32` displacement field
/// of the direct branch that currently targets the block's cold exit stub, and
/// the address of that stub. Once the successor is translated, the dispatcher
/// rewrites the displacement so the branch jumps straight into the successor's
/// host code, keeping the guest register file live across the edge instead of
/// round-tripping through the dispatcher; when the successor is invalidated,
/// the dispatcher rewrites it back to `stub`, severing the edge with the same
/// single atomic store. See [`super::super::sys::mmap`].
pub struct OutEdge {
    pub target_guest: u64,
    pub site: usize,
    pub stub: u64,
}

/// The result of translating one basic block: where its host code begins, the
/// guest PC one past its last decoded byte (so the cache knows which guest pages
/// the block covers, for self-modifying-code invalidation), and its statically
/// known outgoing edges for the dispatcher to link.
pub struct Translation {
    pub host_pc: u64,
    pub guest_end: u64,
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

fn reject_guest_control_instruction(instr: &Instruction) -> Result<(), Error> {
    match instr.code() {
        Code::Wrpkru => Err(Error::Translate(format!(
            "guest WRPKRU is not supported at {:#x}",
            instr.ip()
        ))),
        Code::Sysenter | Code::Sysexitd | Code::Sysexitq => Err(Error::Translate(format!(
            "guest {:?} is not supported at {:#x}",
            instr.mnemonic(),
            instr.ip()
        ))),
        _ => Ok(()),
    }
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
            // The synthesized jump sits at the split point: a thread preempted
            // at its branch resumes at the next guest PC, which is also its
            // target, so either reading of "the terminator's PC" is the same.
            let mut jmp = mkinstr(Instruction::with_branch(Code::Jmp_rel32_64, next_pc))?;
            jmp.set_ip(next_pc);
            break jmp;
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
        reject_guest_control_instruction(&instr)?;
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
    // The block's preemption metadata: a recovery recipe for every span of the
    // host code emitted below (see `recipe`), recorded once the whole block is
    // in place.
    let mut entries = Vec::new();
    let mut meta = BlockMeta {
        guest_pc,
        term_ip: term.ip(),
        ..BlockMeta::default()
    };
    if needs.fp || needs.fs {
        let prologue = build_prologue(needs, block_flags_live_in(&instrs, &term), &mut entries);
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
        let codes = vec![recipe::BODY; instrs.len()];
        emit_body(cache, &instrs, body_pc, guest_pc, &codes, &mut entries)?;
        let term_pc = cache.next_pc() as usize;
        let (bytes, rel_edges) =
            build_linked_terminator(&link, exit_tramp, term_pc, &mut entries, &mut meta);
        cache.emit(&bytes)?;
        rel_edges
            .into_iter()
            .map(|(off, stub_off, target_guest)| OutEdge {
                target_guest,
                site: term_pc + off,
                stub: (term_pc + stub_off) as u64,
            })
            .collect()
    } else {
        let mut codes = vec![recipe::BODY; instrs.len()];
        emit_terminator(
            &mut instrs,
            &term,
            syscall_tramp,
            trap_tramp,
            &mut codes,
            &mut meta,
        )?;
        emit_body(cache, &instrs, body_pc, guest_pc, &codes, &mut entries)?;
        Vec::new()
    };
    meta.host_len = (cache.next_pc() - host_pc) as u32;
    cache.record_block(host_pc, &meta, &entries)?;

    Ok(Translation {
        host_pc,
        guest_end,
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
            let mut movabs = mkinstr(Instruction::with2(Code::Mov_r64_imm64, dest, target))?;
            // Keep the guest's own position and length on the replacement: the
            // preemption map advances the guest PC by each body instruction's
            // guest length, which a synthesized instruction does not carry.
            // (`set_ip` derives the stored next-IP from the current length, so
            // the length goes first.)
            movabs.set_len(instr.len());
            movabs.set_ip(instr.ip());
            *instr = movabs;
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
///
/// `codes` carries one recovery recipe per instruction in `instrs`:
/// [`recipe::BODY`] for a guest instruction, a terminator recipe for each
/// instruction of an appended exit sequence. One preemption [`Entry`] is
/// recorded per *encoded* instruction — a far-RIP-relative rewrite turns one
/// guest instruction into four host ones, each its own boundary — using the
/// offsets the encoder reports, since a fixed-up encoding need not keep the
/// guest's length.
fn emit_body(
    cache: &mut CodeCache,
    instrs: &[Instruction],
    host_pc: u64,
    guest_pc: u64,
    codes: &[u8],
    entries: &mut Vec<Entry>,
) -> Result<(), Error> {
    if instrs.is_empty() {
        return Ok(());
    }
    debug_assert_eq!(codes.len(), instrs.len());
    let rewritten = rewrite_far_rip_operands(instrs, host_pc)?;
    let plain_parts;
    let (encoded, parts): (&[Instruction], &[Part]) = match &rewritten {
        Some((list, parts)) => (list, parts),
        None => {
            plain_parts = (0..instrs.len()).map(Part::plain).collect::<Vec<_>>();
            (instrs, &plain_parts)
        }
    };
    let block = InstructionBlock::new(encoded, host_pc);
    let result = BlockEncoder::encode(
        64,
        block,
        BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
    )
    .map_err(|e| Error::Translate(format!("encode block at {:#x}: {}", guest_pc, e)))?;
    let total = result.code_buffer.len() as u32;
    let offsets = &result.new_instruction_offsets;
    for (j, part) in parts.iter().enumerate() {
        let start = offsets[j];
        let end = offsets.get(j + 1).copied().unwrap_or(total);
        if start == u32::MAX || end == u32::MAX {
            return Err(Error::Translate(format!(
                "encode block at {:#x}: instruction relocated",
                guest_pc
            )));
        }
        let code = codes[part.src];
        let guest_len = if code == recipe::BODY && part.op {
            instrs[part.src].len() as u8
        } else {
            0
        };
        entries.push(Entry {
            host_len: u8::try_from(end - start).expect("encoded instruction exceeds 255 bytes"),
            guest_len,
            code,
            fix: part.fix,
        });
    }
    cache.emit(&result.code_buffer)
}

/// A far-RIP-rewritten instruction list and one [`Part`] per instruction in it.
type Rewritten = (Vec<Instruction>, Vec<Part>);

/// How one encoded instruction relates to the source list `emit_body` was
/// given: which source instruction it came from, whether it is the source
/// instruction itself (so the guest PC advances past it) rather than a
/// register save or reload inserted around it, and the `Entry::fix` register
/// parked across its boundary.
#[derive(Clone, Copy)]
struct Part {
    src: usize,
    op: bool,
    fix: u8,
}

impl Part {
    fn plain(src: usize) -> Self {
        Self {
            src,
            op: true,
            fix: 0,
        }
    }
}

/// Slack subtracted from the `rel32` range when deciding whether a RIP-relative
/// target is still reachable once the instruction moves into the code cache. The
/// displacement is measured from the instruction's own position within the
/// encoded block, not from `host_pc`, and an encoded block is bounded far below
/// this (a block decodes at most [`MAX_BLOCK_GUEST_BYTES`] guest bytes).
const RIP_REACH_SLACK: u64 = 1 << 20;

fn rip_reachable(target: u64, host_pc: u64) -> bool {
    let disp = target as i128 - host_pc as i128;
    disp.unsigned_abs() < (i32::MAX as u64 - RIP_REACH_SLACK) as u128
}

/// Rewrite every RIP-relative memory operand whose effective address is out of
/// `rel32` range of the block's spot in the code cache — guest code mapped tens
/// of terabytes from the cache, such as a JIT region JavaScriptCore places at a
/// randomized high address. `BlockEncoder` preserves a RIP-relative operand's
/// effective address by adjusting its displacement, which fails outright for
/// such a target, so the access is re-expressed through a borrowed register
/// instead: stash the register in the `riprel_scratch` slot, materialize the
/// absolute guest address, run the original instruction with the register as
/// its base, and reload the register. None of the inserted `mov`s touch the
/// arithmetic flags, so flags stay live from the block's body into its
/// terminator, and the borrowed register's guest value is unreachable only
/// within the sequence itself: an SMC write fault inside it resumes at the
/// faulting store with all host registers intact, and any other fault there is
/// terminal.
///
/// Returns `None` when every operand is in range, which is every block outside
/// such far regions, so the common path encodes the original list untouched.
/// Otherwise the rewritten list comes with one [`Part`] per output
/// instruction. The borrowed register is unreachable only from the
/// materializing `movabs` onward — the save before it is a copy — so the parts
/// for the rewritten access and the reload name it as parked, and the access
/// itself is the part the guest PC advances past.
fn rewrite_far_rip_operands(
    instrs: &[Instruction],
    host_pc: u64,
) -> Result<Option<Rewritten>, Error> {
    let far = |i: &Instruction| {
        i.is_ip_rel_memory_operand() && !rip_reachable(i.ip_rel_memory_address(), host_pc)
    };
    if !instrs.iter().any(far) {
        return Ok(None);
    }
    let slot = offset_of!(ThreadState, riprel_scratch) as i64;
    let mut info_factory = InstructionInfoFactory::new();
    let mut out = Vec::with_capacity(instrs.len() + 3);
    let mut parts = Vec::with_capacity(instrs.len() + 3);
    for (src, instr) in instrs.iter().enumerate() {
        if !far(instr) {
            out.push(*instr);
            parts.push(Part::plain(src));
            continue;
        }
        let scratch = pick_scratch(&mut info_factory, instr)?;
        let fix = 1 + state_reg_index(scratch);
        let target = instr.ip_rel_memory_address();
        out.push(mkinstr(Instruction::with2(
            Code::Mov_rm64_r64,
            gs_qword(slot),
            scratch,
        ))?);
        parts.push(Part {
            src,
            op: false,
            fix: 0,
        });
        out.push(mkinstr(Instruction::with2(
            Code::Mov_r64_imm64,
            scratch,
            target,
        ))?);
        parts.push(Part {
            src,
            op: false,
            fix: 0,
        });
        let mut patched = *instr;
        patched.set_memory_base(scratch);
        patched.set_memory_displacement64(0);
        patched.set_memory_displ_size(0);
        out.push(patched);
        parts.push(Part { src, op: true, fix });
        out.push(mkinstr(Instruction::with2(
            Code::Mov_r64_rm64,
            scratch,
            gs_qword(slot),
        ))?);
        parts.push(Part {
            src,
            op: false,
            fix,
        });
    }
    Ok(Some((out, parts)))
}

/// The `ThreadState::regs` index of a general-purpose register (rax, rbx,
/// rcx, rdx, rsi, rdi, rbp, rsp, r8..r15).
fn state_reg_index(reg: Register) -> u8 {
    match reg {
        Register::RAX => 0,
        Register::RBX => 1,
        Register::RCX => 2,
        Register::RDX => 3,
        Register::RSI => 4,
        Register::RDI => 5,
        Register::RBP => 6,
        Register::RSP => 7,
        r => 8 + (r.number() - Register::R8.number()) as u8,
    }
}

/// Pick a register the far-RIP rewrite can borrow around `instr`: any GPR the
/// instruction neither reads nor writes, explicitly or implicitly (`mul m64`
/// consumes rax and rdx with no visible register operand). rsp is never a
/// candidate — the host fault handler runs on whatever stack rsp names — and
/// rbp/r13 are skipped so the zero-displacement base encodes uniformly. An
/// instruction has at most a handful of register operands, so the eight
/// candidates cannot all be taken.
fn pick_scratch(
    info_factory: &mut InstructionInfoFactory,
    instr: &Instruction,
) -> Result<Register, Error> {
    const CANDIDATES: [Register; 8] = [
        Register::RAX,
        Register::RCX,
        Register::RDX,
        Register::RBX,
        Register::RSI,
        Register::RDI,
        Register::R8,
        Register::R9,
    ];
    let info = info_factory.info(instr);
    let free = |c: &Register| {
        !info
            .used_registers()
            .iter()
            .any(|u| u.register().full_register() == *c)
    };
    CANDIDATES.iter().find(|c| free(c)).copied().ok_or_else(|| {
        Error::Translate(format!(
            "no scratch register for far RIP-relative operand at {:#x}",
            instr.ip()
        ))
    })
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

/// Build the raw machine code for a linkable terminator: a fast-path direct
/// branch followed by one cold exit stub per successor. Returns the encoded
/// bytes and, for each edge, the byte offsets of its fast-path `rel32`
/// displacement and of its cold exit stub, paired with the successor's guest
/// PC. The displacements are initialized to target the stubs; the dispatcher
/// rewrites them to the successor blocks as those are translated, and back to
/// the stubs when a successor is invalidated.
///
/// Every span gets a preemption entry: the fast-path branches (and their
/// alignment padding) resume at the terminator, or at the successor once the
/// branch's architectural effect has already happened (a `loop` has
/// decremented rcx, a call has pushed its return address); the stubs are
/// [`recipe::FLOW`], already bound for the dispatcher. `meta` receives the
/// successor PCs those recipes name.
///
/// `term_pc` is the host address at which these bytes will be emitted. It is
/// needed because every patchable `rel32` field is NOP-padded to a 4-byte
/// boundary, so the dispatcher can back-patch it with one aligned atomic store
/// (see [`super::cache::patch_site`]) — a sibling thread executing the branch
/// then reads the old or new target, never a torn displacement.
fn build_linked_terminator(
    link: &LinkTerm,
    exit_tramp: u64,
    term_pc: usize,
    entries: &mut Vec<Entry>,
    meta: &mut BlockMeta,
) -> (Vec<u8>, Vec<(usize, usize, u64)>) {
    let mut out = Vec::new();
    let mut edges = Vec::new();
    match *link {
        LinkTerm::Uncond { target } => {
            // jmp rel32 -> stub (later: -> target's host code)
            let at = out.len();
            pad_rel32_alignment(&mut out, term_pc, 1);
            out.push(0xE9);
            let rel = take_rel32(&mut out);
            mark(entries, &out, at, recipe::PRECISE_T);
            let stub = out.len();
            emit_stub(&mut out, target, exit_tramp);
            mark(entries, &out, stub, recipe::FLOW);
            write_rel32(&mut out, rel, stub);
            edges.push((rel, stub, target));
        }
        LinkTerm::Cond {
            opcode,
            taken,
            fallthrough,
        } => {
            // jcc rel32 -> taken stub ; jmp rel32 -> fall-through stub.
            // The native jcc reads the block's live guest flags directly. Each
            // branch is padded independently so both rel32 fields land aligned.
            // A thread preempted at the jmp has run the jcc untaken; resuming at
            // the jcc re-runs it on the same flags with the same outcome.
            meta.taken = taken;
            meta.fall = fallthrough;
            let at = out.len();
            pad_rel32_alignment(&mut out, term_pc, 2);
            out.extend_from_slice(&[0x0F, opcode]);
            let jcc_rel = take_rel32(&mut out);
            pad_rel32_alignment(&mut out, term_pc, 1);
            out.push(0xE9);
            let jmp_rel = take_rel32(&mut out);
            mark(entries, &out, at, recipe::PRECISE_T);
            let taken_stub = out.len();
            emit_stub(&mut out, taken, exit_tramp);
            let fall_stub = out.len();
            emit_stub(&mut out, fallthrough, exit_tramp);
            mark(entries, &out, taken_stub, recipe::FLOW);
            write_rel32(&mut out, jcc_rel, taken_stub);
            write_rel32(&mut out, jmp_rel, fall_stub);
            edges.push((jcc_rel, taken_stub, taken));
            edges.push((jmp_rel, fall_stub, fallthrough));
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
            // flags, so the two edges link like any other direct branch. Past the
            // native instruction its effect is done, so each path resumes at its
            // own successor rather than re-running it.
            meta.taken = taken;
            meta.fall = fallthrough;
            let at = out.len();
            if prefix67 {
                out.push(0x67);
            }
            // rel8 = +2: taken skips the 2-byte `EB` short jmp that follows.
            out.extend_from_slice(&[opcode, 0x02]);
            mark(entries, &out, at, recipe::PRECISE_T);
            let at = out.len();
            out.push(0xEB);
            let eb_at = out.len();
            out.push(0x00); // disp8 to fall-through path, patched below
            mark(entries, &out, at, recipe::PRECISE_FALL);
            // Taken path.
            let at = out.len();
            pad_rel32_alignment(&mut out, term_pc, 1);
            out.push(0xE9);
            let taken_rel = take_rel32(&mut out);
            mark(entries, &out, at, recipe::PRECISE_TAKEN);
            // Fall-through path — the `EB` above lands here.
            let fall_path = out.len();
            let disp = fall_path as i64 - (eb_at as i64 + 1);
            out[eb_at] = i8::try_from(disp).expect("shortcond jmp displacement out of range") as u8;
            let at = out.len();
            pad_rel32_alignment(&mut out, term_pc, 1);
            out.push(0xE9);
            let fall_rel = take_rel32(&mut out);
            mark(entries, &out, at, recipe::PRECISE_FALL);
            let taken_stub = out.len();
            emit_stub(&mut out, taken, exit_tramp);
            let fall_stub = out.len();
            emit_stub(&mut out, fallthrough, exit_tramp);
            mark(entries, &out, taken_stub, recipe::FLOW);
            write_rel32(&mut out, taken_rel, taken_stub);
            write_rel32(&mut out, fall_rel, fall_stub);
            edges.push((taken_rel, taken_stub, taken));
            edges.push((fall_rel, fall_stub, fallthrough));
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
            // touched (mov/movabs/push only). Once the push has happened the
            // call is complete, so the reload and the branch resume at the
            // callee.
            meta.taken = target;
            let at = out.len();
            mov_gs_rax(&mut out, RAX_SLOT); // mov gs:[rax_slot], rax
            mark(entries, &out, at, recipe::PRECISE_T);
            let at = out.len();
            movabs_rax(&mut out, ret); // movabs rax, ret
            out.push(0x50); // push rax
            mark(entries, &out, at, recipe::RAXSLOT_T);
            let at = out.len();
            gs_load(&mut out, MODRM_RAX, RAX_SLOT); // mov rax, gs:[rax_slot]
            mark(entries, &out, at, recipe::RAXSLOT_TAKEN);
            let at = out.len();
            pad_rel32_alignment(&mut out, term_pc, 1);
            out.push(0xE9);
            let rel = take_rel32(&mut out);
            mark(entries, &out, at, recipe::PRECISE_TAKEN);
            let stub = out.len();
            emit_stub(&mut out, target, exit_tramp);
            mark(entries, &out, stub, recipe::FLOW);
            write_rel32(&mut out, rel, stub);
            edges.push((rel, stub, target));
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
///
/// Every instruction gets a [`recipe::PRO`] entry naming which of rax, rdx,
/// and the status flags are in their slots at the boundary before it, so a
/// thread preempted mid-prologue recovers the block's entry state. The
/// installs themselves need no recipe: an `xrstor64` already run leaves the
/// registers equal to the still-canonical `fpstate`, and a `wrfsbase` already
/// run is detected from the FS base itself (see `preempt`).
fn build_prologue(needs: BlockNeeds, flags_live_in: bool, entries: &mut Vec<Entry>) -> Vec<u8> {
    let d_fp = offset_of!(ThreadState, fp_in_regs) as i32;
    let d_fps = offset_of!(ThreadState, fpstate) as i32;
    let d_flags = offset_of!(ThreadState, fp_flags) as i32;
    let d_scr = offset_of!(ThreadState, fp_scratch) as i32;
    let d_fs = offset_of!(ThreadState, fs_is_guest) as i32;
    let d_guest_fs = offset_of!(ThreadState, guest_fs_base) as i32;
    const P: u8 = recipe::PRO;
    const R: u8 = recipe::PRO | recipe::PRO_RAX;
    const RF: u8 = recipe::PRO | recipe::PRO_RAX | recipe::PRO_FLAGS;
    let mut p = Prologue {
        out: Vec::new(),
        entries,
    };

    if flags_live_in {
        // Park rax and the guest flags up front; the guards' cmp may then
        // clobber flags, and the installs may clobber rax, freely. The flags
        // slot is valid from its store on; until then the live flags are still
        // the guest's (lahf/seto read them without writing).
        p.ins(P, |o| gs_store(o, MODRM_RAX, RAX_SLOT)); // mov gs:[rax], rax
        p.ins(R, |o| o.push(0x9f)); // lahf
        p.ins(R, |o| o.extend_from_slice(&[0x0f, 0x90, 0xc0])); // seto al
        p.ins(R, |o| gs_store(o, MODRM_RAX, d_flags)); // mov gs:[fp_flags], rax
        if needs.fp {
            p.guarded(d_fp, RF, |p| p.fp_restore(RF, d_fps, d_scr, d_fp));
        }
        if needs.fs {
            p.guarded(d_fs, RF, |p| p.fs_install(RF, d_guest_fs, d_fs));
        }
        // rax<-flags; add al,0x7f; sahf — restore the guest status flags.
        p.ins(RF, |o| gs_load(o, MODRM_RAX, d_flags));
        p.ins(RF, |o| o.extend_from_slice(&[0x04, 0x7f]));
        p.ins(RF, |o| o.push(0x9e));
        p.ins(R, |o| gs_load(o, MODRM_RAX, RAX_SLOT)); // mov rax, gs:[rax]
    } else {
        // Flags are dead, so each guard's cmp clobbers them harmlessly and rax
        // is saved only in the install path that actually runs.
        if needs.fp {
            p.guarded(d_fp, P, |p| {
                p.ins(P, |o| gs_store(o, MODRM_RAX, RAX_SLOT)); // mov gs:[rax], rax
                p.fp_restore(R, d_fps, d_scr, d_fp);
                p.ins(R, |o| gs_load(o, MODRM_RAX, RAX_SLOT)); // mov rax, gs:[rax]
            });
        }
        if needs.fs {
            p.guarded(d_fs, P, |p| {
                p.ins(P, |o| gs_store(o, MODRM_RAX, RAX_SLOT)); // mov gs:[rax], rax
                p.fs_install(R, d_guest_fs, d_fs);
                p.ins(R, |o| gs_load(o, MODRM_RAX, RAX_SLOT)); // mov rax, gs:[rax]
            });
        }
    }
    p.out
}

/// The prologue under construction: its bytes and the preemption entries that
/// describe them, one per emitted instruction.
struct Prologue<'a> {
    out: Vec<u8>,
    entries: &'a mut Vec<Entry>,
}

impl Prologue<'_> {
    /// Emit one instruction whose boundary-before state is `code`.
    fn ins(&mut self, code: u8, emit: impl FnOnce(&mut Vec<u8>)) {
        let at = self.out.len();
        emit(&mut self.out);
        mark(self.entries, &self.out, at, code);
    }

    /// Emit `cmp byte gs:[flag], 0; jne skip; <body>; skip:` — the install runs
    /// only when the flag is clear. The `jne` is a short `rel8`; an install is
    /// well under 128 bytes. `code` describes the boundaries around the guard
    /// itself (the `cmp` has clobbered the flags by the `jne`, so a prologue
    /// that parked them passes the flags-in-slot code here).
    fn guarded(&mut self, flag_disp: i32, code: u8, body: impl FnOnce(&mut Self)) {
        self.ins(code, |o| cmp_gs_byte_zero(o, flag_disp));
        let jne = self.out.len() + 1;
        self.ins(code, |o| o.extend_from_slice(&[0x75, 0])); // jne skip
        body(self);
        let skip = self.out.len();
        self.out[jne] = jcc_rel8(jne, skip);
    }

    /// Emit the FP restore proper: park rdx, restore the full extended state
    /// from `gs:[fpstate]` via `xrstor64` (mask `0xe7` in edx:eax), reload rdx,
    /// and set `fp_in_regs`. Assumes rax is already saved (it is loaded with
    /// the mask's low half) and that the caller reloads rax after; `base` is
    /// the caller's boundary code, to which rdx-in-slot is added across the
    /// two instructions that run with rdx clobbered.
    fn fp_restore(&mut self, base: u8, d_fps: i32, d_scr: i32, d_in: i32) {
        let rdx = base | recipe::PRO_RDX;
        self.ins(base, |o| gs_store(o, MODRM_RDX, d_scr)); // mov gs:[fp_scratch], rdx
        self.ins(base, |o| {
            o.extend_from_slice(&[0xb8, 0xe7, 0x00, 0x00, 0x00])
        }); // mov eax, 0xe7
        self.ins(base, |o| o.extend_from_slice(&[0x31, 0xd2])); // xor edx, edx
        self.ins(rdx, |o| {
            o.extend_from_slice(&[0x65, 0x48, 0x0f, 0xae, 0x2c, 0x25]); // xrstor64 gs:[
            emit_u32(o, d_fps as u32); //   fpstate]
        });
        self.ins(rdx, |o| gs_load(o, MODRM_RDX, d_scr)); // mov rdx, gs:[fp_scratch]
        self.ins(base, |o| mov_gs_byte_one(o, d_in)); // mov byte gs:[fp_in_regs], 1
    }

    /// Emit the FS-base install: load the guest base from `gs:[guest_fs_base]`
    /// into rax, `wrfsbase` it, and set `fs_is_guest`. Assumes rax is already
    /// saved and reloaded by the caller.
    fn fs_install(&mut self, base: u8, d_guest_fs: i32, d_fs: i32) {
        self.ins(base, |o| gs_load(o, MODRM_RAX, d_guest_fs)); // mov rax, gs:[guest_fs_base]
        self.ins(base, |o| {
            o.extend_from_slice(&[0xf3, 0x48, 0x0f, 0xae, 0xd0])
        }); // wrfsbase rax
        self.ins(base, |o| mov_gs_byte_one(o, d_fs)); // mov byte gs:[fs_is_guest], 1
    }
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
/// `opcode_len` opcode bytes lands on a 4-byte boundary in the cache — but not
/// a 16-byte one — given the terminator will be emitted at `term_pc`.
///
/// The 4-byte alignment lets the field be back-patched with a single atomic
/// 32-bit store; that alone, though, only makes the *store* indivisible. A
/// sibling core's instruction fetch is not an atomic 4-byte load: it decodes
/// from aligned 16-byte fetch windows, and a branch whose opcode sits at the
/// end of one window with its displacement in the next can pair a stale opcode
/// fetch with a fresh displacement fetch (or half of each) — a spliced branch
/// that lands mid-instruction. Keeping the field off 16-byte boundaries puts
/// the whole instruction — opcode (1 or 2 bytes, so field offsets 4, 8, and 12
/// all work) plus displacement — inside one fetch window, so any single fetch
/// observes the branch either entirely old or entirely new.
fn pad_rel32_alignment(out: &mut Vec<u8>, term_pc: usize, opcode_len: usize) {
    loop {
        let rel = term_pc + out.len() + opcode_len;
        if rel.is_multiple_of(4) && !rel.is_multiple_of(16) {
            break;
        }
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
///
/// One recovery recipe per appended instruction is pushed onto `codes` (see
/// [`recipe`]): the first instruction of every sequence is still the precise
/// guest state at the terminator; past the rax save, rax lives in its slot;
/// and once a sequence has pushed or popped on the guest stack the guest PC
/// has moved to the successor — the pushed indirect-call target in the rbx
/// slot, the popped return address in rax. The syscall and `int3` sequences
/// are bound for the dispatcher on their own ([`recipe::FLOW`]). `meta`
/// receives the constants those recipes need.
fn emit_terminator(
    instrs: &mut Vec<Instruction>,
    t: &Instruction,
    syscall_tramp: u64,
    trap_tramp: u64,
    codes: &mut Vec<u8>,
    meta: &mut BlockMeta,
) -> Result<(), Error> {
    let next_ip = t.next_ip();
    let start = instrs.len();
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
        emit_exit_tail(instrs, syscall_tramp)?;
        codes.resize(instrs.len(), recipe::FLOW);
        return Ok(());
    }
    // `int3` (and `int1`): a software breakpoint. The instruction does not run
    // in the cache; it exits through `exit_trap`, which sets `exit_kind = TRAP`
    // so the run loop raises SIGTRAP. As a trap (not a fault), the resumed rip is
    // the instruction *after* the breakpoint, exactly where a real `int3` leaves
    // it, so the saved next guest PC is `next_ip`.
    if matches!(t.code(), Code::Int3 | Code::Int1) {
        emit_save_rax(instrs)?;
        emit_load_rax_imm(instrs, next_ip)?;
        emit_exit_tail(instrs, trap_tramp)?;
        codes.resize(instrs.len(), recipe::FLOW);
        return Ok(());
    }
    // Index (from `start`) of the first appended instruction past which the
    // guest PC has moved on from the terminator, with the recipe that applies
    // from there; `None` when the PC stays at the terminator throughout.
    let mut moved: Option<(usize, u8)> = None;
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
            // The push completes the call: the guest is at the callee.
            meta.taken = target;
            moved = Some((instrs.len() - start, recipe::RAXSLOT_TAKEN));
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
            // The push completes the call: the guest is at the target parked
            // in the rbx slot.
            moved = Some((instrs.len() - start, recipe::RAXSLOT_RIP_RBXSLOT));
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
            // The pop completes the return: the guest is at the address now in
            // rax, and — for `ret imm16`, until the `lea` below has run — its
            // stack pointer is the host's plus the release.
            moved = Some((instrs.len() - start, recipe::RAXSLOT_RIP_RAX));
            // `ret imm16` releases imm16 more bytes of stack after popping the
            // return address — callee-popped arguments (V8 builtins use this
            // form). Drop them with `lea`, which, like `ret`, leaves the
            // arithmetic flags untouched.
            if t.code() == Code::Retnq_imm16 {
                let imm = t.immediate16();
                meta.rsp_adj = imm;
                instrs.push(mkinstr(Instruction::with2(
                    Code::Lea_r64_m,
                    Register::RSP,
                    MemoryOperand::with_base_displ(Register::RSP, imm as i64),
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
    emit_jmp_ib_lookup(instrs)?;
    for i in 0..instrs.len() - start {
        let code = match moved {
            _ if i == 0 => recipe::PRECISE_T,
            Some((at, code)) if i >= at => {
                // Only the `lea` of a `ret imm16` sits between the pop and the
                // final jump with the stack release still outstanding.
                if code == recipe::RAXSLOT_RIP_RAX
                    && meta.rsp_adj != 0
                    && i + 1 < instrs.len() - start
                {
                    recipe::RAXSLOT_RIP_RAX_RSPADJ
                } else {
                    code
                }
            }
            _ => recipe::RAXSLOT_T,
        };
        codes.push(code);
    }
    debug_assert_eq!(codes.len(), instrs.len());
    Ok(())
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
    /// 4-byte boundary — but off 16-byte ones, so the whole branch instruction
    /// sits inside one aligned 16-byte fetch window — whatever address the
    /// terminator is emitted at, so the dispatcher can back-patch it with one
    /// store that is atomic for a sibling's instruction fetch too. Check every
    /// residue of `term_pc` mod 16 so the padding is exercised for every
    /// starting alignment.
    fn assert_edges_aligned(link: &LinkTerm) {
        for term_pc in 0..16usize {
            let mut entries = Vec::new();
            let mut meta = BlockMeta::default();
            let (bytes, edges) =
                build_linked_terminator(link, 0xdead_beef, term_pc, &mut entries, &mut meta);
            assert!(!edges.is_empty(), "a linked terminator must expose an edge");
            assert_entries_cover(&entries, bytes.len());
            for (off, stub, _target) in edges {
                assert!(
                    off + 4 <= bytes.len(),
                    "rel32 field at off={off} runs past the {} terminator bytes",
                    bytes.len(),
                );
                assert!(
                    stub <= bytes.len(),
                    "stub at off={stub} lies past the {} terminator bytes",
                    bytes.len(),
                );
                assert_eq!(
                    (term_pc + off) % 4,
                    0,
                    "rel32 field at term_pc={term_pc}, off={off} is not 4-byte aligned",
                );
                assert_ne!(
                    (term_pc + off) % 16,
                    0,
                    "rel32 field at term_pc={term_pc}, off={off} straddles a fetch window",
                );
            }
        }
    }

    /// The preemption entries of a terminator must tile its bytes exactly:
    /// every host offset inside it resolves to one recipe.
    fn assert_entries_cover(entries: &[Entry], len: usize) {
        let total: usize = entries.iter().map(|e| e.host_len as usize).sum();
        assert_eq!(
            total, len,
            "entries cover {total} of {len} terminator bytes"
        );
        assert!(entries.iter().all(|e| e.host_len > 0));
        // Round-trips through the compact encoding.
        let mut bytes = Vec::new();
        for e in entries {
            e.encode(&mut bytes);
        }
        let mut rest = &bytes[..];
        for e in entries {
            let (d, r) = Entry::decode(rest).expect("decode");
            assert_eq!(&d, e);
            rest = r;
        }
        assert!(rest.is_empty());
    }

    #[test]
    fn uncond_edge_is_aligned() {
        assert_edges_aligned(&LinkTerm::Uncond { target: 0x1000 });
    }

    /// A `loop`-family terminator's entries resume at the right successor once
    /// the native instruction has run: the short `jmp` it skips (fall-through)
    /// and the taken branch past it, never back at the terminator, where the
    /// decrement would be repeated.
    #[test]
    fn shortcond_entries_follow_the_native_instruction() {
        let link = LinkTerm::ShortCond {
            prefix67: false,
            opcode: 0xE2, // loop
            taken: 0x1000,
            fallthrough: 0x2000,
        };
        let mut entries = Vec::new();
        let mut meta = BlockMeta::default();
        let (bytes, _) = build_linked_terminator(&link, 0xdead_beef, 0, &mut entries, &mut meta);
        assert_entries_cover(&entries, bytes.len());
        assert_eq!(meta.taken, 0x1000);
        assert_eq!(meta.fall, 0x2000);
        assert_eq!(entries[0].code, recipe::PRECISE_T); // the `loop` itself
        assert_eq!(entries[0].host_len, 2);
        assert_eq!(entries[1].code, recipe::PRECISE_FALL); // `jmp short` fall path
        assert_eq!(entries[2].code, recipe::PRECISE_TAKEN); // padded `jmp rel32` taken
        // The walker lands each offset on its span.
        let mut enc = Vec::new();
        for e in &entries {
            e.encode(&mut enc);
        }
        let (e, pc, at) = walk_entries(&enc, 2, 0x500).unwrap();
        assert_eq!((e.code, pc, at), (recipe::PRECISE_FALL, 0x500, true));
        let (e, _, _) = walk_entries(&enc, bytes.len() - 1, 0x500).unwrap();
        assert_eq!(e.code, recipe::FLOW);
        assert!(walk_entries(&enc, bytes.len(), 0x500).is_none());
    }

    /// A direct call's entries: precise at the rax save, rax-in-slot up to the
    /// push, and at the callee once the return address is on the stack.
    #[test]
    fn direct_call_entries_cross_the_push() {
        let link = LinkTerm::DirectCall {
            target: 0x1000,
            ret: 0x1005,
        };
        let mut entries = Vec::new();
        let mut meta = BlockMeta::default();
        let (bytes, _) = build_linked_terminator(&link, 0xdead_beef, 0, &mut entries, &mut meta);
        assert_entries_cover(&entries, bytes.len());
        assert_eq!(meta.taken, 0x1000);
        let codes: Vec<u8> = entries.iter().map(|e| e.code).collect();
        assert_eq!(
            codes,
            [
                recipe::PRECISE_T,
                recipe::RAXSLOT_T,
                recipe::RAXSLOT_TAKEN,
                recipe::PRECISE_TAKEN,
                recipe::FLOW
            ]
        );
        assert_eq!(entries[0].host_len, 9); // mov gs:[rax], rax
        assert_eq!(entries[1].host_len, 11); // movabs + push
        assert_eq!(entries[2].host_len, 9); // mov rax, gs:[rax]
    }

    /// Body entries advance the guest PC by each instruction's guest length
    /// even where the host encoding differs (an `lea rip` rewritten to a
    /// 10-byte `movabs`), and a far-RIP-relative rewrite parks its scratch
    /// register only across the access and the reload.
    #[test]
    fn body_entries_track_guest_pc_and_scratch() {
        let mut cache = CodeCache::new(1 << 16).unwrap();
        let host_pc = cache.next_pc();
        // lea rax, [rip+0x10]; add qword [rip+0x1e00], 0x1e (far from the cache); nop
        let mut bytes = vec![0x48, 0x8d, 0x05, 0x10, 0x00, 0x00, 0x00];
        bytes.extend_from_slice(&[0x48, 0x83, 0x05, 0x00, 0x1e, 0x00, 0x00, 0x1e]);
        bytes.push(0x90);
        let guest_pc = 0x35f5_0000_0200u64;
        let mut dec = Decoder::with_ip(64, &bytes, guest_pc, DecoderOptions::NONE);
        let mut instrs: Vec<Instruction> = Vec::new();
        while dec.can_decode() {
            instrs.push(dec.decode());
        }
        assert_eq!(instrs.len(), 3);
        rewrite_rip_relative_leas(&mut instrs).unwrap();
        let codes = vec![recipe::BODY; 3];
        let mut entries = Vec::new();
        emit_body(&mut cache, &instrs, host_pc, guest_pc, &codes, &mut entries).unwrap();
        // lea→movabs (1 entry), the far add (4 entries), nop (1 entry).
        assert_eq!(entries.len(), 6);
        assert_eq!((entries[0].host_len, entries[0].guest_len), (10, 7));
        assert_eq!(entries[1].fix, 0);
        assert_eq!(entries[2].fix, 0);
        assert_ne!(entries[3].fix, 0);
        assert_eq!(entries[3].guest_len, 8);
        assert_eq!(entries[3].fix, entries[4].fix);
        assert_eq!(entries[4].guest_len, 0);
        assert_eq!(
            (entries[5].host_len, entries[5].guest_len, entries[5].fix),
            (1, 1, 0)
        );
        let mut enc = Vec::new();
        for e in &entries {
            e.encode(&mut enc);
        }
        // The nop's boundary is 7 + 8 guest bytes in, whatever the host layout.
        let nop_off: usize = entries[..5].iter().map(|e| e.host_len as usize).sum();
        let (e, pc, at) = walk_entries(&enc, nop_off, guest_pc).unwrap();
        assert_eq!((e.code, pc, at), (recipe::BODY, guest_pc + 15, true));
        // Before the reload, the access has run: the PC has advanced.
        let reload_off: usize = entries[..4].iter().map(|e| e.host_len as usize).sum();
        let (e, pc, _) = walk_entries(&enc, reload_off, guest_pc).unwrap();
        assert_eq!((e.fix, pc), (entries[4].fix, guest_pc + 15));
        // Before the access itself, it has not.
        let access_off: usize = entries[..3].iter().map(|e| e.host_len as usize).sum();
        let (_, pc, _) = walk_entries(&enc, access_off, guest_pc).unwrap();
        assert_eq!(pc, guest_pc + 7);
    }

    /// A prologue's entries tile its bytes and name the parked registers: the
    /// flags-live form parks rax and the flags across everything after the
    /// first store, and rdx only across the `xrstor64` and its reload.
    #[test]
    fn prologue_entries_name_parked_registers() {
        let mut entries = Vec::new();
        let bytes = build_prologue(BlockNeeds { fp: true, fs: true }, true, &mut entries);
        let total: usize = entries.iter().map(|e| e.host_len as usize).sum();
        assert_eq!(total, bytes.len());
        assert!(entries.iter().all(|e| e.code & recipe::PRO != 0));
        assert_eq!(entries[0].code, recipe::PRO);
        let rdx: Vec<&Entry> = entries
            .iter()
            .filter(|e| e.code & recipe::PRO_RDX != 0)
            .collect();
        assert_eq!(rdx.len(), 1); // xrstor64 and the reload, merged
        assert_eq!(rdx[0].host_len, 10 + 9);
        assert!(entries[1..].iter().all(|e| e.code & recipe::PRO_RAX != 0));

        let mut entries = Vec::new();
        let bytes = build_prologue(
            BlockNeeds {
                fp: true,
                fs: false,
            },
            false,
            &mut entries,
        );
        let total: usize = entries.iter().map(|e| e.host_len as usize).sum();
        assert_eq!(total, bytes.len());
        assert!(entries.iter().all(|e| e.code & recipe::PRO_FLAGS == 0));
        assert_eq!(entries[0].code, recipe::PRO); // cmp, jne, mov gs:[rax],rax
        assert_eq!(entries[0].host_len, 9 + 2 + 9);
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

#[cfg(test)]
mod riprel_tests {
    use super::*;

    fn decode_at(bytes: &[u8], ip: u64) -> Instruction {
        Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE).decode()
    }

    /// A RIP-relative operand within `rel32` reach of the cache is left to the
    /// encoder's ordinary fixup: no rewrite, no inserted instructions.
    #[test]
    fn near_rip_operand_is_untouched() {
        // addq $0x1e, 0x1e00(%rip)
        let instr = decode_at(
            &[0x48, 0x83, 0x05, 0x00, 0x1e, 0x00, 0x00, 0x1e],
            0x7f0000001000,
        );
        assert!(instr.is_ip_rel_memory_operand());
        assert!(
            rewrite_far_rip_operands(&[instr], 0x7f0012340000)
                .unwrap()
                .is_none()
        );
    }

    /// A far RIP-relative operand becomes the borrow sequence — save scratch,
    /// materialize the absolute target, run the access `[scratch]`-based,
    /// reload scratch — and the result must actually encode from the cache
    /// address the original operand could not reach.
    #[test]
    fn far_rip_operand_is_rewritten_and_encodes() {
        // addq $0x1e, 0x1e00(%rip) linked tens of terabytes from the cache.
        let instr = decode_at(
            &[0x48, 0x83, 0x05, 0x00, 0x1e, 0x00, 0x00, 0x1e],
            0x35f50000200,
        );
        let target = instr.ip_rel_memory_address();
        let host_pc = 0x7f0012340000u64;
        let (out, parts) = rewrite_far_rip_operands(&[instr], host_pc)
            .unwrap()
            .unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(
            parts.iter().map(|p| p.op).collect::<Vec<_>>(),
            [false, false, true, false]
        );
        assert_eq!(parts[0].fix, 0);
        assert_eq!(parts[1].fix, 0);
        assert_ne!(parts[2].fix, 0);
        assert_eq!(parts[2].fix, parts[3].fix);
        assert_eq!(out[1].code(), Code::Mov_r64_imm64);
        assert_eq!(out[1].immediate64(), target);
        let scratch = out[1].op0_register();
        assert_eq!(out[2].code(), Code::Add_rm64_imm8);
        assert_eq!(out[2].memory_base(), scratch);
        assert!(!out[2].is_ip_rel_memory_operand());
        BlockEncoder::encode(
            64,
            InstructionBlock::new(&out, host_pc),
            BlockEncoderOptions::NONE,
        )
        .expect("rewritten block must encode");
    }

    /// The borrowed register must not collide with any register the
    /// instruction touches, including implicit uses: `mul qword [rip+d]`
    /// names no register operand yet consumes rax and rdx.
    #[test]
    fn scratch_avoids_implicit_registers() {
        // mulq 0x1e00(%rip)
        let instr = decode_at(&[0x48, 0xf7, 0x25, 0x00, 0x1e, 0x00, 0x00], 0x35f50000200);
        let (out, _) = rewrite_far_rip_operands(&[instr], 0x7f0012340000)
            .unwrap()
            .unwrap();
        let scratch = out[1].op0_register();
        assert!(!matches!(scratch, Register::RAX | Register::RDX));
    }
}
