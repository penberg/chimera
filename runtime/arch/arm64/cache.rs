//! The translated-block cache for one guest address space on arm64: a
//! `MAP_JIT`-backed host code buffer ([`CodeCache`]), a guest-PC → host-PC map,
//! and the page → blocks index that drives self-modifying-code invalidation.
//!
//! This is the dispatcher-only cache: every translated block ends by returning
//! the next guest PC to the run loop, so no block is ever linked directly into
//! another and the map needs only a host PC per block — no deopt stub and no
//! inline indirect-branch table. The direct-branch linking and inline IB probe
//! the x86 backend carries are deferred to a later Darwin-port phase (see the
//! sequencing note in `DARWIN.md`), where the safepoint poll they require is
//! added alongside them.

use std::{
    collections::{BTreeMap, HashMap},
    ptr,
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

use crate::Error;

use super::translate::{OutEdge, Translation, translate};

/// Bounds of the `MAP_JIT` code buffer, published for the fault handler.
static CODE_CACHE_LO: AtomicUsize = AtomicUsize::new(0);
static CODE_CACHE_HI: AtomicUsize = AtomicUsize::new(0);

/// Whether `addr` lies inside the translated-code cache. The fault handler uses
/// this to distinguish a guest fault — the interrupted context faulted while
/// executing translated code — from a genuine fault taken in Chimera's own
/// Rust; it also means the faulting thread is not holding the address-space
/// lock (only ever held off the code cache), so the handler can take it
/// without self-deadlock.
pub fn code_cache_contains(addr: usize) -> bool {
    let lo = CODE_CACHE_LO.load(Ordering::Relaxed);
    let hi = CODE_CACHE_HI.load(Ordering::Relaxed);
    lo != 0 && addr >= lo && addr < hi
}

/// Guest page size on Apple Silicon (16 KiB). SMC invalidation works at this
/// granularity: a write anywhere in a page drops every block whose guest code
/// touches it.
const PAGE: u64 = 16 * 1024;

/// A bump-allocated `MAP_JIT` region into which `translate()` emits blocks. On
/// Apple Silicon a JIT mapping is writable or executable per thread, never both
/// at once, so writes are bracketed by `pthread_jit_write_protect_np` toggles;
/// and because arm64 does not keep instruction and data caches coherent, freshly
/// written code is invalidated in the I-cache before it can be executed.
pub struct CodeCache {
    base: *mut u8,
    size: usize,
    used: usize,
    /// Bump-allocated table of direct-branch link slots, in ordinary
    /// read-write memory rather than the JIT region. Each slot holds the
    /// linked entry of one edge's successor, or zero while unlinked. Keeping
    /// them out of the code means linking never rewrites an instruction: no
    /// JIT write-protect toggle, no I-cache maintenance, and a concurrent
    /// reader in translated code sees either the old value or the new one.
    links: *mut u64,
    link_slots: usize,
    links_used: usize,
    /// The indirect-branch table (see [`IB_BITS`]), also outside the JIT
    /// region: translated code only reads it, and the runtime rewrites
    /// entries with plain aligned stores.
    ib_table: *mut u64,
}

/// One link slot per 64 bytes of code cache: the shortest linkable exit stub
/// is well over that, so the table never runs dry before the code buffer
/// does. (An exhausted table is handled anyway — the edge stays unlinked.)
const CODE_BYTES_PER_LINK_SLOT: usize = 64;

/// Direct-mapped indirect-branch table: `1 << IB_BITS` entries of
/// `(guest_pc, linked_host_pc)`, probed inline by translated indirect
/// branches (see `translate::emit_ib_probe`).
pub const IB_BITS: u32 = 14;
pub const IB_ENTRIES: usize = 1 << IB_BITS;
pub const IB_ENTRY_BYTES: usize = 16;

/// Fibonacci hashing multiplier (2^64 / golden ratio, odd). Mixing the whole
/// target through the high bits keeps neighbouring branch targets — which
/// share their low bits — off the same slot.
pub const IB_HASH_MULT: u64 = 0x9E37_79B9_7F4A_7C15;

/// The slot a guest PC maps to; the inline probe computes exactly this.
fn ib_slot(guest_pc: u64) -> usize {
    (guest_pc.wrapping_mul(IB_HASH_MULT) >> (64 - IB_BITS)) as usize
}

/// A slot is empty exactly when its host PC is null, which the inline probe
/// rejects before branching. That is what makes the key field safe to leave
/// stale: no reserved key value would do, since a guest can branch to any
/// 64-bit address.
const IB_EMPTY_HOST: u64 = 0;

impl CodeCache {
    pub fn new(cache_size: usize) -> Result<Self, Error> {
        let base = unsafe {
            libc::mmap(
                ptr::null_mut(),
                cache_size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_JIT,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(Error::last_os_error("code cache mmap (MAP_JIT)"));
        }
        let link_slots = cache_size / CODE_BYTES_PER_LINK_SLOT;
        let links = unsafe {
            libc::mmap(
                ptr::null_mut(),
                link_slots * 8,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if links == libc::MAP_FAILED {
            unsafe { libc::munmap(base, cache_size) };
            return Err(Error::last_os_error("link table mmap"));
        }
        let ib_table = unsafe {
            libc::mmap(
                ptr::null_mut(),
                IB_ENTRIES * IB_ENTRY_BYTES,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ib_table == libc::MAP_FAILED {
            unsafe {
                libc::munmap(base, cache_size);
                libc::munmap(links, link_slots * 8);
            }
            return Err(Error::last_os_error("indirect-branch table mmap"));
        }
        CODE_CACHE_LO.store(base as usize, Ordering::Relaxed);
        CODE_CACHE_HI.store(base as usize + cache_size, Ordering::Relaxed);
        let cache = Self {
            base: base as *mut u8,
            size: cache_size,
            used: 0,
            links: links as *mut u64,
            link_slots,
            links_used: 0,
            ib_table: ib_table as *mut u64,
        };
        cache.clear_ib_table();
        Ok(cache)
    }

    /// Base address of the indirect-branch table, baked into the inline probe.
    pub fn ib_table_addr(&self) -> u64 {
        self.ib_table as u64
    }

    /// Publish `guest_pc -> host_pc` in the table. Direct-mapped, so this
    /// evicts any prior occupant of the slot; the evicted target just misses
    /// and re-resolves through the dispatcher next time.
    ///
    /// The pair is written with a single aligned `stp`, which Apple Silicon
    /// (FEAT_LSE2 — the only hardware this backend targets) makes
    /// single-copy atomic, matching the probe's `ldp`. A reader therefore
    /// never pairs one generation's key with another's host PC.
    pub fn ib_insert(&self, guest_pc: u64, host_pc: u64) {
        let entry = unsafe { self.ib_table.add(ib_slot(guest_pc) * 2) };
        unsafe { ib_store(entry, guest_pc, host_pc) };
    }

    /// Drop `guest_pc` from the table if it currently owns its slot, so an
    /// indirect branch there misses and re-resolves through the dispatcher.
    /// Emptying is a lone store of the null host PC: the probe rejects it
    /// whether or not the reader also matched the (now stale) key.
    pub fn ib_remove(&self, guest_pc: u64) {
        let entry = unsafe { self.ib_table.add(ib_slot(guest_pc) * 2) };
        if unsafe { (*(entry as *const AtomicU64)).load(Ordering::Acquire) } == guest_pc {
            unsafe {
                (*(entry.add(1) as *const AtomicU64)).store(IB_EMPTY_HOST, Ordering::Release)
            };
        }
    }

    /// Empty every slot.
    fn clear_ib_table(&self) {
        unsafe { ptr::write_bytes(self.ib_table as *mut u8, 0, IB_ENTRIES * IB_ENTRY_BYTES) };
    }

    /// Reserve one link slot, zeroed (i.e. unlinked). `None` once the table is
    /// exhausted, which leaves the edge permanently unlinked.
    pub fn alloc_link_slot(&mut self) -> Option<u64> {
        if self.links_used >= self.link_slots {
            return None;
        }
        let slot = unsafe { self.links.add(self.links_used) };
        self.links_used += 1;
        unsafe { (*(slot as *const AtomicU64)).store(0, Ordering::Release) };
        Some(slot as u64)
    }

    /// Host PC of the next byte `emit` will write — the address the block a
    /// following `emit` produces will begin at.
    pub fn next_pc(&self) -> u64 {
        (self.base as u64) + self.used as u64
    }

    /// Append a block's instruction words to the buffer and return the host PC
    /// they were written at. Toggles the JIT mapping writable across the store,
    /// then flushes the affected range from the I-cache so the core fetches the
    /// new code rather than stale bytes.
    pub fn emit(&mut self, words: &[u32]) -> Result<u64, Error> {
        let bytes = words.len() * 4;
        if self.used + bytes > self.size {
            return Err(Error::CodeCacheExhausted);
        }
        let host_pc = self.next_pc();
        unsafe {
            jit_write_protect(false);
            let dst = self.base.add(self.used) as *mut u32;
            for (i, w) in words.iter().enumerate() {
                ptr::write_unaligned(dst.add(i), *w);
            }
            jit_write_protect(true);
            sys_icache_invalidate(host_pc as *mut libc::c_void, bytes);
        }
        self.used += bytes;
        Ok(host_pc)
    }

    /// Rewind the buffer to empty, discarding every block, link, and cached
    /// indirect-branch target. The mappings are kept.
    pub fn reset(&mut self) {
        self.used = 0;
        self.links_used = 0;
        self.clear_ib_table();
    }
}

impl Drop for CodeCache {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base as *mut libc::c_void, self.size);
            libc::munmap(self.links as *mut libc::c_void, self.link_slots * 8);
            libc::munmap(
                self.ib_table as *mut libc::c_void,
                IB_ENTRIES * IB_ENTRY_BYTES,
            );
        }
    }
}

/// Store an indirect-branch entry's `(key, host)` pair with one aligned
/// `stp`. Apple Silicon implements FEAT_LSE2, which makes an aligned 16-byte
/// pair access single-copy atomic — so this pairs with the probe's `ldp` and
/// a reader can never splice a key from one generation onto a host PC from
/// another. This backend targets Apple Silicon only, so the guarantee holds
/// wherever it runs.
///
/// # Safety
/// `entry` must be a 16-byte-aligned indirect-branch table entry.
unsafe fn ib_store(entry: *mut u64, key: u64, host: u64) {
    unsafe {
        std::arch::asm!(
            "stp {k}, {h}, [{p}]",
            k = in(reg) key,
            h = in(reg) host,
            p = in(reg) entry,
            options(nostack, preserves_flags),
        );
    }
}

/// Publish a link slot's value: the successor's linked entry, or zero to
/// unlink. A single aligned atomic store, so translated code reading the slot
/// never sees a torn value — it takes either the old destination (or the cold
/// path, if zero) or the new one.
fn write_link_slot(slot: u64, value: u64) {
    unsafe { (*(slot as *const AtomicU64)).store(value, Ordering::Release) };
}

unsafe extern "C" {
    fn pthread_jit_write_protect_np(enabled: libc::c_int);
    fn sys_icache_invalidate(start: *mut libc::c_void, len: libc::size_t);
}

/// Toggle the calling thread's `MAP_JIT` mapping between writable (`false`) and
/// executable (`true`). Apple Silicon permits only one at a time per thread.
unsafe fn jit_write_protect(enabled: bool) {
    unsafe { pthread_jit_write_protect_np(if enabled { 1 } else { 0 }) };
}

impl CodeCache {
    /// Whether `[start, start+len)` overlaps the code buffer at all. A guest
    /// `mprotect` over the cache is the runtime's to service, not the host's.
    pub fn contains_range(&self, start: usize, len: usize) -> bool {
        let end = start.saturating_add(len);
        let lo = self.base as usize;
        let hi = lo + self.size;
        start < hi && end > lo
    }
}

/// The translated-block cache: the host code buffer, the guest-PC → block map,
/// the page → blocks index that drives SMC invalidation, and the direct-branch
/// link table that lets blocks jump straight into one another.
pub struct BlockCache {
    cache: CodeCache,
    /// Guest block PC → the block translated there.
    map: HashMap<u64, Block>,
    /// Guest target PC → the link slots of every edge branching there. A slot
    /// is filled when its target is translated and re-zeroed when the target
    /// is invalidated, so a stale link can never survive the block it names.
    /// Edges are never removed from this index — the emitted stub that owns
    /// the slot lives as long as the code buffer — so an edge is re-linked
    /// automatically when its target is translated again.
    links: HashMap<u64, Vec<u64>>,
    /// Guest page base → start PCs of the blocks whose code touches that page,
    /// so a write to the page can find and drop them. A block spanning two pages
    /// is listed under both; stale entries (a block already dropped) are skipped
    /// on lookup, so the index is only ever cleared, never eagerly pruned. A
    /// `BTreeMap` so [`invalidate_range`](BlockCache::invalidate_range) can drop
    /// a span of pages without walking the whole index.
    page_blocks: BTreeMap<u64, Vec<u64>>,
    /// Host block PC → its extent and host-offset → guest-PC map, so a faulting
    /// host PC maps back to the guest instruction that produced it. Entries
    /// survive guest-side invalidation deliberately: the emitted code is
    /// bump-allocated and never reused until [`reset`](BlockCache::reset), so an
    /// orphaned block a thread is still executing keeps correct attribution.
    host_blocks: BTreeMap<u64, HostBlock>,
}

struct HostBlock {
    host_end: u64,
    pc_map: Vec<(u32, u64)>,
}

/// One translated block: the dispatcher's entry point and the entry a linked
/// predecessor branches to (which pops the guest x16/x17 off the guest stack
/// before the body).
#[derive(Clone, Copy)]
struct Block {
    host: u64,
    linked: u64,
}

impl BlockCache {
    pub fn new(cache_size: usize) -> Result<Self, Error> {
        Ok(Self {
            cache: CodeCache::new(cache_size)?,
            map: HashMap::new(),
            links: HashMap::new(),
            page_blocks: BTreeMap::new(),
            host_blocks: BTreeMap::new(),
        })
    }

    /// The guest PC of the instruction whose translation contains host address
    /// `host_pc`, for fault attribution. `None` if the address is not inside a
    /// translated block.
    pub fn guest_pc_at(&self, host_pc: u64) -> Option<u64> {
        let (&start, block) = self.host_blocks.range(..=host_pc).next_back()?;
        if host_pc >= block.host_end || block.pc_map.is_empty() {
            return None;
        }
        let off = ((host_pc - start) / 4) as u32;
        // Last entry at-or-before the offset; the prologue words before the
        // first entry belong to the block's first guest instruction.
        let idx = block.pc_map.partition_point(|&(o, _)| o <= off);
        Some(block.pc_map[idx.max(1) - 1].1)
    }

    /// Return the host PC for a guest block, translating it on first sight. On a
    /// fresh translation, also returns the block's guest span `(start, end)` so
    /// the caller can arm the page(s) it covers for SMC; a cache hit returns
    /// `None` for the span.
    ///
    /// A translation is also linked into the cache's branch graph: every edge
    /// already waiting on this guest PC gets the new block's linked entry, and
    /// the new block's own outgoing edges take their targets' entries where
    /// those are already translated. After warm-up a chain of linked blocks
    /// runs back-to-back inside one `dispatch` call instead of returning to
    /// the run loop at every basic-block boundary.
    pub fn resolve(
        &mut self,
        guest_pc: u64,
        block_exit: u64,
        syscall_exit: u64,
        trap_exit: u64,
    ) -> Result<(u64, Option<(u64, u64)>), Error> {
        if let Some(&block) = self.map.get(&guest_pc) {
            // Reaching here means an indirect branch missed the table (a
            // direct-mapped collision evicted the entry); reinstate it so the
            // hot target returns to the inline probe's fast path.
            self.cache.ib_insert(guest_pc, block.linked);
            return Ok((block.host, None));
        }
        let Translation {
            host_pc,
            host_end,
            guest_end,
            linked_pc,
            pc_map,
            edges,
        } = translate(
            &mut self.cache,
            guest_pc,
            block_exit,
            syscall_exit,
            trap_exit,
        )?;
        self.map.insert(
            guest_pc,
            Block {
                host: host_pc,
                linked: linked_pc,
            },
        );
        // Attribution is keyed by the emitted extent, which begins at the
        // linked entry — a fault taken there belongs to this block too.
        self.host_blocks
            .insert(linked_pc, HostBlock { host_end, pc_map });
        // Index the block under every guest page it touches.
        let mut page = guest_pc & !(PAGE - 1);
        let last = guest_end.saturating_sub(1) & !(PAGE - 1);
        loop {
            self.page_blocks.entry(page).or_default().push(guest_pc);
            if page >= last {
                break;
            }
            page += PAGE;
        }
        // Publish for the inline indirect-branch probe, then link the edges
        // waiting for this block and this block's own.
        self.cache.ib_insert(guest_pc, linked_pc);
        if let Some(slots) = self.links.get(&guest_pc) {
            for &slot in slots {
                write_link_slot(slot, linked_pc);
            }
        }
        for OutEdge { target_guest, slot } in edges {
            if let Some(&target) = self.map.get(&target_guest) {
                write_link_slot(slot, target.linked);
            }
            self.links.entry(target_guest).or_default().push(slot);
        }
        Ok((host_pc, Some((guest_pc, guest_end))))
    }

    /// Unlink every edge pointing at `guest_pc`, so a predecessor's stub takes
    /// its cold path (back through the run loop, which re-translates) instead
    /// of jumping into a block that no longer describes the guest's code.
    /// Allocation-free — it runs inside the synchronous fault handler.
    fn unlink_target(&mut self, guest_pc: u64) {
        self.cache.ib_remove(guest_pc);
        if let Some(slots) = self.links.get(&guest_pc) {
            for &slot in slots {
                write_link_slot(slot, 0);
            }
        }
    }

    /// Drop every block whose code touches guest `page`, for self-modifying
    /// code: each is removed from the map and unlinked from its predecessors,
    /// so the next dispatch of that guest PC re-translates from the rewritten
    /// bytes. Returns whether anything was dropped. Allocation-free: it runs
    /// inside the synchronous fault handler, so the page's block list is
    /// cleared in place, `HashMap::remove` neither shrinks nor frees, and
    /// unlinking only stores to already-allocated slots.
    pub fn invalidate_page(&mut self, page: u64) -> bool {
        let mut dropped = false;
        // Take the list out to unlink through `&mut self`, then put the
        // (cleared, capacity-preserving) buffer back — no allocation either way.
        if let Some(starts) = self.page_blocks.get_mut(&page) {
            let mut starts = std::mem::take(starts);
            for &guest_pc in starts.iter() {
                if self.map.remove(&guest_pc).is_some() {
                    self.unlink_target(guest_pc);
                    dropped = true;
                }
            }
            starts.clear();
            if let Some(slot) = self.page_blocks.get_mut(&page) {
                *slot = starts;
            }
        }
        dropped
    }

    /// Drop every block whose code touches any page in `[lo_page, hi_page]`
    /// (both page-aligned). For a guest `munmap`/`mprotect` of stale or
    /// re-protected code; only the pages actually carrying translations are
    /// visited, so a multi-gigabyte range is cheap when little of it ran.
    pub fn invalidate_range(&mut self, lo_page: u64, hi_page: u64) {
        let mut dropped: Vec<u64> = Vec::new();
        for (_, starts) in self.page_blocks.range_mut(lo_page..=hi_page) {
            for &guest_pc in starts.iter() {
                if self.map.remove(&guest_pc).is_some() {
                    dropped.push(guest_pc);
                }
            }
            starts.clear();
        }
        for guest_pc in dropped {
            self.unlink_target(guest_pc);
        }
    }

    pub fn code_contains_range(&self, start: usize, len: usize) -> bool {
        self.cache.contains_range(start, len)
    }

    pub fn code_allow_writes(&self) {
        unsafe { jit_write_protect(false) };
    }

    pub fn code_deny_writes(&self) {
        unsafe { jit_write_protect(true) };
    }

    /// Flush every translated block: the code buffer and link table are
    /// rewound and the guest-PC map, page index, and link index cleared.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.map.clear();
        self.links.clear();
        self.page_blocks.clear();
        self.host_blocks.clear();
    }
}
