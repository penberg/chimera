//! The translated-block cache for one guest address space: the bump-allocated
//! host code buffer ([`CodeCache`]), the guest-PC → host-PC map, and the
//! direct-branch link bookkeeping that lets translated blocks jump straight
//! into one another.
//!
//! [`BlockCache::resolve`] is the dispatcher's entry point: it returns the host
//! PC for a guest block, translating it on first sight and linking it into the
//! branch graph. Linking keeps a chain of blocks running back-to-back inside a
//! single `dispatch()` call instead of round-tripping through the run loop at
//! every basic-block boundary.

use std::{
    collections::{BTreeMap, HashMap},
    sync::atomic::{AtomicI32, Ordering},
};

use crate::Error;

use super::translate::{CodeCache, OutEdge, Translation, translate};

/// Guest page size (x86-64). Self-modifying-code invalidation works at this
/// granularity: a write anywhere in a page drops every block whose guest code
/// touches it.
const PAGE: u64 = 4096;

/// One translated block's bookkeeping: where its host code begins and the
/// address of its deopt stub (used to neutralize the block on invalidation).
#[derive(Clone, Copy)]
struct Block {
    host: u64,
    deopt: u64,
}

/// The translated-block cache: the host code buffer, the guest-PC → block map,
/// the page → blocks index that drives SMC invalidation, and the pending
/// direct-branch links awaiting their successors.
pub struct BlockCache {
    cache: CodeCache,
    /// Guest block PC -> the block translated there.
    map: HashMap<u64, Block>,
    /// Guest page base -> start PCs of the blocks whose code touches that page,
    /// so a write to the page can find and drop them. A block spanning two pages
    /// is listed under both; stale entries (a block already dropped) are skipped
    /// on lookup, so the index is only ever cleared, never eagerly pruned. A
    /// `BTreeMap` so [`invalidate_range`] can drop a span of pages without
    /// walking the whole index — a guest `munmap`/`mprotect` may cover gigabytes.
    page_blocks: BTreeMap<u64, Vec<u64>>,
    /// Direct branches awaiting their successor: guest target PC -> addresses
    /// of the `rel32` displacement fields to rewrite once that block exists.
    /// An edge whose target is already translated is patched on the spot and
    /// never lands here. Drained in [`BlockCache::resolve`].
    pending: HashMap<u64, Vec<usize>>,
}

impl BlockCache {
    pub fn new(cache_size: usize) -> Result<Self, Error> {
        Ok(Self {
            cache: CodeCache::new(cache_size)?,
            map: HashMap::new(),
            page_blocks: BTreeMap::new(),
            pending: HashMap::new(),
        })
    }

    /// Return the host PC for a guest block, translating it on first sight.
    /// On a fresh translation, also returns the block's guest span `(start,
    /// end)` so the caller can arm the page(s) it covers for SMC; a cache hit
    /// returns `None` for the span.
    ///
    /// On a translation, the new block is linked into the cache's branch graph:
    /// any direct branches that were waiting for this guest PC are patched to
    /// jump straight here, and the new block's own outgoing direct branches are
    /// either patched to their (already-translated) successors or recorded as
    /// pending. After warm-up, chains of linked blocks execute back-to-back
    /// without returning to the dispatcher.
    pub fn resolve(
        &mut self,
        guest_pc: u64,
        block_exit: u64,
        syscall_exit: u64,
    ) -> Result<(u64, Option<(u64, u64)>), Error> {
        if let Some(&block) = self.map.get(&guest_pc) {
            // Refresh the in-cache lookup entry: reaching here means an indirect
            // branch missed it (a direct-mapped collision evicted it), so
            // reinstate it to return the hot target to the fast path.
            self.cache.ib_insert(guest_pc, block.host);
            return Ok((block.host, None));
        }
        let Translation {
            host_pc,
            guest_end,
            deopt_pc,
            edges,
        } = translate(&mut self.cache, guest_pc, block_exit, syscall_exit)?;
        self.map.insert(
            guest_pc,
            Block {
                host: host_pc,
                deopt: deopt_pc,
            },
        );
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
        // Mirror the mapping into the in-cache lookup table so indirect
        // branches to this block resolve without leaving the cache.
        self.cache.ib_insert(guest_pc, host_pc);
        if let Some(sites) = self.pending.remove(&guest_pc) {
            for site in sites {
                patch_site(site, host_pc);
            }
        }
        for OutEdge { target_guest, site } in edges {
            match self.map.get(&target_guest) {
                Some(&target) => patch_site(site, target.host),
                None => self.pending.entry(target_guest).or_default().push(site),
            }
        }
        Ok((host_pc, Some((guest_pc, guest_end))))
    }

    /// Drop every block whose code touches guest `page`, for self-modifying
    /// code: each is removed from the map, evicted from the indirect-branch
    /// table, and its host entry redirected to its deopt stub so any surviving
    /// direct link into it re-translates. Returns whether anything was dropped.
    ///
    /// Allocation-free: it runs inside the synchronous fault handler, so it must
    /// not call the allocator. The page's block list is cleared in place
    /// (retaining its buffer) rather than removed, and `HashMap::remove` neither
    /// shrinks nor frees.
    pub fn invalidate_page(&mut self, page: u64) -> bool {
        let mut dropped = false;
        if let Some(starts) = self.page_blocks.get_mut(&page) {
            for &guest_pc in starts.iter() {
                if let Some(block) = self.map.remove(&guest_pc) {
                    self.cache.ib_remove(guest_pc);
                    self.cache.neutralize(block.host, block.deopt);
                    dropped = true;
                }
            }
            starts.clear();
        }
        dropped
    }

    /// Drop every block whose code touches any page in `[lo_page, hi_page]`
    /// (both page-aligned). For a guest `munmap`/`mprotect` of stale or
    /// re-protected code; runs off the fault path, so it may allocate. Only the
    /// pages actually carrying translations are visited, so a multi-gigabyte
    /// range is cheap when little of it was ever executed.
    pub fn invalidate_range(&mut self, lo_page: u64, hi_page: u64) {
        for (_, starts) in self.page_blocks.range_mut(lo_page..=hi_page) {
            for &guest_pc in starts.iter() {
                if let Some(block) = self.map.remove(&guest_pc) {
                    self.cache.ib_remove(guest_pc);
                    self.cache.neutralize(block.host, block.deopt);
                }
            }
            starts.clear();
        }
    }

    /// Emit (once per cache) the shared inline indirect-branch lookup routine
    /// and return its host address so translated indirect branches can reach it.
    pub fn ensure_ib_lookup(&mut self, block_exit: u64) -> Result<u64, Error> {
        self.cache.ensure_ib_lookup(block_exit)
    }

    /// Flush every translated block and its link bookkeeping. The backing code
    /// buffer is rewound; the guest-PC map, page index, and pending links are
    /// cleared.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.map.clear();
        self.page_blocks.clear();
        self.pending.clear();
    }
}

/// Rewrite the `rel32` displacement at `site` (an address inside the RWX code
/// cache) so its branch lands at `host_pc`. The displacement is measured from
/// the end of its own four bytes. The cache stays well under 2 GiB, so the
/// signed distance always fits in an `i32`.
///
/// The translator aligns every patchable `rel32` field to a 4-byte boundary
/// (see `build_linked_terminator`), so this is a single aligned atomic store:
/// a sibling thread executing the branch reads either the old target (its cold
/// exit stub, which round-trips through the dispatcher) or the new one, never a
/// torn displacement spliced from both. x86 keeps instruction and data caches
/// coherent, so the executing core picks up the new target on its next fetch.
fn patch_site(site: usize, host_pc: u64) {
    let disp = host_pc as i64 - (site as i64 + 4);
    debug_assert!(
        i32::try_from(disp).is_ok(),
        "link displacement {disp} out of rel32 range"
    );
    debug_assert!(
        site.is_multiple_of(4),
        "link site {site:#x} is not 4-byte aligned"
    );
    let slot = unsafe { &*(site as *const AtomicI32) };
    slot.store(disp as i32, Ordering::Release);
}
