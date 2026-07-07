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
//!
//! Sibling threads execute out of the cache the whole time this bookkeeping
//! runs, so every mutation of reachable code is a single aligned atomic store
//! into a `rel32` field ([`patch_site`]) — a branch is re-aimed, but no
//! instruction bytes are ever spliced. Invalidation severs a dropped block's
//! incoming edges the same way, re-aiming each one back at its own cold exit
//! stub; the dropped block's body is left intact (cache memory is never reused
//! short of [`BlockCache::reset`]), so a thread already in flight through a
//! stale edge executes one intact pre-invalidation block and re-resolves at
//! its next boundary — the same staleness a native core sees when it executes
//! prefetched bytes an instant before a cross-modifying write lands.

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

/// One direct-branch link site: the address of a patchable `rel32` field and
/// the address of the cold exit stub the field targets while its edge is
/// unlinked.
struct LinkSite {
    site: usize,
    stub: u64,
}

/// The translated-block cache: the host code buffer, the guest-PC → host-PC
/// map, the page → blocks index that drives SMC invalidation, and the link
/// registry that records every direct branch by successor.
pub struct BlockCache {
    cache: CodeCache,
    /// Guest block PC -> its host entry.
    map: HashMap<u64, u64>,
    /// Guest page base -> start PCs of the blocks whose code touches that page,
    /// so a write to the page can find and drop them. A block spanning two pages
    /// is listed under both; stale entries (a block already dropped) are skipped
    /// on lookup, so the index is only ever cleared, never eagerly pruned. A
    /// `BTreeMap` so [`invalidate_range`] can drop a span of pages without
    /// walking the whole index — a guest `munmap`/`mprotect` may cover gigabytes.
    page_blocks: BTreeMap<u64, Vec<u64>>,
    /// Successor guest PC -> every direct-branch site that targets it, live for
    /// the lifetime of the cache. A site is aimed at the successor's host entry
    /// exactly while that guest PC is in `map`, and at its own cold exit stub
    /// otherwise: translation patches the list toward the new entry, and
    /// invalidation patches it back ([`invalidate_page`] runs inside the
    /// synchronous fault handler, so severing edges must not allocate — it only
    /// walks this list and stores). Sites inside blocks that have themselves
    /// been dropped linger here; re-aiming them writes into cache bytes nothing
    /// jumps to any more, which is harmless, and they are shed only at
    /// [`reset`].
    links: HashMap<u64, Vec<LinkSite>>,
}

impl BlockCache {
    pub fn new(cache_size: usize) -> Result<Self, Error> {
        Ok(Self {
            cache: CodeCache::new(cache_size)?,
            map: HashMap::new(),
            page_blocks: BTreeMap::new(),
            links: HashMap::new(),
        })
    }

    /// Return the host PC for a guest block, translating it on first sight.
    /// On a fresh translation, also returns the block's guest span `(start,
    /// end)` so the caller can arm the page(s) it covers for SMC; a cache hit
    /// returns `None` for the span.
    ///
    /// On a translation, the new block is linked into the cache's branch graph:
    /// every direct branch registered for this guest PC — waiting since its own
    /// translation, or severed by an earlier invalidation — is patched to jump
    /// straight here, and the new block's own outgoing direct branches are
    /// registered under their successors and patched to any that are already
    /// translated. After warm-up, chains of linked blocks execute back-to-back
    /// without returning to the dispatcher.
    pub fn resolve(
        &mut self,
        guest_pc: u64,
        block_exit: u64,
        syscall_exit: u64,
        trap_exit: u64,
    ) -> Result<(u64, Option<(u64, u64)>), Error> {
        if let Some(&host) = self.map.get(&guest_pc) {
            // Refresh the in-cache lookup entry: reaching here means an indirect
            // branch missed it (a direct-mapped collision evicted it), so
            // reinstate it to return the hot target to the fast path.
            self.cache.ib_insert(guest_pc, host);
            return Ok((host, None));
        }
        let Translation {
            host_pc,
            guest_end,
            edges,
        } = translate(
            &mut self.cache,
            guest_pc,
            block_exit,
            syscall_exit,
            trap_exit,
        )?;
        self.map.insert(guest_pc, host_pc);
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
        if let Some(sites) = self.links.get(&guest_pc) {
            for s in sites {
                patch_site(s.site, host_pc);
            }
        }
        for OutEdge {
            target_guest,
            site,
            stub,
        } in edges
        {
            if let Some(&target) = self.map.get(&target_guest) {
                patch_site(site, target);
            }
            self.links
                .entry(target_guest)
                .or_default()
                .push(LinkSite { site, stub });
        }
        Ok((host_pc, Some((guest_pc, guest_end))))
    }

    /// Drop every block whose code touches guest `page`, for self-modifying
    /// code: each is removed from the map, evicted from the indirect-branch
    /// table, and severed from the branch graph — every direct branch linked to
    /// it is re-aimed at its own cold exit stub, so the next traversal of the
    /// edge falls back to the dispatcher and re-translates from current guest
    /// memory. Returns whether anything was dropped.
    ///
    /// Allocation-free: it runs inside the synchronous fault handler, so it must
    /// not call the allocator. The page's block list is cleared in place
    /// (retaining its buffer) rather than removed, `HashMap::remove` neither
    /// shrinks nor frees, and unlinking only walks `links` and stores.
    pub fn invalidate_page(&mut self, page: u64) -> bool {
        let mut dropped = false;
        if let Some(starts) = self.page_blocks.get_mut(&page) {
            for &guest_pc in starts.iter() {
                if self.map.remove(&guest_pc).is_some() {
                    self.cache.ib_remove(guest_pc);
                    unlink(&self.links, guest_pc);
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
                if self.map.remove(&guest_pc).is_some() {
                    self.cache.ib_remove(guest_pc);
                    unlink(&self.links, guest_pc);
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
    /// buffer is rewound; the guest-PC map, page index, and link registry are
    /// cleared.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.map.clear();
        self.page_blocks.clear();
        self.links.clear();
    }
}

/// Sever every direct branch targeting `guest_pc` by re-aiming its `rel32`
/// back at its own cold exit stub — the reverse of the linking in
/// [`BlockCache::resolve`], and the same single atomic store.
fn unlink(links: &HashMap<u64, Vec<LinkSite>>, guest_pc: u64) {
    if let Some(sites) = links.get(&guest_pc) {
        for s in sites {
            patch_site(s.site, s.stub);
        }
    }
}

/// Rewrite the `rel32` displacement at `site` (an address inside the RWX code
/// cache) so its branch lands at `host_pc`. The displacement is measured from
/// the end of its own four bytes. The cache stays well under 2 GiB, so the
/// signed distance always fits in an `i32`.
///
/// The translator places every patchable `rel32` field on a 4-byte boundary
/// that keeps the whole branch instruction inside one aligned 16-byte fetch
/// window (see `pad_rel32_alignment`), so this single aligned atomic store is
/// indivisible for a sibling thread's instruction fetch as well: the sibling
/// executes the branch toward the old target or the new one, never a torn
/// displacement spliced from both. x86 keeps instruction and data caches
/// coherent, so the executing core picks up the new target on its next fetch.
fn patch_site(site: usize, host_pc: u64) {
    let disp = host_pc as i64 - (site as i64 + 4);
    debug_assert!(
        i32::try_from(disp).is_ok(),
        "link displacement {disp} out of rel32 range"
    );
    debug_assert!(
        site.is_multiple_of(4) && !site.is_multiple_of(16),
        "link site {site:#x} straddles a fetch window"
    );
    let slot = unsafe { &*(site as *const AtomicI32) };
    slot.store(disp as i32, Ordering::Release);
}
