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

use std::collections::HashMap;

use crate::Error;

use super::translate::{CodeCache, OutEdge, translate};

/// The translated-block cache: the host code buffer, the guest-PC → host-PC
/// map, and the pending direct-branch links awaiting their successors.
pub struct BlockCache {
    cache: CodeCache,
    /// Guest block PC -> host PC where that block's translation begins.
    map: HashMap<u64, u64>,
    /// Direct branches awaiting their successor: guest target PC -> addresses
    /// of the `rel32` displacement fields to rewrite once that block exists.
    /// An edge whose target is already translated is patched on the spot and
    /// never lands here. Drained in [`BlockCache::resolve`].
    pending: HashMap<u64, Vec<usize>>,
}

impl BlockCache {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            cache: CodeCache::new()?,
            map: HashMap::new(),
            pending: HashMap::new(),
        })
    }

    /// Return the host PC for a guest block, translating it on first sight.
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
    ) -> Result<u64, Error> {
        if let Some(&host_pc) = self.map.get(&guest_pc) {
            // Refresh the in-cache lookup entry: reaching here means an indirect
            // branch missed it (a direct-mapped collision evicted it), so
            // reinstate it to return the hot target to the fast path.
            self.cache.ib_insert(guest_pc, host_pc);
            return Ok(host_pc);
        }
        let (host_pc, edges) = translate(&mut self.cache, guest_pc, block_exit, syscall_exit)?;
        self.map.insert(guest_pc, host_pc);
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
                Some(&target_host) => patch_site(site, target_host),
                None => self.pending.entry(target_guest).or_default().push(site),
            }
        }
        Ok(host_pc)
    }

    /// Emit (once per cache) the shared inline indirect-branch lookup routine
    /// and return its host address so translated indirect branches can reach it.
    pub fn ensure_ib_lookup(&mut self, block_exit: u64) -> Result<u64, Error> {
        self.cache.ensure_ib_lookup(block_exit)
    }

    /// Flush every translated block and its link bookkeeping. The backing code
    /// buffer is rewound; the guest-PC map and pending links are cleared.
    pub fn reset(&mut self) {
        self.cache.reset();
        self.map.clear();
        self.pending.clear();
    }
}

/// Rewrite the `rel32` displacement at `site` (an address inside the RWX code
/// cache) so its branch lands at `host_pc`. The displacement is measured from
/// the end of its own four bytes. The cache is at most a few megabytes, so the
/// signed distance always fits in an `i32`. Safe to do unsynchronized: patching
/// only happens in the dispatcher, between cache entries, while no translated
/// code is executing; x86 keeps instruction and data caches coherent.
fn patch_site(site: usize, host_pc: u64) {
    let disp = host_pc as i64 - (site as i64 + 4);
    debug_assert!(
        i32::try_from(disp).is_ok(),
        "link displacement {disp} out of rel32 range"
    );
    unsafe { (site as *mut i32).write_unaligned(disp as i32) };
}
