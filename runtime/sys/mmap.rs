//! The guest address space: the cache of translated blocks and the map from
//! guest PC to the host PC where each block begins.

use std::collections::HashMap;

use crate::{Error, arch::x86::translate::CodeCache};

/// A guest address space: the cache of translated blocks and the map from
/// guest PC to the host PC where each block begins. Mirrors the Linux kernel's
/// `mm_struct`.
pub struct AddressSpace {
    pub cache: CodeCache,
    pub map: HashMap<u64, u64>,
}

impl AddressSpace {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            cache: CodeCache::new()?,
            map: HashMap::new(),
        })
    }

    pub fn reset(&mut self) {
        self.cache.reset();
        self.map.clear();
    }
}
