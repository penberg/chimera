//! The guest address space: the translated-block cache ([`BlockCache`]) and the
//! guest mappings Chimera owns on the host.

use std::sync::OnceLock;

use crate::{Error, arch::x86::cache::BlockCache};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Region {
    start: usize,
    len: usize,
}

/// A guest address space: the translated-block cache and the host mappings
/// created for the guest. Mirrors the Linux kernel's `mm_struct`.
pub struct AddressSpace {
    pub code: BlockCache,
    regions: Vec<Region>,
}

impl AddressSpace {
    pub fn new(code_cache_size: usize) -> Result<Self, Error> {
        Ok(Self {
            code: BlockCache::new(code_cache_size)?,
            regions: Vec::new(),
        })
    }

    pub fn add_region(&mut self, start: usize, len: usize) {
        let len = round_mapping_len(len);
        if len == 0 {
            return;
        }
        self.remove_region(start, len);
        self.regions.push(Region { start, len });
        self.coalesce_regions();
    }

    pub fn remove_region(&mut self, start: usize, len: usize) {
        let len = round_mapping_len(len);
        if len == 0 {
            return;
        }
        let Some(end) = start.checked_add(len) else {
            return;
        };
        let mut kept = Vec::with_capacity(self.regions.len() + 1);
        for region in self.regions.drain(..) {
            let region_end = region.start.saturating_add(region.len);
            if end <= region.start || start >= region_end {
                kept.push(region);
                continue;
            }
            if region.start < start {
                kept.push(Region {
                    start: region.start,
                    len: start - region.start,
                });
            }
            if end < region_end {
                kept.push(Region {
                    start: end,
                    len: region_end - end,
                });
            }
        }
        self.regions = kept;
    }

    pub fn remap_region(
        &mut self,
        old_start: usize,
        old_len: usize,
        new_start: usize,
        new_len: usize,
        dontunmap: bool,
    ) {
        let old_len = round_mapping_len(old_len);
        let new_len = round_mapping_len(new_len);

        if dontunmap {
            self.add_region(new_start, new_len);
            return;
        }
        if old_len != 0 {
            self.remove_region(old_start, old_len);
        }
        self.add_region(new_start, new_len);
    }

    pub fn reset(&mut self) {
        self.clear_regions();
        self.code.reset();
    }

    fn clear_regions(&mut self) {
        for region in self.regions.drain(..) {
            let ret = unsafe { libc::munmap(region.start as *mut libc::c_void, region.len) };
            debug_assert_eq!(ret, 0, "guest region munmap failed");
        }
    }

    fn coalesce_regions(&mut self) {
        self.regions.sort_unstable_by_key(|region| region.start);
        let mut merged: Vec<Region> = Vec::with_capacity(self.regions.len());
        for region in self.regions.drain(..) {
            if let Some(last) = merged.last_mut() {
                let last_end = last.start.saturating_add(last.len);
                let region_end = region.start.saturating_add(region.len);
                if region.start <= last_end {
                    last.len = region_end.max(last_end) - last.start;
                    continue;
                }
            }
            merged.push(region);
        }
        self.regions = merged;
    }
}

impl Drop for AddressSpace {
    fn drop(&mut self) {
        self.clear_regions();
    }
}

/// Copy `buf` into guest memory at `addr` without trusting the pointer — the
/// write-side twin of [`copy_from_guest`], a self-targeted
/// `process_vm_writev`. The kernel walks the page tables, so an unmapped or
/// read-only range fails the copy exactly where a raw store would fault the
/// runtime. Returns false on any failed or short copy; callers that mirror
/// one of the kernel's unchecked `put_user` sites (the clone set-TID words,
/// the exit-time `clear_child_tid` store) ignore the result, since the kernel
/// skips those writes silently.
pub fn copy_to_guest(addr: u64, buf: &[u8]) -> bool {
    if buf.is_empty() {
        return true;
    }
    if addr == 0 {
        return false;
    }
    let local = libc::iovec {
        iov_base: buf.as_ptr() as *mut libc::c_void,
        iov_len: buf.len(),
    };
    let remote = libc::iovec {
        iov_base: addr as *mut libc::c_void,
        iov_len: buf.len(),
    };
    let copied = unsafe { libc::process_vm_writev(libc::getpid(), &local, 1, &remote, 1, 0) };
    copied == buf.len() as isize
}

fn round_mapping_len(len: usize) -> usize {
    if len == 0 {
        return 0;
    }

    let page_size = host_page_size();
    match len.checked_add(page_size - 1) {
        Some(rounded) => rounded / page_size * page_size,
        None => usize::MAX / page_size * page_size,
    }
}

fn host_page_size() -> usize {
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();

    *PAGE_SIZE.get_or_init(|| {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        assert!(page_size > 0, "host page size unavailable");
        page_size as usize
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    const PAGE_SIZE: usize = 4096;

    fn mmap_anon(len: usize) -> usize {
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(addr, libc::MAP_FAILED);
        addr as usize
    }

    #[test]
    fn add_region_coalesces_adjacent_ranges() {
        let mut addr_space = AddressSpace::new(crate::DEFAULT_CODE_CACHE_SIZE).unwrap();
        let base = mmap_anon(PAGE_SIZE * 2);

        addr_space.add_region(base, PAGE_SIZE);
        addr_space.add_region(base + PAGE_SIZE, PAGE_SIZE);

        assert_eq!(
            addr_space.regions,
            vec![Region {
                start: base,
                len: PAGE_SIZE * 2,
            }]
        );
    }

    #[test]
    fn remove_region_splits_partial_unmap() {
        let mut addr_space = AddressSpace::new(crate::DEFAULT_CODE_CACHE_SIZE).unwrap();
        let base = mmap_anon(PAGE_SIZE * 3);
        addr_space.add_region(base, PAGE_SIZE * 3);

        addr_space.remove_region(base + PAGE_SIZE, PAGE_SIZE);
        assert_eq!(
            addr_space.regions,
            vec![
                Region {
                    start: base,
                    len: PAGE_SIZE,
                },
                Region {
                    start: base + (PAGE_SIZE * 2),
                    len: PAGE_SIZE,
                },
            ]
        );

        let ret = unsafe { libc::munmap((base + PAGE_SIZE) as *mut libc::c_void, PAGE_SIZE) };
        assert_eq!(ret, 0);
    }

    #[test]
    fn remap_region_moves_mapping() {
        let mut addr_space = AddressSpace::new(crate::DEFAULT_CODE_CACHE_SIZE).unwrap();
        let old = mmap_anon(PAGE_SIZE);
        let new = mmap_anon(PAGE_SIZE);

        addr_space.add_region(old, PAGE_SIZE);
        addr_space.remap_region(old, PAGE_SIZE, new, PAGE_SIZE, false);

        assert_eq!(
            addr_space.regions,
            vec![Region {
                start: new,
                len: PAGE_SIZE,
            }]
        );

        let ret = unsafe { libc::munmap(old as *mut libc::c_void, PAGE_SIZE) };
        assert_eq!(ret, 0);
    }
}
