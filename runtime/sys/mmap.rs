//! The guest address space: the translated-block cache ([`BlockCache`]) and the
//! guest mappings Chimera owns on the host.

use std::{
    collections::HashSet,
    io, ptr,
    sync::{
        OnceLock,
        atomic::{AtomicI32, Ordering},
    },
};

use crate::{
    Error,
    arch::x86::{cache::BlockCache, trampoline::guarded_copy},
};

/// Guest page size (x86-64); SMC arming and invalidation work per page.
const PAGE: usize = 4096;

/// Size of the guest's break arena. `PROT_NONE` + `MAP_NORESERVE`, so the
/// reservation costs address space only until the guest grows into it. A guest
/// whose `brk` outgrows the arena sees the call fail with the break unchanged —
/// the same face the kernel shows when `RLIMIT_DATA` is exhausted — and every
/// mainstream allocator falls back to `mmap`.
const BREAK_ARENA_LEN: usize = 1 << 30;

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
    /// Pages Chimera has write-protected on the host to trap self-modifying
    /// code: every page it has translated a block from. A guest store to one
    /// traps into [`on_smc_write`]. A page leaves the set when its trap fires
    /// (and write is restored) or when the guest re-protects or unmaps it; it is
    /// re-armed the next time a block is translated from it. A `HashSet` so the
    /// fault-path `remove` neither allocates nor frees.
    armed: HashSet<usize>,
    /// Pages whose write permission Chimera has restored (after a trap) and not
    /// yet re-armed. Needed to disambiguate a concurrent-store race: two threads
    /// can both store to an armed page and both trap; the first restores write
    /// and disarms, so the second sees the page un-armed even though its store
    /// will now succeed. A fault on a `granted` page is that benign race — re-run
    /// the store — not a genuine fault. Pre-reserved so the fault-path `insert`
    /// does not allocate in the common case.
    granted: HashSet<usize>,
    program_break: Option<usize>,
    /// The reservation the guest's virtualized `brk` lives in; see
    /// [`AddressSpace::init_program_break`].
    break_arena: Option<Region>,
}

impl AddressSpace {
    pub fn new(code_cache_size: usize) -> Result<Self, Error> {
        let mut granted = HashSet::new();
        granted.reserve(1 << 16);
        Ok(Self {
            code: BlockCache::new(code_cache_size)?,
            regions: Vec::new(),
            armed: HashSet::new(),
            granted,
            program_break: None,
            break_arena: None,
        })
    }

    /// Translate (or look up) the block at guest `rip` and return its host PC.
    /// A freshly translated block's guest page(s) are armed for SMC so a later
    /// in-place rewrite of that code traps.
    pub fn resolve(
        &mut self,
        rip: u64,
        block_exit: u64,
        syscall_exit: u64,
        trap_exit: u64,
    ) -> Result<u64, Error> {
        let (host_pc, span) = self
            .code
            .resolve(rip, block_exit, syscall_exit, trap_exit)?;
        if let Some((start, end)) = span {
            self.arm_span(start, end);
        }
        Ok(host_pc)
    }

    pub fn code_contains_range(&self, start: usize, len: usize) -> bool {
        self.code.code_contains_range(start, len)
    }

    pub fn code_allow_writes(&self) {
        self.code.code_allow_writes()
    }

    pub fn code_deny_writes(&self) {
        self.code.code_deny_writes()
    }

    /// A fresh `mmap` reset the host protection of `[start, start+len)`: clear
    /// any stale armed bits and translations left by a previous mapping that
    /// used these addresses, so the new mapping arms and translates from a clean
    /// slate. Without this a reused page keeps its old armed bit, `arm_span`
    /// skips re-protecting it, and a write to the new code never traps.
    pub fn note_map(&mut self, start: usize, len: usize) {
        self.invalidate_and_disarm(start, len);
    }

    /// A guest `mprotect` reset the host protection of `[start, start+len)`
    /// (so its pages leave the armed set) and may precede a rewrite of code on a
    /// W^X-toggled JIT page, so drop the stale translations there. Cheap for a
    /// huge range: only translated pages are visited, and only armed pages
    /// cleared.
    pub fn note_prot(&mut self, start: usize, len: usize) {
        self.invalidate_and_disarm(start, len);
    }

    /// Forget a range the guest unmapped: drop its translations and disarm it.
    /// The host mapping is already gone, so a later mapping that reuses these
    /// addresses starts clean.
    pub fn note_unmap(&mut self, start: usize, len: usize) {
        self.invalidate_and_disarm(start, len);
    }

    /// Handle a write fault at host/guest address `addr` (they share the address
    /// space). If it lands on an armed SMC page, drop that page's translations,
    /// restore write permission so the store completes, and report it handled.
    /// A fault on any other page is not ours — a genuine guest fault. Runs in the
    /// synchronous fault handler, so it allocates nothing.
    pub fn on_smc_write(&mut self, addr: usize) -> bool {
        let page = addr & !(PAGE - 1);
        if !self.armed.remove(&page) {
            // Already disarmed: a benign race if another thread just restored
            // write to this same page (re-run the store); otherwise not ours.
            return self.granted.contains(&page);
        }
        self.code.code_allow_writes();
        self.code.invalidate_page(page as u64);
        self.code.code_deny_writes();
        self.granted.insert(page);
        // Restore the guest's writable mapping (never executable on the host).
        unsafe {
            libc::mprotect(
                page as *mut libc::c_void,
                PAGE,
                libc::PROT_READ | libc::PROT_WRITE,
            );
        }
        true
    }

    /// Write-protect every page a freshly translated block covers, so a later
    /// store to that code traps into [`on_smc_write`]. Only executed pages reach
    /// here, so data and stacks are never armed; static read-only code is armed
    /// too, but the `mprotect` to read-only is a no-op and the guest never writes
    /// it.
    fn arm_span(&mut self, start: u64, end: u64) {
        for page in pages(start as usize, (end - start) as usize) {
            if self.armed.insert(page) {
                self.granted.remove(&page);
                unsafe { libc::mprotect(page as *mut libc::c_void, PAGE, libc::PROT_READ) };
            }
        }
    }

    /// Drop translations for, and disarm, every page in `[start, start+len)`.
    /// The host protection of the range was just reset by the guest (`mprotect`)
    /// or removed (`munmap`), so the armed pages are no longer Chimera's to
    /// re-protect — clearing the bit lets them re-arm on the next translation.
    fn invalidate_and_disarm(&mut self, start: usize, len: usize) {
        if len == 0 {
            return;
        }
        let lo = (start & !(PAGE - 1)) as u64;
        let hi = ((start + len - 1) & !(PAGE - 1)) as u64;
        self.code.invalidate_range(lo, hi);
        self.armed.retain(|&p| (p as u64) < lo || (p as u64) > hi);
        self.granted.retain(|&p| (p as u64) < lo || (p as u64) > hi);
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

    pub fn contains_region(&self, start: usize, len: usize) -> bool {
        let len = round_mapping_len(len);
        if len == 0 {
            return false;
        }
        let Some(end) = start.checked_add(len) else {
            return false;
        };
        self.regions
            .iter()
            .any(|region| region.start <= start && region.start.saturating_add(region.len) >= end)
    }

    pub fn reset(&mut self) {
        self.clear_regions();
        self.code.reset();
        self.armed.clear();
        self.granted.clear();
    }

    /// Reserve the guest's break arena and place the initial program break at
    /// its base.
    ///
    /// The guest's `brk` is virtualized over this private reservation, never
    /// the process's real program break. Chimera and the guest share one
    /// address space, and the host libc's `malloc` keeps a `brk`-backed
    /// `main_arena` that a Rust `#[global_allocator]` cannot shield — glibc's
    /// internal callers (`opendir`, `getaddrinfo`, …) reach `__libc_malloc`
    /// directly. A guest moving the shared kernel break would interleave two
    /// allocators on one segment, and unmapping guest-grown break pages at
    /// teardown would punch a hole below the break that the host allocator
    /// later grows across. Keeping the guest break in a runtime-owned arena
    /// leaves the real break with a single owner for the process's lifetime.
    pub fn init_program_break(&mut self) -> Result<(), Error> {
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                BREAK_ARENA_LEN,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(Error::io("break arena", io::Error::last_os_error()));
        }
        let start = addr as usize;
        self.break_arena = Some(Region {
            start,
            len: BREAK_ARENA_LEN,
        });
        self.program_break = Some(start);
        Ok(())
    }

    /// Service a guest `brk` against the break arena, with the kernel's
    /// contract: success returns the requested break, and a request the arena
    /// cannot satisfy leaves the break unchanged and returns the current one
    /// (which is also how `brk(0)` reads it back). Grown pages join the guest
    /// regions; a shrink discards its pages, so regrowth reads zeroes just as
    /// it does from the kernel.
    pub fn handle_brk(&mut self, requested: usize) -> usize {
        let (Some(current), Some(arena)) = (self.program_break, self.break_arena.clone()) else {
            return 0;
        };
        if requested < arena.start || requested > arena.start + arena.len {
            return current;
        }
        let old_end = round_mapping_len(current);
        let new_end = round_mapping_len(requested);
        if new_end > old_end {
            let ret = unsafe {
                libc::mprotect(
                    old_end as *mut libc::c_void,
                    new_end - old_end,
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            };
            if ret != 0 {
                return current;
            }
            self.add_region(old_end, new_end - old_end);
            self.note_map(old_end, new_end - old_end);
        } else if new_end < old_end {
            // Map fresh `PROT_NONE` pages over the vacated range rather than
            // unmapping it, so the arena's reservation stays in place and no
            // unrelated mapping can land inside it.
            let ret = unsafe {
                libc::mmap(
                    new_end as *mut libc::c_void,
                    old_end - new_end,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED,
                    -1,
                    0,
                )
            };
            if ret == libc::MAP_FAILED {
                return current;
            }
            self.remove_region(new_end, old_end - new_end);
            self.note_unmap(new_end, old_end - new_end);
        }
        self.program_break = Some(requested);
        requested
    }

    fn clear_regions(&mut self) {
        self.program_break = None;
        for region in self.regions.drain(..) {
            let ret = unsafe { libc::munmap(region.start as *mut libc::c_void, region.len) };
            debug_assert_eq!(ret, 0, "guest region munmap failed");
        }
        if let Some(arena) = self.break_arena.take() {
            let ret = unsafe { libc::munmap(arena.start as *mut libc::c_void, arena.len) };
            debug_assert_eq!(ret, 0, "break arena munmap failed");
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

/// The runtime's pid, cached for the self-targeted `process_vm` copies below.
/// glibc has not cached `getpid()` since 2.25, so taking it per copy doubles
/// each copy's syscall bill. A fork invalidates the value (a stale pid would
/// aim the copies at the *parent's* address space), so Chimera registers a
/// `pthread_atfork` child hook for host forks and still drops the cache from
/// [`Thread::reset_after_fork`] for the guest's raw-`clone` fork path, which
/// never runs libc's fork handlers.
///
/// [`Thread::reset_after_fork`]: crate::arch::x86::dispatch::Thread::reset_after_fork
static CACHED_PID: AtomicI32 = AtomicI32::new(0);

extern "C" fn reset_cached_pid_after_fork() {
    reset_cached_pid();
}

pub fn init() -> Result<(), Error> {
    static INIT: OnceLock<Result<(), i32>> = OnceLock::new();

    match INIT.get_or_init(|| {
        let ret = unsafe { libc::pthread_atfork(None, None, Some(reset_cached_pid_after_fork)) };
        if ret == 0 { Ok(()) } else { Err(ret) }
    }) {
        Ok(()) => Ok(()),
        Err(err) => Err(Error::io(
            "pthread_atfork",
            io::Error::from_raw_os_error(*err),
        )),
    }
}

fn own_pid() -> libc::pid_t {
    let pid = CACHED_PID.load(Ordering::Relaxed);
    if pid != 0 {
        return pid;
    }
    let pid = unsafe { libc::getpid() };
    CACHED_PID.store(pid, Ordering::Relaxed);
    pid
}

/// Drop the cached pid in the child of a fork; the next copy re-reads it.
pub fn reset_cached_pid() {
    CACHED_PID.store(0, Ordering::Relaxed);
}

/// Copy `buf.len()` bytes out of guest memory at `addr` without trusting the
/// pointer. Chimera and the guest share one address space, so the copy is a
/// plain load loop ([`guarded_copy`]) whose faults the `SIGSEGV`/`SIGBUS`
/// handler recovers — the kernel's `copy_from_user` exception-table pattern.
/// An unmapped or partially mapped range fails the copy exactly where a bare
/// dereference would fault the runtime, and the common, fully mapped case
/// costs a `memcpy` instead of a system call: the translator takes this path
/// for every basic block it decodes, hundreds of thousands of times in a JIT
/// warm-up. Returns false on any failed copy, so a caller reports `EFAULT`
/// (or forwards for the kernel to) the way a native `copy_from_user` failure
/// would.
pub fn copy_from_guest(addr: u64, buf: &mut [u8]) -> bool {
    if buf.is_empty() {
        return true;
    }
    if addr == 0 {
        return false;
    }
    unsafe { guarded_copy(buf.as_mut_ptr(), addr as *const u8, buf.len()) != 0 }
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
    let copied = unsafe { libc::process_vm_writev(own_pid(), &local, 1, &remote, 1, 0) };
    copied == buf.len() as isize
}

/// Page-aligned base addresses of every page the range `[start, start+len)`
/// touches. Empty when `len` is zero.
fn pages(start: usize, len: usize) -> impl Iterator<Item = usize> {
    let first = start & !(PAGE - 1);
    let end = start.saturating_add(len);
    let last = end.saturating_sub(1) & !(PAGE - 1);
    let count = if len == 0 {
        0
    } else {
        (last - first) / PAGE + 1
    };
    (0..count).map(move |i| first + i * PAGE)
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
