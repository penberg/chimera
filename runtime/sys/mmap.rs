//! The guest address space: the translated-block cache ([`BlockCache`]) and the
//! guest mappings Chimera owns on the host.

use std::{
    collections::HashSet,
    io,
    sync::{
        OnceLock,
        atomic::{AtomicI32, Ordering},
    },
};

use crate::{Error, arch::cache::BlockCache};

/// The guarded load loop the Linux fault-safe read uses. It lives in the x86
/// trampoline, which is the only host that has one; Darwin reads guest memory
/// through `mach_vm_read_overwrite` instead (see `guest_read`).
#[cfg(target_os = "linux")]
use crate::arch::x86::trampoline::guarded_copy;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Region {
    start: usize,
    len: usize,
    /// Whether the runtime created this mapping for the guest (its image and
    /// its initial stack) rather than merely observing the guest ask for it.
    /// Only these are unmapped when an `execve` replaces the image — see
    /// [`AddressSpace::reset`].
    runtime_owned: bool,
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

    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub fn code_contains_range(&self, start: usize, len: usize) -> bool {
        self.code.code_contains_range(start, len)
    }

    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub fn code_allow_writes(&self) {
        self.code.code_allow_writes()
    }

    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
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
        let page_size = host_page_size();
        let page = addr & !(page_size - 1);
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
                page_size,
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
    ///
    /// Only pages inside a region the guest mapped are armed. A guest can also
    /// execute code the runtime never mapped for it — on Darwin it shares
    /// Chimera's already-loaded libSystem, and the translator reads (and would
    /// otherwise arm) the very shared-cache pages Chimera itself calls into.
    /// Re-protecting those to read-only strips their execute permission out
    /// from under the runtime, whose next call into that function faults on
    /// the instruction fetch — a fault no SMC servicing can resolve, since
    /// restoring write does not restore execute. Such code is not the guest's
    /// to modify anyway, so skipping it costs no coverage.
    fn arm_span(&mut self, start: u64, end: u64) {
        let page_size = host_page_size();
        for page in pages(start as usize, (end - start) as usize) {
            if !self.in_guest_region(page) {
                continue;
            }
            if self.armed.insert(page) {
                self.granted.remove(&page);
                unsafe { libc::mprotect(page as *mut libc::c_void, page_size, libc::PROT_READ) };
            }
        }
    }

    /// Describe `addr` for a crash report: the recorded region containing it
    /// (start, len, runtime-owned), and whether its page is armed or granted.
    /// Allocation-free; runs in the fault handler.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn describe_addr(&self, addr: usize) -> (Option<(usize, usize, bool)>, bool, bool) {
        let page = addr & !(host_page_size() - 1);
        let region = match self.regions.binary_search_by_key(&addr, |r| r.start) {
            Ok(i) => Some(&self.regions[i]),
            Err(0) => None,
            Err(i) => Some(&self.regions[i - 1]),
        }
        .filter(|r| addr < r.start.saturating_add(r.len))
        .map(|r| (r.start, r.len, r.runtime_owned));
        (
            region,
            self.armed.contains(&page),
            self.granted.contains(&page),
        )
    }

    /// Whether `[start, start+len)` lies entirely within one recorded guest
    /// region — the precondition for the runtime to unmap it on the guest's
    /// behalf.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn covers_range(&self, start: usize, len: usize) -> bool {
        match self.regions.binary_search_by_key(&start, |r| r.start) {
            Ok(i) => start.saturating_add(len) <= self.regions[i].start + self.regions[i].len,
            Err(0) => false,
            Err(i) => {
                let region = &self.regions[i - 1];
                start.saturating_add(len) <= region.start.saturating_add(region.len)
            }
        }
    }

    /// Whether `addr` falls in a mapping the runtime made for the guest.
    /// `regions` is kept sorted and coalesced, so this is a binary search.
    fn in_guest_region(&self, addr: usize) -> bool {
        match self.regions.binary_search_by_key(&addr, |r| r.start) {
            Ok(_) => true,
            Err(0) => false,
            Err(i) => {
                let region = &self.regions[i - 1];
                addr < region.start.saturating_add(region.len)
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
        let page_size = host_page_size();
        let lo = (start & !(page_size - 1)) as u64;
        let hi = ((start + len - 1) & !(page_size - 1)) as u64;
        self.code.invalidate_range(lo, hi);
        self.armed.retain(|&p| (p as u64) < lo || (p as u64) > hi);
        self.granted.retain(|&p| (p as u64) < lo || (p as u64) > hi);
    }

    /// Record a mapping the guest made, for SMC arming and bookkeeping.
    pub fn add_region(&mut self, start: usize, len: usize) {
        self.insert_region(start, len, false);
    }

    /// Record a mapping the runtime made *for* the guest — its image and its
    /// initial stack — which an `execve` must tear down to install the next
    /// image.
    pub fn add_runtime_region(&mut self, start: usize, len: usize) {
        self.insert_region(start, len, true);
    }

    fn insert_region(&mut self, start: usize, len: usize, runtime_owned: bool) {
        let len = round_mapping_len(len);
        if len == 0 {
            return;
        }
        self.remove_region(start, len);
        self.regions.push(Region {
            start,
            len,
            runtime_owned,
        });
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
                    runtime_owned: region.runtime_owned,
                });
            }
            if end < region_end {
                kept.push(Region {
                    start: end,
                    len: region_end - end,
                    runtime_owned: region.runtime_owned,
                });
            }
        }
        self.regions = kept;
    }

    /// Linux-only caller (`mremap` handling); compiled for tests everywhere.
    #[cfg(any(target_os = "linux", test))]
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

    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
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
        self.code.reset();
        // Give every armed page its write permission back before forgetting
        // it: pages in regions that survive the teardown (a guest's own
        // mappings deliberately do) would otherwise stay read-only with no
        // bookkeeping left to service the trap, and the next write — the
        // runtime's own, via the shared libSystem, included — faults with no
        // one able to fix it. For a page about to be unmapped this is a
        // harmless failed mprotect.
        let page_size = host_page_size();
        for &page in &self.armed {
            unsafe {
                libc::mprotect(
                    page as *mut libc::c_void,
                    page_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                );
            }
        }
        self.armed.clear();
        self.granted.clear();
        self.clear_regions();
    }

    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub fn set_program_break(&mut self, brk: usize) {
        self.program_break = Some(brk);
    }

    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub fn update_program_break(&mut self, brk: usize) {
        if let Some(old_brk) = self.program_break {
            let old_end = round_mapping_len(old_brk);
            let new_end = round_mapping_len(brk);
            if new_end > old_end {
                self.add_region(old_end, new_end - old_end);
            } else if new_end < old_end {
                self.remove_region(new_end, old_end - new_end);
            }
        }
        self.program_break = Some(brk);
    }

    /// Unmap the mappings the runtime made for the guest and forget every
    /// region. A mapping the *guest* asked for is deliberately left alone: on
    /// Darwin the runtime and the guest share one dyld, so a guest `dlopen`
    /// runs the loader's own code and its image mappings arrive through the
    /// same `mmap` interception as everything else. Unmapping those would
    /// leave the shared dyld referencing images whose pages are gone, and its
    /// next `dlsym` — walking the image list — faults reading a Mach header.
    /// The cost is that a guest's own anonymous mappings outlive an `execve`
    /// that should have replaced them, which leaks address space but changes
    /// nothing the new image can observe.
    fn clear_regions(&mut self) {
        self.program_break = None;
        for region in self.regions.drain(..) {
            if !region.runtime_owned {
                continue;
            }
            #[cfg(target_os = "macos")]
            if crate::trace::trace() {
                eprintln!("chimera: exec unmaps {:#x}+{:#x}", region.start, region.len);
            }
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
                // Only merge like with like: ownership is what decides
                // whether an `execve` unmaps a range, so absorbing a guest
                // mapping into an abutting runtime-owned one hands it to the
                // teardown in [`AddressSpace::clear_regions`] — which is the
                // very thing that function refuses to do on purpose.
                if region.start <= last_end && region.runtime_owned == last.runtime_owned {
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

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
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
    guest_read(addr, buf)
}

/// Linux fault-safe guest read: a guarded load loop whose faults the
/// `SIGSEGV`/`SIGBUS` handler recognises and reports as a short copy.
#[cfg(target_os = "linux")]
fn guest_read(addr: u64, buf: &mut [u8]) -> bool {
    unsafe { guarded_copy(buf.as_mut_ptr(), addr as *const u8, buf.len()) != 0 }
}

/// Darwin fault-safe guest read: `mach_vm_read_overwrite` against this task's
/// own port. An unmapped or unreadable source returns a nonzero
/// `kern_return_t` rather than faulting the runtime, the way `process_vm_readv`
/// returns short.
#[cfg(target_os = "macos")]
fn guest_read(addr: u64, buf: &mut [u8]) -> bool {
    // `mach_vm_read_overwrite` is a MIG routine, so it can fail for reasons
    // that say nothing about `addr`. Measured here: `MACH_SEND_INVALID_REPLY`
    // (0x10000009) on a read of a mapping the guest could dereference a
    // moment later — the *message* failed, not the copy, because the thread's
    // MIG reply port was no longer valid for this caller. Chimera and the
    // guest share one libSystem, and on the main thread they share one TSD
    // array, so the reply port cached in it is not exclusively the runtime's.
    // Reporting `EFAULT` for such a failure turns a transient into a hard
    // error: `prepare_exec` reads the `execve` path this way, and rustc
    // reports "unable to run `rust-objcopy`: Bad address" and fails the
    // build. A retry gets a freshly allocated reply port.
    //
    // Only the send family is retried. A genuine `KERN_INVALID_ADDRESS` is
    // the answer the caller asked for, and repeating it would both cost time
    // and blunt the fault-safety this function exists to provide.
    const MACH_SEND_FAMILY: u32 = 0x1000_0000;
    for _ in 0..4 {
        let mut out: u64 = 0;
        let kr = unsafe {
            mach_vm_read_overwrite(
                mach_task_self_,
                addr,
                buf.len() as u64,
                buf.as_mut_ptr() as u64,
                &mut out,
            )
        };
        if kr == 0 {
            return out == buf.len() as u64;
        }
        if (kr as u32) & MACH_SEND_FAMILY == 0 {
            return false;
        }
    }
    false
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
    guest_write(addr, buf)
}

/// Linux fault-safe guest write: a self-targeted `process_vm_writev`.
#[cfg(target_os = "linux")]
fn guest_write(addr: u64, buf: &[u8]) -> bool {
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

/// Darwin fault-safe guest write: `mach_vm_write` against this task's own port.
/// An unmapped or read-only target returns a nonzero `kern_return_t` rather
/// than faulting the runtime, the way `process_vm_writev` returns short.
#[cfg(target_os = "macos")]
fn guest_write(addr: u64, buf: &[u8]) -> bool {
    let kr = unsafe { mach_vm_write(mach_task_self_, addr, buf.as_ptr() as u64, buf.len() as u32) };
    kr == 0
}

// Mach VM primitives for fault-safe self-copies (declared here rather than via
// the `libc` crate, which does not surface the `mach_vm_*` family).
// `mach_task_self_` is the task port for the current process.
#[cfg(target_os = "macos")]
unsafe extern "C" {
    static mach_task_self_: u32;
    fn mach_vm_read_overwrite(
        target: u32,
        address: u64,
        size: u64,
        data: u64,
        out_size: *mut u64,
    ) -> i32;
    fn mach_vm_write(target: u32, address: u64, data: u64, count: u32) -> i32;
}

/// Page-aligned base addresses of every page the range `[start, start+len)`
/// touches, at host page granularity. Empty when `len` is zero.
fn pages(start: usize, len: usize) -> impl Iterator<Item = usize> {
    let page_size = host_page_size();
    let first = start & !(page_size - 1);
    let end = start.saturating_add(len);
    let last = end.saturating_sub(1) & !(page_size - 1);
    let count = if len == 0 {
        0
    } else {
        (last - first) / page_size + 1
    };
    (0..count).map(move |i| first + i * page_size)
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

/// A guest-memory read failed or overran its bound while decoding a syscall's
/// string arguments. Carried as an [`Error::Io`] so the syscall driver hands
/// the errno to the caller exactly as the kernel's user-copy path would.
fn arg_error(errno: i32, what: &str) -> Error {
    Error::io(
        format!("guest argument: {what}"),
        std::io::Error::from_raw_os_error(errno),
    )
}

/// Copy a NUL-terminated string out of guest memory without trusting the
/// pointer (see [`copy_from_guest`]), in chunks that stop at host-page
/// boundaries so a string ending just before an unmapped page is not failed by
/// over-reading past it. `cap` is the kernel's limit for this kind of string:
/// filling it without a NUL fails with `toolong` (`ENAMETOOLONG` for a
/// pathname, `E2BIG` for an argv/envp entry), and an unreadable byte fails
/// with `EFAULT`, the way the kernel's user-copy would.
pub fn read_guest_cstr(ptr: u64, cap: usize, toolong: i32) -> Result<Vec<u8>, Error> {
    let page = host_page_size() as u64;
    let mut out = Vec::new();
    let mut addr = ptr;
    while out.len() < cap {
        let page_left = (page - (addr % page)) as usize;
        let mut chunk = vec![0u8; page_left.min(cap - out.len())];
        if !copy_from_guest(addr, &mut chunk) {
            return Err(arg_error(libc::EFAULT, "unreadable string"));
        }
        if let Some(nul) = chunk.iter().position(|&b| b == 0) {
            chunk.truncate(nul);
            out.extend_from_slice(&chunk);
            return Ok(out);
        }
        addr += chunk.len() as u64;
        out.extend_from_slice(&chunk);
    }
    Err(arg_error(toolong, "string exceeds its limit"))
}

/// Copy a NULL-terminated array of C-string pointers (an argv or envp) and its
/// strings out of guest memory, trusting none of the pointers. A null array
/// pointer is the kernels' "no arguments" degenerate case, not a fault. More
/// than `max_count` entries — or a single string longer than `max_strlen` —
/// fails with `E2BIG`, the kernel's argument-block bound.
pub fn read_guest_ptr_array(
    ptr: u64,
    max_count: usize,
    max_strlen: usize,
) -> Result<Vec<Vec<u8>>, Error> {
    if ptr == 0 {
        return Ok(Vec::new());
    }
    let mut out: Vec<Vec<u8>> = Vec::new();
    loop {
        if out.len() >= max_count {
            return Err(arg_error(libc::E2BIG, "argument list too long"));
        }
        let mut raw = [0u8; 8];
        if !copy_from_guest(ptr + (out.len() as u64) * 8, &mut raw) {
            return Err(arg_error(libc::EFAULT, "unreadable argument pointer"));
        }
        let entry = u64::from_ne_bytes(raw);
        if entry == 0 {
            return Ok(out);
        }
        out.push(read_guest_cstr(entry, max_strlen, libc::E2BIG)?);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

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
        let base = mmap_anon(host_page_size() * 2);

        addr_space.add_region(base, host_page_size());
        addr_space.add_region(base + host_page_size(), host_page_size());

        assert_eq!(
            addr_space.regions,
            vec![Region {
                start: base,
                len: host_page_size() * 2,
                runtime_owned: false,
            }]
        );
    }

    #[test]
    fn remove_region_splits_partial_unmap() {
        let mut addr_space = AddressSpace::new(crate::DEFAULT_CODE_CACHE_SIZE).unwrap();
        let base = mmap_anon(host_page_size() * 3);
        addr_space.add_region(base, host_page_size() * 3);

        addr_space.remove_region(base + host_page_size(), host_page_size());
        assert_eq!(
            addr_space.regions,
            vec![
                Region {
                    start: base,
                    len: host_page_size(),
                    runtime_owned: false,
                },
                Region {
                    start: base + (host_page_size() * 2),
                    len: host_page_size(),
                    runtime_owned: false,
                },
            ]
        );

        let ret = unsafe {
            libc::munmap(
                (base + host_page_size()) as *mut libc::c_void,
                host_page_size(),
            )
        };
        assert_eq!(ret, 0);
    }

    #[test]
    fn remap_region_moves_mapping() {
        let mut addr_space = AddressSpace::new(crate::DEFAULT_CODE_CACHE_SIZE).unwrap();
        let old = mmap_anon(host_page_size());
        let new = mmap_anon(host_page_size());

        addr_space.add_region(old, host_page_size());
        addr_space.remap_region(old, host_page_size(), new, host_page_size(), false);

        assert_eq!(
            addr_space.regions,
            vec![Region {
                start: new,
                len: host_page_size(),
                runtime_owned: false,
            }]
        );

        let ret = unsafe { libc::munmap(old as *mut libc::c_void, host_page_size()) };
        assert_eq!(ret, 0);
    }
}
