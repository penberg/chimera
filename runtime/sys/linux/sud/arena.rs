//! The guest half of the address space.
//!
//! The dispatch-exempt range is everything at or above [`EXEMPT_FLOOR`], so
//! the placement of guest *code* is the security boundary: an instruction
//! that lives above the line issues syscalls unintercepted. Guest images are
//! therefore mapped into an arena below the line, and so is every `NULL`-hint
//! guest `mmap`, which is what extends the guarantee past load time — a JIT
//! writes its code into arena pages, so the syscall instructions it emits
//! trap like any other. The arena is one bump allocator over a fixed range,
//! shared by every thread of the process because the address space is.

use std::sync::{
    Mutex,
    atomic::{AtomicU64, Ordering},
};

use crate::{Error, SyscallResult, SystemCall};

use super::{
    super::{
        elf::{LoadedElf, PAGE_SIZE, ParsedElf, map_elf_native},
        exec::build_stack,
        syscall::host_syscall,
    },
    EXEMPT_FLOOR,
};

/// Where guest images and guest `NULL`-hint mappings are bump-allocated,
/// safely below [`EXEMPT_FLOOR`].
const GUEST_ARENA_BASE: u64 = 0x5100_0000_0000;
const GUEST_ARENA_CEILING: u64 = 0x5400_0000_0000;

const _: () = assert!(GUEST_ARENA_CEILING <= EXEMPT_FLOOR);

/// Gap left after each image placed in the arena, room for `brk`-less heaps
/// and a guard against off-by-a-page neighbors.
const ARENA_IMAGE_GAP: u64 = 2 * 1024 * 1024;

pub struct Arena {
    /// The bump pointer. Only ever advanced: a concurrent `mmap` that loses
    /// the race for a hint sees `EEXIST` and moves past it, so the pointer
    /// converges on free space without a lock.
    bump: AtomicU64,
    /// Mappings owned by the current guest image, in the arena or not —
    /// `ET_EXEC` segments sit at their fixed low addresses and the initial
    /// stack where the kernel put it — torn down together with the arena when
    /// an `execve` replaces the image.
    regions: Mutex<Vec<(u64, u64)>>,
}

impl Arena {
    pub fn new() -> Self {
        Self {
            bump: AtomicU64::new(GUEST_ARENA_BASE),
            regions: Mutex::new(Vec::new()),
        }
    }

    /// Map an image and its interpreter into the arena and build the initial
    /// stack for it; returns the entry `rip` and the initial `rsp`. The
    /// commit phase of an exec: the mappings it makes cannot be rolled back.
    pub fn load_image(
        &self,
        parsed: &ParsedElf,
        parsed_interp: Option<&ParsedElf>,
        argv: &[Vec<u8>],
        envp: &[Vec<u8>],
        execfn: &[u8],
    ) -> Result<(u64, u64), Error> {
        let main = self.load(parsed)?;
        let (rip, interp_base, interp) = match parsed_interp {
            Some(parsed_interp) => {
                let interp = self.load(parsed_interp)?;
                (interp.entry, interp.base, Some(interp))
            }
            None => (main.entry, 0, None),
        };
        let (rsp, stack_start, stack_len) = build_stack(argv, envp, execfn, &main, interp_base)?;
        let mut regions = self.regions.lock().unwrap();
        regions.extend(&main.regions);
        if let Some(interp) = &interp {
            regions.extend(&interp.regions);
        }
        regions.push((stack_start as u64, stack_len as u64));
        Ok((rip, rsp))
    }

    /// Map one image, drawing `ET_DYN` placement from the bump pointer and
    /// advancing it past whatever landed in the arena.
    fn load(&self, parsed: &ParsedElf) -> Result<LoadedElf, Error> {
        let elf = map_elf_native(parsed, self.bump.load(Ordering::Relaxed))?;
        for &(start, len) in &elf.regions {
            if (GUEST_ARENA_BASE..GUEST_ARENA_CEILING).contains(&start) {
                let end = (start + len + ARENA_IMAGE_GAP + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
                self.bump.fetch_max(end, Ordering::Relaxed);
            }
        }
        Ok(elf)
    }

    /// Unmap the current image: the tracked regions, then the arena wholesale
    /// up to its watermark. Guest mappings the kernel placed on its own — an
    /// explicit high hint, past the arena's ceiling — are not tracked and
    /// leak across an `execve`.
    pub fn teardown(&self) {
        for (start, len) in self.regions.lock().unwrap().drain(..) {
            unsafe { libc::munmap(start as *mut libc::c_void, len as usize) };
        }
        let watermark = self.bump.swap(GUEST_ARENA_BASE, Ordering::Relaxed);
        if watermark > GUEST_ARENA_BASE {
            unsafe {
                libc::munmap(
                    GUEST_ARENA_BASE as *mut libc::c_void,
                    (watermark - GUEST_ARENA_BASE) as usize,
                )
            };
        }
    }

    /// Service a guest `mmap` whose fd is already resolved: steer a
    /// `NULL`-hint request into the arena, so fresh guest pages — code the
    /// guest may write and jump to — stay below the exempt floor. Explicitly
    /// placed requests forward untouched.
    pub fn mmap(&self, call: &mut SystemCall) {
        let flags = call.args[3] as libc::c_int;
        let fixed = flags & (libc::MAP_FIXED | libc::MAP_FIXED_NOREPLACE) != 0;
        if call.args[0] != 0 || fixed {
            call.set_result(host_syscall(call));
            return;
        }
        let len = (call.args[1] + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        loop {
            let hint = self.bump.load(Ordering::Relaxed);
            if hint + len > GUEST_ARENA_CEILING {
                // Arena exhausted; let the kernel place it and accept that a
                // syscall from such a page would go unintercepted.
                call.set_result(host_syscall(call));
                return;
            }
            let placed = SystemCall::new(
                call.number,
                [
                    hint,
                    call.args[1],
                    call.args[2],
                    (flags | libc::MAP_FIXED_NOREPLACE) as u64,
                    call.args[4],
                    call.args[5],
                ],
            );
            let result = host_syscall(&placed);
            match result {
                SyscallResult::Error(libc::EEXIST) => {
                    self.bump
                        .fetch_max(hint + len.max(ARENA_IMAGE_GAP), Ordering::Relaxed);
                }
                _ => {
                    if matches!(result, SyscallResult::Ok(_)) {
                        self.bump.fetch_max(hint + len, Ordering::Relaxed);
                    }
                    call.set_result(result);
                    return;
                }
            }
        }
    }

    /// Take the arena's lock, to be held across a forwarded `fork` (see
    /// `Process::lock_for_fork`).
    pub fn lock_for_fork(&self) -> std::sync::MutexGuard<'_, Vec<(u64, u64)>> {
        self.regions.lock().unwrap()
    }
}
