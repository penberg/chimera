//! The Linux x86-64 raw-syscall bridge. This is the one place Chimera
//! actually reaches the host kernel; the dispatch logic around it lives in
//! [`crate::syscall`].

use std::arch::asm;

use crate::{SyscallResult, SystemCall, sys::mmap::copy_from_guest};

/// Issue the host kernel's `syscall` instruction with `call`'s number in `rax`
/// and the six argument registers in Linux x86-64 syscall ABI order. Decodes
/// the kernel's negative-errno convention into a [`SyscallResult`] so callers
/// don't have to inspect the sign themselves.
pub fn host_syscall(call: &SystemCall) -> SyscallResult {
    let ret: i64;
    unsafe {
        asm!(
            "syscall",
            in("rax") call.number,
            in("rdi") call.args[0],
            in("rsi") call.args[1],
            in("rdx") call.args[2],
            in("r10") call.args[3],
            in("r8")  call.args[4],
            in("r9")  call.args[5],
            lateout("rax") ret,
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack, preserves_flags),
        );
    }
    // Linux signals errors in the closed range `[-4095, -1]`; anything else
    // is a successful result (including "negative-looking" high addresses
    // `mmap` can hand back).
    if (-4095..0).contains(&ret) {
        SyscallResult::Error(-ret as i32)
    } else {
        SyscallResult::Ok(ret)
    }
}

/// The base `struct clone_args` (`CLONE_ARGS_SIZE_VER0`): the 8 `u64` fields
/// every `clone3` must supply. The kernel rejects a smaller struct with
/// `EINVAL` and one larger than a page with `E2BIG`.
const CLONE_ARGS_SIZE_VER0: usize = 64;
const CLONE_ARGS_SIZE_MAX: usize = 4096;

/// `clone3`'s `CLONE_CLEAR_SIGHAND`. Defined here because the flag lives in
/// bit 32 and the libc crate's `c_int` constant truncates it to 0. Only
/// `clone3` accepts it — the legacy `clone` entry keeps just the low 32 flag
/// bits. It must never reach the host: the kernel would flush *Chimera's*
/// handlers out of the child's slots, so each backend strips it from the
/// forwarded call and applies it to the guest's virtual table instead.
pub const CLONE_CLEAR_SIGHAND: u64 = 1 << 32;

/// A guest `clone3` argument block, copied out of guest memory.
///
/// `clone3` only requires the guest's struct to be readable, so a shape that
/// has to be patched before it reaches the host — `CLONE_VM` stripped from a
/// spawn, `CLONE_CLEAR_SIGHAND` taken off a fork — is forwarded from this
/// private copy rather than rewritten in place, which would be illegal in a
/// read-only mapping and visible to the caller afterwards.
pub struct Clone3Args {
    raw: [u8; CLONE_ARGS_SIZE_MAX],
}

impl Clone3Args {
    /// Copy the struct `clone3` points at, fault-safely (see
    /// `copy_from_guest`). The error is the errno the kernel would report
    /// for the same request — `EINVAL` for a struct smaller than the base,
    /// `E2BIG` for one larger than a page, `EFAULT` for an unreadable one —
    /// so a caller can either fail the call with it or forward the original
    /// for the kernel's own verdict.
    pub fn read(args_ptr: u64, size: u64) -> Result<Self, i32> {
        let size = usize::try_from(size).map_err(|_| libc::E2BIG)?;
        if size < CLONE_ARGS_SIZE_VER0 {
            return Err(libc::EINVAL);
        }
        if size > CLONE_ARGS_SIZE_MAX {
            return Err(libc::E2BIG);
        }
        let mut args = Self {
            raw: [0; CLONE_ARGS_SIZE_MAX],
        };
        if !copy_from_guest(args_ptr, &mut args.raw[..size]) {
            return Err(libc::EFAULT);
        }
        Ok(args)
    }

    /// The base fields in uapi `<linux/sched.h>` order: flags, pidfd,
    /// child_tid, parent_tid, exit_signal, stack, stack_size, tls. Unlike
    /// `clone`, `stack` is the *lowest* address of the child stack and
    /// `stack_size` its length, so the child's stack pointer is their sum.
    pub fn fields(&self) -> [u64; 8] {
        let mut fields = [0u64; 8];
        for (slot, chunk) in fields
            .iter_mut()
            .zip(self.raw[..CLONE_ARGS_SIZE_VER0].chunks_exact(8))
        {
            *slot = u64::from_ne_bytes(chunk.try_into().unwrap());
        }
        fields
    }

    pub fn flags(&self) -> u64 {
        self.fields()[0]
    }

    /// The child's stack pointer, or 0 when the caller supplied no stack.
    pub fn child_stack_top(&self) -> u64 {
        let fields = self.fields();
        if fields[5] == 0 {
            0
        } else {
            fields[5].wrapping_add(fields[6])
        }
    }

    pub fn set_flags(&mut self, flags: u64) {
        self.raw[0..8].copy_from_slice(&flags.to_ne_bytes());
    }

    /// Zero the `stack` and `stack_size` fields, for a shape forwarded as a
    /// fork: the kernel installs a child stack whatever the flags, and a
    /// fork child running on the guest's few pages would put the runtime's
    /// own frames there.
    pub fn clear_stack(&mut self) {
        self.raw[40..56].fill(0);
    }

    /// The address to forward in place of the guest's own struct. The
    /// declared size is unchanged, so the call's size argument stands.
    pub fn as_ptr(&self) -> u64 {
        self.raw.as_ptr() as u64
    }
}

/// Whether clone flags describe a new thread of the calling process. The
/// kernel demands the full shape — `CLONE_THREAD` requires `CLONE_SIGHAND`,
/// which requires `CLONE_VM` (anything less is `EINVAL`) — so gating the
/// host-thread intercept on all three bits means a malformed thread clone
/// falls through to forwarding and gets the kernel's authoritative error.
pub fn is_thread_clone(flags: u64) -> bool {
    const THREAD_SHAPE: u64 =
        libc::CLONE_THREAD as u64 | libc::CLONE_SIGHAND as u64 | libc::CLONE_VM as u64;
    flags & THREAD_SHAPE == THREAD_SHAPE
}
