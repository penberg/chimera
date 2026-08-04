//! AArch64 backend (Darwin/arm64): the guest register file and translate-execute
//! loop (`dispatch`), the `MAP_JIT` translated-block cache (`cache`), the
//! basic-block translator that emits arm64 from arm64 (`translate`), and the
//! assembly trampolines that cross between Chimera and translated guest code
//! (`trampoline`). Satisfies the backend contract documented in [`crate::arch`].

use std::sync::OnceLock;

use crate::Error;

pub mod cache;
pub mod dispatch;
pub mod trampoline;
pub mod translate;

pub fn init() -> Result<(), Error> {
    static INIT: OnceLock<Result<(), &'static str>> = OnceLock::new();

    match INIT.get_or_init(|| {
        // Probe a `MAP_JIT` mapping: on Apple Silicon a JIT region requires the
        // hardened runtime's `allow-jit` entitlement (or an unhardened process).
        // Without it the translated-code cache can never be created, so fail
        // early here with a clear message rather than at the first translation.
        let page = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                16 * 1024,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_JIT,
                -1,
                0,
            )
        };
        if page == libc::MAP_FAILED {
            Err("host denies MAP_JIT mappings; the JIT code cache is unavailable (missing allow-jit entitlement?)")
        } else {
            unsafe { libc::munmap(page, 16 * 1024) };
            // Sample trace env vars once, before the guest runs, so the hot path
            // never calls getenv and races a libSystem lock the guest holds.
            crate::trace::init();
            Ok(())
        }
    }) {
        Ok(()) => Ok(()),
        Err(msg) => Err(Error::Unsupported((*msg).into())),
    }
}
