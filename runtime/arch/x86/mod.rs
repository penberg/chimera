//! x86-64 backend: the guest register file and translate-execute loop
//! (`dispatch`), the assembly trampolines that cross between Chimera and
//! translated guest code (`trampoline`), the basic-block translator that emits
//! x86-64 from x86-64 (`translate`), and the translated-block cache that links
//! those blocks into one another (`cache`), and the preemption of a thread
//! executing translated code from a host signal catcher (`preempt`).

use std::sync::OnceLock;

use crate::Error;

pub mod cache;
pub mod dispatch;
pub mod preempt;
pub mod trampoline;
pub mod translate;

pub use translate::mpk_enabled;

pub fn init() -> Result<(), Error> {
    static INIT: OnceLock<Result<(), &'static str>> = OnceLock::new();

    match INIT.get_or_init(|| {
        let cpuid = std::arch::x86_64::__cpuid(1);
        if cpuid.ecx & (1 << 27) == 0 {
            return Err(
                "host CPU lacks OSXSAVE; XSAVE-based FP/SIMD context switching unavailable",
            );
        }
        // The exit trampolines save guest FP/SIMD state with XSAVEOPT, whose
        // init/modified optimizations let a syscall that never touched FP/SIMD
        // skip the bulk of the save. It is reported in CPUID.(EAX=0Dh,ECX=1):
        // EAX[0] and has shipped on every x86-64 part since ~2012 — a weaker
        // requirement than the FSGSBASE the trampolines already assume.
        let xstate = std::arch::x86_64::__cpuid_count(0xd, 1);
        if xstate.eax & 1 == 0 {
            return Err("host CPU lacks XSAVEOPT; FP/SIMD context switching unavailable");
        }
        Ok(())
    }) {
        Ok(()) => Ok(()),
        Err(msg) => Err(Error::Unsupported((*msg).into())),
    }
}
