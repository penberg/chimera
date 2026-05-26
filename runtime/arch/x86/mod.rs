//! x86-64 backend: the guest register file and translate-execute loop
//! (`dispatch`), the assembly trampolines that cross between Chimera and
//! translated guest code (`trampoline`), and the basic-block translator
//! that emits x86-64 from x86-64.

use std::sync::OnceLock;

use crate::Error;

pub mod dispatch;
pub mod trampoline;
pub mod translate;

pub fn init() -> Result<(), Error> {
    static INIT: OnceLock<Result<(), &'static str>> = OnceLock::new();

    match INIT.get_or_init(|| {
        let cpuid = std::arch::x86_64::__cpuid(1);
        if cpuid.ecx & (1 << 27) == 0 {
            Err("host CPU lacks OSXSAVE; XSAVE-based FP/SIMD context switching unavailable")
        } else {
            Ok(())
        }
    }) {
        Ok(()) => Ok(()),
        Err(msg) => Err(Error::Unsupported((*msg).into())),
    }
}
