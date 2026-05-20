//! AArch64 backend: the guest register file and translate-execute loop
//! (`dispatch`), the assembly trampolines that cross between Chimera and
//! translated guest code (`trampoline`), and the basic-block translator
//! that emits AArch64 from AArch64.

pub mod dispatch;
pub mod trampoline;
pub mod translate;
