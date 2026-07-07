//! Linux-specific glue: ELF loading, the Linux process bootstrap (auxv,
//! initial stack), and the entry point that hands the guest off to the
//! dispatcher.

mod delta;
pub mod elf;
pub mod exec;
pub mod fault;
mod hostfs;
mod namespace;
mod personality;
pub mod signal;
pub mod syscall;
mod vfs;

pub use hostfs::HostFs;
pub use namespace::{MountFlags, Namespace};
pub use personality::Personality;
pub use vfs::{
    DirEntry, Errno, File, FileType, Mode, OpenFlags, RenameFlags, Stat, StatFs, Timespec, Vfs,
    WriteResult,
};
