//! Linux-specific glue: ELF loading, the Linux process bootstrap (auxv,
//! initial stack), and the entry point that hands the guest off to the
//! dispatcher.

pub mod elf;
pub mod exec;
pub mod fault;
pub mod signal;
pub mod syscall;
mod vfs;

pub use vfs::{
    DirEntry, Errno, File, FileType, Mode, OpenFlags, RenameFlags, Stat, StatFs, Timespec, Vfs,
    WriteResult,
};
