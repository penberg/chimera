//! Linux-specific glue: ELF loading, the Linux process bootstrap (auxv,
//! initial stack), and the entry point that hands the guest off to the
//! dispatcher.

pub mod elf;
pub mod exec;
