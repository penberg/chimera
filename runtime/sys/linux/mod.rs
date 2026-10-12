//! Linux-specific glue: ELF loading, the Linux process bootstrap (auxv,
//! initial stack), and the entry point that hands the guest off to the
//! dispatcher.

pub mod elf;
pub mod exec;
pub mod fault;
pub mod fsops;
pub mod hostfs;
mod personality;
pub mod policy;
pub mod signal;
pub mod syscall;
pub mod thread;

pub use personality::Personality;
