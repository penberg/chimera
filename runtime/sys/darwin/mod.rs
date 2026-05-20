//! Darwin-specific glue: Mach-O loading, the macOS process bootstrap
//! (`argc`, `argv`, `envp`, `apple` strings), and the entry point that hands
//! the guest off to the dispatcher.

pub mod dyld;
pub mod exec;
pub mod macho;
