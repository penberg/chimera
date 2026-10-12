//! examples/strace: print every system call a guest program makes.
//!
//! Usage: `cargo run --example strace -- <program> [args...]`
//!
//! The handler runs in Chimera's address space, so guest pointers can be
//! dereferenced directly when formatting arguments. Output is written to
//! stderr, with the syscall-name-and-args column padded so return values
//! line up the way `strace(1)` does.
//!
//! The handler implements [`chimera::SystemCalls`] by logging in
//! `post_syscall(&SystemCall)`. The `SystemCall` value carries the number,
//! argument registers, and the final result when one exists.
//!
//! A guest speaks its host kernel's system-call interface, and no two hosts
//! share one: numbers, argument conventions, and struct layouts all differ.
//! Each host therefore has its own tracer — `strace_linux.rs` today — and
//! this file only picks the one for the host it is built on.

use mimalloc::MiMalloc;

/// Route this embedder's allocations through mimalloc's `mmap`-backed
/// segments, keeping the embedder's heap traffic out of the process's `brk`
/// segment. A `#[global_allocator]` is per-binary, so every embedder needs
/// its own.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(target_os = "linux")]
#[path = "strace_linux.rs"]
mod os;

use os::main;
