//! The userspace filesystem layer, shared by every host: the [`vfs`] traits a
//! filesystem implements, the [`namespace`] that resolves a guest path to the
//! filesystem serving it, the copy-on-write [`overlayfs`] and the on-disk
//! [`delta`] format it writes, and the [`personality`] that drives them from
//! a guest's syscalls.
//!
//! Nothing here is specific to one kernel. What differs per host — how a
//! directory is confined, how an xattr is spelled, how an atomic rename is
//! requested, which syscall numbers and struct layouts a guest speaks — lives
//! behind two seams: `crate::sys::hostfs` (the passthrough filesystem) and
//! `crate::sys::fsops` (the handful of primitives the delta format needs),
//! plus one syscall decoder per host under [`personality`].

pub mod delta;
pub mod namespace;
pub mod overlayfs;
pub mod personality;
pub mod vfs;
