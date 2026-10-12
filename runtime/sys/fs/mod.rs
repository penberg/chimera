//! The userspace filesystem layer, shared by every host: the [`vfs`] traits a
//! filesystem implements, the [`namespace`] that resolves a guest path to the
//! filesystem serving it, and the copy-on-write [`overlayfs`] and the on-disk
//! [`delta`] format it writes.
//!
//! Nothing here is specific to one kernel. What differs per host — how a
//! directory is confined, how an xattr is spelled, how an atomic rename is
//! requested — lives behind two seams: `crate::sys::hostfs` (the passthrough
//! filesystem) and `crate::sys::fsops` (the handful of primitives the delta
//! format needs).

pub mod delta;
pub mod namespace;
pub mod overlayfs;
pub mod vfs;
