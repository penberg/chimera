use log::debug;
use std::{
    ffi::{OsStr, OsString},
    sync::mpsc,
};

/// A queued invalidation operation to be flushed by the notify thread.
#[derive(Debug)]
pub enum NotifyOp {
    InvalEntry { parent: u64, name: OsString },
}

/// Queues kernel cache invalidation requests for deferred execution.
///
/// FUSE notification writes to /dev/fuse cannot be issued from the session
/// loop thread — even outside callbacks — because the kernel processes
/// FUSE_NOTIFY_INVAL_ENTRY synchronously within the writev() call. That
/// processing can trigger d_invalidate → iput → FUSE_FORGET, which needs
/// the daemon to be reading /dev/fuse. Since the session loop thread is
/// blocked in writev(), it can't read, causing a deadlock.
///
/// DeferredNotifier solves this by sending operations over an mpsc channel
/// to a dedicated background thread that writes to /dev/fuse independently
/// of the session loop.
#[derive(Debug, Clone)]
pub struct DeferredNotifier {
    tx: mpsc::Sender<NotifyOp>,
}

impl DeferredNotifier {
    pub(crate) fn new(tx: mpsc::Sender<NotifyOp>) -> Self {
        Self { tx }
    }

    pub fn inval_entry(&self, parent: u64, name: &OsStr) {
        if let Err(e) = self.tx.send(NotifyOp::InvalEntry {
            parent,
            name: name.to_os_string(),
        }) {
            debug!("deferred inval_entry send failed (notify thread gone?): {e}");
        }
    }
}
