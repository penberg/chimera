//! The Darwin `posix_spawn` emulation. The kernel's posix_spawn is a
//! fork+exec in one trap: forwarding it would start the spawned image
//! natively, outside the sandbox, so Chimera decomposes it — validate and map
//! the image (kernel-style, errors report synchronously before anything
//! forks), fork, apply the spawn attributes and file actions in the child,
//! and commit the prepared image through the same `request_exec`/`drive`
//! machinery as `execve`. A pipe carries a child-side failure back to the
//! parent, and the `FD_CLOEXEC` sweep at install closes its write end, so the
//! parent's EOF marks the moment the exec committed.
//!
//! The argument blob is xnu's `_posix_spawn_args_desc` (a private header, so
//! the layouts here were verified empirically against libc-built blobs): the
//! descriptor is four `(size, pointer)` heads we care about two of, the
//! attribute block leads with `(flags: u16, sigdefault: u32, sigmask: u32,
//! pgroup: i32)`, and file actions are 1040-byte records — `(type: u32,
//! filedes: u32)` then per-type arguments, an open's `(oflag: i32, mode: u16,
//! path[PATH_MAX])` inline.

use crate::{
    SyscallResult,
    arch::dispatch::Thread,
    sys::mmap::{copy_from_guest, copy_to_guest},
};

use super::exec;

// `posix_spawnattr_setflags` flag bits (from `<spawn.h>`).
const POSIX_SPAWN_SETPGROUP: u16 = 0x0002;
const POSIX_SPAWN_SETSIGDEF: u16 = 0x0004;
const POSIX_SPAWN_SETSIGMASK: u16 = 0x0008;
const POSIX_SPAWN_SETEXEC: u16 = 0x0040;
const POSIX_SPAWN_DISABLE_ASLR: u16 = 0x0100;
const POSIX_SPAWN_SETSID: u16 = 0x0400;
const POSIX_SPAWN_CLOEXEC_DEFAULT: u16 = 0x4000;

/// The flags the emulation honors. `DISABLE_ASLR` is accepted and ignored —
/// Chimera slides every image regardless, and the flag only weakens layout
/// randomization for a debugger's benefit. Everything outside the mask —
/// `RESETIDS` (0x1) and `START_SUSPENDED` (0x80) among them — is refused:
/// silently skipping a setuid reset or a suspended start would change
/// guest-visible semantics.
const SUPPORTED_FLAGS: u16 = POSIX_SPAWN_SETPGROUP
    | POSIX_SPAWN_SETSIGDEF
    | POSIX_SPAWN_SETSIGMASK
    | POSIX_SPAWN_SETEXEC
    | POSIX_SPAWN_DISABLE_ASLR
    | POSIX_SPAWN_SETSID
    | POSIX_SPAWN_CLOEXEC_DEFAULT;

// File-action record types (xnu `psfa_t`).
const PSFA_OPEN: u32 = 0;
const PSFA_CLOSE: u32 = 1;
const PSFA_DUP2: u32 = 2;
const PSFA_INHERIT: u32 = 3;
const PSFA_CHDIR: u32 = 5;
const PSFA_FCHDIR: u32 = 6;

/// One file-action record: 8 bytes of `(type, filedes)` then the union, whose
/// largest member (an open's inline `path[PATH_MAX]` at offset 6) pads the
/// whole record to 1040 bytes.
const ACTION_SIZE: usize = 1040;
const ACTIONS_HEADER_SIZE: usize = 8;
/// Sanity bound on the record count; the real bound is the descriptor's own
/// `file_actions_size`, checked against this many whole records.
const MAX_ACTIONS: usize = 4096;

enum FileAction {
    Open {
        fd: i32,
        path: Vec<u8>,
        oflag: i32,
        mode: u16,
    },
    Close {
        fd: i32,
    },
    Dup2 {
        src: i32,
        dst: i32,
    },
    Inherit {
        fd: i32,
    },
    Chdir {
        path: Vec<u8>,
    },
    Fchdir {
        fd: i32,
    },
}

struct SpawnRequest {
    flags: u16,
    sigdefault: u32,
    sigmask: u32,
    pgroup: i32,
    actions: Vec<FileAction>,
}

/// Service the `posix_spawn` trap: `args` are `(pid *, path, desc, argv,
/// envp)`. On success the child pid has been written through `args[0]` (or
/// the caller has become the new image, under `SETEXEC`); the error is the
/// positive errno the trap reports, which libc's wrapper hands back as
/// `posix_spawn`'s return value.
pub fn spawn(thread: &mut Thread, args: &[u64; 8]) -> Result<(), i32> {
    let req = read_request(args[2])?;
    if req.flags & !SUPPORTED_FLAGS != 0 {
        return Err(libc::ENOTSUP);
    }

    // Validate and map the image now, in the calling process, so a missing or
    // unloadable program fails the call synchronously — the kernel's
    // posix_spawn reports exec errors the same way. The exec request reuses
    // the execve argument shape `(path, argv, envp)`.
    let exec_args = [args[1], args[3], args[4], 0, 0, 0, 0, 0];
    let prepared =
        exec::prepare_exec(&exec_args).map_err(|e| exec::exec_errno(&e).unwrap_or(libc::EIO))?;

    // SETEXEC: no child — this process becomes the new image, an execve with
    // spawn attributes. Apply them in place and commit.
    if req.flags & POSIX_SPAWN_SETEXEC != 0 {
        apply_attrs(thread, &req);
        apply_file_actions(&req.actions)?;
        if req.flags & POSIX_SPAWN_CLOEXEC_DEFAULT != 0 {
            close_all_except(&keep_fds(&req.actions, None));
        }
        if let Err(loser) = thread.process().request_exec(prepared, &thread.state) {
            loser.discard();
        }
        return Ok(());
    }

    // The failure pipe: the child reports a post-fork errno through it, and
    // the install's FD_CLOEXEC sweep closes the write end on success, so the
    // parent blocks until the spawn's outcome is decided — the synchronous
    // reporting posix_spawn promises. The write end moves clear of the low
    // descriptors the file actions typically remap.
    let mut fds = [0i32; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        prepared.discard();
        return Err(unsafe { *libc::__error() });
    }
    let read_fd = fds[0];
    let mut write_fd = fds[1];
    unsafe {
        libc::fcntl(read_fd, libc::F_SETFD, libc::FD_CLOEXEC);
        let moved = libc::fcntl(write_fd, libc::F_DUPFD_CLOEXEC, 100);
        if moved >= 0 {
            libc::close(write_fd);
            write_fd = moved;
        }
    }

    let (result, is_child) = forked_native(thread);
    let pid = match result {
        SyscallResult::Ok(pid) => pid,
        SyscallResult::Error(errno) => {
            unsafe {
                libc::close(read_fd);
                libc::close(write_fd);
            }
            prepared.discard();
            return Err(errno);
        }
    };

    if is_child != 0 {
        unsafe { libc::close(read_fd) };
        apply_attrs(thread, &req);
        if let Err(errno) = apply_file_actions(&req.actions) {
            return child_failed(thread, write_fd, errno);
        }
        if req.flags & POSIX_SPAWN_CLOEXEC_DEFAULT != 0 {
            close_all_except(&keep_fds(&req.actions, Some(write_fd)));
        }
        if let Err(loser) = thread.process().request_exec(prepared, &thread.state) {
            loser.discard();
        }
        // The pending exec stops this (only) thread at its next boundary and
        // the drive loop installs the image; its sweep closes `write_fd`,
        // which is the parent's success signal.
        return Ok(());
    }

    // Parent: this process keeps running the old image, so its copy of the
    // prepared mapping (the child inherited its own) must not leak.
    prepared.discard();
    unsafe { libc::close(write_fd) };
    let mut buf = [0u8; 4];
    let n = unsafe { libc::read(read_fd, buf.as_mut_ptr().cast(), buf.len()) };
    unsafe { libc::close(read_fd) };
    if n == buf.len() as isize {
        let errno = i32::from_ne_bytes(buf);
        if errno != 0 {
            // The child reported a post-fork failure and exits 127; reap it
            // so it leaves no zombie (the caller gets no PID to wait on).
            unsafe { libc::waitpid(pid as libc::pid_t, std::ptr::null_mut(), 0) };
            return Err(errno);
        }
    }
    if args[0] != 0 {
        copy_to_guest(args[0], &(pid as i32).to_ne_bytes());
    }
    Ok(())
}

/// Forward the fork trap under the `Process` fork-lock discipline and rebuild
/// the child's bookkeeping, exactly as the policy's `fork` arm does. Returns
/// the trap result and the kernel's is-child flag. The caller is responsible
/// for libSystem's atfork discipline: a guest `fork(3)`'s own wrapper
/// brackets the trap translated, so bracketing again here would deadlock on
/// the (non-recursive) libSystem locks.
pub fn forked(thread: &mut Thread) -> (SyscallResult, u64) {
    // The guest's atfork handlers live on the runtime's list (see
    // `sys::darwin::GUEST_ATFORK`), so its own translated wrapper never sees
    // them; they are owed here, translated, in POSIX order — prepare in
    // reverse registration order, parent/child in registration order. A
    // handler's error is swallowed the way a native handler's misbehaviour
    // would be: atfork has no failure channel.
    let handlers = crate::sys::darwin::guest_atfork_handlers();
    for h in handlers.iter().rev() {
        if h.prepare != 0 {
            let _ = thread.run_guest_call(h.prepare, 0);
        }
    }
    let call = crate::SystemCall::new(2, [0; 8]);
    let fork_locks = thread.process().lock_for_fork();
    let (result, is_child) = super::syscall::host_syscall2(&call);
    drop(fork_locks);
    if matches!(result, SyscallResult::Ok(_)) && is_child != 0 {
        thread.signals_mut().reset_pending_after_fork();
        thread.reset_after_fork();
        for h in &handlers {
            if h.child != 0 {
                let _ = thread.run_guest_call(h.child, 0);
            }
        }
    } else {
        // fork returned in the parent — on failure too, which is when POSIX
        // still owes the parent handlers.
        for h in &handlers {
            if h.parent != 0 {
                let _ = thread.run_guest_call(h.parent, 0);
            }
        }
    }
    (result, is_child)
}

/// [`forked`]'s analogue for the fork this emulation issues itself, where no
/// guest libc wrapper is involved: the host's own `fork(3)` wrapper runs
/// libSystem's atfork triple natively, repairing the child's cached Mach
/// state — the per-thread MIG reply port among it — that a raw trap leaves
/// stale (the first `copy_from_guest` in such a child fails with a phantom
/// `EFAULT`). The guest-facing fork shapes must keep the raw trap of
/// [`forked`]: their wrappers run the same handlers translated, and running
/// them twice deadlocks libSystem's non-recursive locks.
fn forked_native(thread: &mut Thread) -> (SyscallResult, u64) {
    let fork_locks = thread.process().lock_for_fork();
    let pid = unsafe { libc::fork() };
    drop(fork_locks);
    match pid {
        -1 => (SyscallResult::Error(unsafe { *libc::__error() }), 0),
        0 => {
            thread.signals_mut().reset_pending_after_fork();
            thread.reset_after_fork();
            (SyscallResult::Ok(0), 1)
        }
        pid => (SyscallResult::Ok(pid as i64), 0),
    }
}

/// A post-fork failure in the child: report the errno through the pipe and
/// end the child with status 127, posix_spawn's convention for a child that
/// failed between fork and exec. The parent reaps it.
fn child_failed(thread: &mut Thread, write_fd: i32, errno: i32) -> Result<(), i32> {
    unsafe {
        let bytes = errno.to_ne_bytes();
        libc::write(write_fd, bytes.as_ptr().cast(), bytes.len());
        libc::close(write_fd);
    }
    thread.process().request_exit_group(127, &thread.state);
    thread.exit_code = 127;
    thread.running = false;
    Ok(())
}

/// Apply the spawn attributes: session/process-group placement on the host
/// (guest and host process identities coincide), signal state on the guest's
/// virtualized table.
fn apply_attrs(thread: &mut Thread, req: &SpawnRequest) {
    unsafe {
        if req.flags & POSIX_SPAWN_SETSID != 0 {
            libc::setsid();
        }
        if req.flags & POSIX_SPAWN_SETPGROUP != 0 {
            libc::setpgid(0, req.pgroup);
        }
    }
    if req.flags & POSIX_SPAWN_SETSIGDEF != 0 {
        thread.signals_mut().apply_spawn_sigdefault(req.sigdefault);
    }
    if req.flags & POSIX_SPAWN_SETSIGMASK != 0 {
        thread.signals_mut().apply_spawn_sigmask(req.sigmask);
    }
}

/// Run the file actions in order against the host descriptor table (guest and
/// host fds coincide). A failing action aborts the spawn with its errno, the
/// way the kernel's `exec_handle_file_actions` does.
fn apply_file_actions(actions: &[FileAction]) -> Result<(), i32> {
    let err = || -> i32 { unsafe { *libc::__error() } };
    for action in actions {
        match action {
            FileAction::Open {
                fd,
                path,
                oflag,
                mode,
            } => {
                let mut p = path.clone();
                p.push(0);
                let opened =
                    unsafe { libc::open(p.as_ptr().cast(), *oflag, *mode as libc::c_uint) };
                if opened < 0 {
                    return Err(err());
                }
                if opened != *fd {
                    if unsafe { libc::dup2(opened, *fd) } < 0 {
                        let e = err();
                        unsafe { libc::close(opened) };
                        return Err(e);
                    }
                    unsafe { libc::close(opened) };
                }
            }
            FileAction::Close { fd } => {
                if unsafe { libc::close(*fd) } != 0 {
                    return Err(err());
                }
            }
            FileAction::Dup2 { src, dst } => {
                if unsafe { libc::dup2(*src, *dst) } < 0 {
                    return Err(err());
                }
            }
            FileAction::Inherit { .. } => {}
            FileAction::Chdir { path } => {
                let mut p = path.clone();
                p.push(0);
                if unsafe { libc::chdir(p.as_ptr().cast()) } != 0 {
                    return Err(err());
                }
            }
            FileAction::Fchdir { fd } => {
                if unsafe { libc::fchdir(*fd) } != 0 {
                    return Err(err());
                }
            }
        }
    }
    Ok(())
}

/// The descriptors `CLOEXEC_DEFAULT` leaves open: every fd a file action
/// created or named as surviving (an open's target, a dup2's destination, an
/// inherit), plus the runtime's failure-pipe write end, whose close is the
/// parent's success signal and belongs to the install sweep.
fn keep_fds(actions: &[FileAction], report_fd: Option<i32>) -> Vec<i32> {
    let mut keep: Vec<i32> = actions
        .iter()
        .filter_map(|a| match a {
            FileAction::Open { fd, .. } => Some(*fd),
            FileAction::Dup2 { dst, .. } => Some(*dst),
            FileAction::Inherit { fd } => Some(*fd),
            _ => None,
        })
        .collect();
    keep.extend(report_fd);
    keep
}

/// `POSIX_SPAWN_CLOEXEC_DEFAULT`: every descriptor not explicitly kept by a
/// file action behaves as close-on-exec, the standard streams included.
fn close_all_except(keep: &[i32]) {
    let Ok(entries) = std::fs::read_dir("/dev/fd") else {
        return;
    };
    let fds: Vec<i32> = entries
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
        .collect();
    for fd in fds {
        if keep.contains(&fd) {
            continue;
        }
        // Skip fds the walk itself no longer holds; a close of an
        // already-closed fd is harmless here.
        if unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0 {
            unsafe { libc::close(fd) };
        }
    }
}

/// Decode the trap's `_posix_spawn_args_desc` out of guest memory. A null
/// descriptor means no attributes and no file actions. Only the attribute and
/// file-action heads are honored; the later `(size, pointer)` pairs (port
/// actions, MAC extensions, coalition/persona info) carry Mach and policy
/// baggage with no analogue under Chimera and are ignored.
fn read_request(desc: u64) -> Result<SpawnRequest, i32> {
    let mut req = SpawnRequest {
        flags: 0,
        sigdefault: 0,
        sigmask: 0,
        pgroup: 0,
        actions: Vec::new(),
    };
    if desc == 0 {
        return Ok(req);
    }
    let mut head = [0u8; 32];
    if !copy_from_guest(desc, &mut head) {
        return Err(libc::EFAULT);
    }
    let attr_size = u64::from_ne_bytes(head[0..8].try_into().unwrap());
    let attrp = u64::from_ne_bytes(head[8..16].try_into().unwrap());
    let fa_size = u64::from_ne_bytes(head[16..24].try_into().unwrap());
    let fa_ptr = u64::from_ne_bytes(head[24..32].try_into().unwrap());

    if attrp != 0 && attr_size >= 16 {
        let mut attr = [0u8; 16];
        if !copy_from_guest(attrp, &mut attr) {
            return Err(libc::EFAULT);
        }
        req.flags = u16::from_ne_bytes(attr[0..2].try_into().unwrap());
        req.sigdefault = u32::from_ne_bytes(attr[4..8].try_into().unwrap());
        req.sigmask = u32::from_ne_bytes(attr[8..12].try_into().unwrap());
        req.pgroup = i32::from_ne_bytes(attr[12..16].try_into().unwrap());
    }

    if fa_ptr != 0 && fa_size >= ACTIONS_HEADER_SIZE as u64 {
        let mut header = [0u8; ACTIONS_HEADER_SIZE];
        if !copy_from_guest(fa_ptr, &mut header) {
            return Err(libc::EFAULT);
        }
        let count = i32::from_ne_bytes(header[4..8].try_into().unwrap());
        if count < 0
            || count as usize > MAX_ACTIONS
            || (ACTIONS_HEADER_SIZE + count as usize * ACTION_SIZE) as u64 > fa_size
        {
            return Err(libc::EINVAL);
        }
        for i in 0..count as usize {
            let mut record = [0u8; ACTION_SIZE];
            let addr = fa_ptr + (ACTIONS_HEADER_SIZE + i * ACTION_SIZE) as u64;
            if !copy_from_guest(addr, &mut record) {
                return Err(libc::EFAULT);
            }
            req.actions.push(parse_action(&record)?);
        }
    }
    Ok(req)
}

/// Read a NUL-terminated path out of an action record's inline `PATH_MAX`
/// array.
fn record_path(record: &[u8], at: usize) -> Result<Vec<u8>, i32> {
    let field = &record[at..];
    let Some(nul) = field.iter().position(|&b| b == 0) else {
        return Err(libc::ENAMETOOLONG);
    };
    Ok(field[..nul].to_vec())
}

fn parse_action(record: &[u8; ACTION_SIZE]) -> Result<FileAction, i32> {
    let ty = u32::from_ne_bytes(record[0..4].try_into().unwrap());
    let fd = i32::from_ne_bytes(record[4..8].try_into().unwrap());
    Ok(match ty {
        PSFA_OPEN => FileAction::Open {
            fd,
            oflag: i32::from_ne_bytes(record[8..12].try_into().unwrap()),
            mode: u16::from_ne_bytes(record[12..14].try_into().unwrap()),
            path: record_path(record, 14)?,
        },
        PSFA_CLOSE => FileAction::Close { fd },
        PSFA_DUP2 => FileAction::Dup2 {
            src: fd,
            dst: i32::from_ne_bytes(record[8..12].try_into().unwrap()),
        },
        PSFA_INHERIT => FileAction::Inherit { fd },
        PSFA_CHDIR => FileAction::Chdir {
            path: record_path(record, 8)?,
        },
        PSFA_FCHDIR => FileAction::Fchdir { fd },
        // PSFA_FILEPORT_DUP2 and anything newer: a Mach right in the middle
        // of the file-action stream cannot be emulated faithfully; refuse
        // visibly rather than mis-spawn.
        _ => return Err(libc::ENOTSUP),
    })
}
