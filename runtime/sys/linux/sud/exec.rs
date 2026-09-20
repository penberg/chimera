//! `execve`, emulated in place.
//!
//! A forwarded exec would replace the whole process image — and the kernel
//! clears syscall user dispatch across a real `execve`, so the replacement
//! would run unintercepted. Chimera tears the guest image down, loads the new
//! one into a fresh arena, and points the trapped context at its entry so
//! `sigreturn` lands on the new program.

use std::os::fd::AsRawFd;

use crate::{Error, SyscallResult, SystemCall};

use super::{
    super::exec::{ExecRequest, PreparedExec, close_cloexec_fds, exec_errno, prepare_exec},
    clone::report_spawn,
    signal::reset_guest_signals,
    thread::{Thread, aim_context, unwind},
};

/// Emulated `execve`: validate and parse in place (a failure reports
/// `-errno` and resumes the old image untouched), then commit — tear down
/// the old guest, map the new one, and rewrite the trapped context so
/// `sigreturn` resumes at the fresh entry point.
pub fn do_execve(t: &Thread, call: &mut SystemCall, uc: &mut libc::ucontext_t) {
    let prepared = match prepare_exec(call.number, &call.args, &*t.process.handler) {
        Ok(prepared) => prepared,
        Err(err) => {
            let errno = exec_errno(&err).unwrap_or(libc::EIO);
            // Remembered, not reported: `posix_spawnp` walks `$PATH` inside
            // the child, so a failed attempt is routinely followed by one
            // that succeeds. Only the child's exit makes this final.
            t.spawn_exec_errno.set(Some(errno));
            call.set_result(SyscallResult::Error(errno));
            return;
        }
    };
    // A spawn child reaching a loadable image is a successful spawn: unblock
    // the parent now, before the install, so it returns the child's PID while
    // the report pipe is still open — the install's close-on-exec sweep is
    // about to close it.
    report_spawn(t, 0);

    match install_image(t, prepared) {
        Ok((rip, rsp)) => {
            aim_context(&mut uc.uc_mcontext.gregs, rip, rsp);
            call.set_result(SyscallResult::Ok(0))
        }
        // Past teardown there is no image to resume; end the run the way a
        // shell reports an exec that died mid-flight.
        Err(err) => {
            eprintln!("chimera: execve: {err}");
            unwind(t, 127);
        }
    }
}

/// Replace the guest image with a prepared one; returns the entry `rip` and
/// initial `rsp` of the new program.
pub fn install_image(t: &Thread, prepared: PreparedExec) -> Result<(u64, u64), Error> {
    let PreparedExec {
        req,
        parsed,
        parsed_interp,
    } = prepared;
    let ExecRequest {
        path, argv, envp, ..
    } = req;

    // The handler first: a descriptor table's close-on-exec flags live in
    // the table, invisible to the host-fd sweep.
    t.process.handler.on_execve(&path);
    let mut keep = vec![parsed.as_raw_fd()];
    if let Some(interp) = &parsed_interp {
        keep.push(interp.as_raw_fd());
    }
    close_cloexec_fds(&keep)?;

    t.process.arena.teardown();
    let entry = t.process.arena.load_image(
        &parsed,
        parsed_interp.as_ref(),
        &argv,
        &envp,
        path.as_os_str().as_encoded_bytes(),
    )?;

    reset_guest_signals();
    // A fresh image has no TLS yet; hand the handler epilogue a base that at
    // least keeps the host thread coherent until the new libc sets its own.
    t.guest_fs.set(t.runtime_fs);
    Ok(entry)
}
