//! Guest runtime entry: build the initial stack, hand control to the
//! dispatcher, and service `execve` by loading the new image and re-entering.

use std::{
    ffi::{OsStr, OsString},
    fs,
    os::{
        fd::{AsRawFd, RawFd},
        unix::ffi::OsStrExt,
    },
    path::{Path, PathBuf},
    ptr,
    sync::Arc,
};

use crate::{
    Error, SystemCalls,
    arch::dispatch::{self, ExitReason},
    process::Process,
    sys::mmap::{AddressSpace, copy_from_guest},
};

use super::{
    elf::{LoadedElf, PAGE_SIZE, ParsedElf, load_elf, map_elf, parse_elf},
    signal::HostMaskGuard,
};

const STACK_SIZE: usize = 8 * 1024 * 1024;

/// `execveat` special dirfd and flag values (from `<fcntl.h>`).
const AT_FDCWD: i32 = -100;
const AT_EMPTY_PATH: i32 = 0x1000;

pub fn execv(
    program: &Path,
    arg0: Option<&OsStr>,
    args: &[OsString],
    envs: Option<&[(OsString, OsString)]>,
    handler: Box<dyn SystemCalls>,
    code_cache_size: usize,
) -> Result<i32, Error> {
    // The first image's argv and envp come from the embedder: argv[0] is the
    // guest-visible program name (the program path unless overridden), then
    // the supplied args; the environment is the explicit set if one was
    // given, otherwise the host's.
    let mut argv: Vec<Vec<u8>> = Vec::with_capacity(args.len() + 1);
    argv.push(arg0.unwrap_or(program.as_os_str()).as_bytes().to_vec());
    for a in args {
        argv.push(a.as_bytes().to_vec());
    }
    let envp: Vec<Vec<u8>> = match envs {
        Some(over) => over.iter().map(|(k, v)| env_pair(k, v)).collect(),
        None => std::env::vars_os().map(|(k, v)| env_pair(&k, &v)).collect(),
    };

    let main = load_elf(program)?;
    let (rip, interp_base, interp) = match &main.interp {
        Some(interp_path) => {
            let interp = load_elf(&resolve_image_path(interp_path, handler.as_ref())?)?;
            (interp.entry, interp.base, Some(interp))
        }
        None => (main.entry, 0, None),
    };
    // `AT_EXECFN` is guest-visible the same way argv[0] is; both carry the
    // name the guest knows itself by, not the backing file the loader read.
    let (rsp, stack_start, stack_len) = build_stack(
        &argv,
        &envp,
        arg0.unwrap_or(program.as_os_str()).as_bytes(),
        &main,
        interp_base,
    )?;

    // The process's shared state: the address space, code cache, and the
    // embedder's syscall handler. The first thread is created against it; future
    // `clone(CLONE_VM)` siblings will share clones of this `Arc`.
    let process = Arc::new(Process::new(handler, code_cache_size)?);
    // Publish the process for the synchronous fault handler before any guest
    // code runs, so a self-modifying-code write trap can reach the address
    // space.
    super::fault::set_process(&process);
    let mut thread = dispatch::Thread::new(process, rip, rsp)?;
    let _host_mask = HostMaskGuard::save();
    record_regions(
        &mut thread.addr_space(),
        &main,
        interp.as_ref(),
        stack_start,
        stack_len,
    );

    let reason = thread.run()?;
    drive(&mut thread, reason)
}

/// Drive a main thread to process termination: install each committed
/// execve's published image and re-enter the run, until the guest exits;
/// returns the process exit status. Two callers: [`execv`] above, with the
/// initial thread's first run reason, and a `clone` child promoted to main
/// by a fork in its thread (see `Thread::reset_after_fork`) — that child's
/// host thread is all its new process has, so it must drive itself.
pub fn drive(thread: &mut dispatch::Thread, mut reason: ExitReason) -> Result<i32, Error> {
    loop {
        match reason {
            ExitReason::Exited(code) => return Ok(code),
            ExitReason::Execve => {
                // A committed execve: the calling thread — main or a clone
                // sibling — already validated and parsed the image and
                // published it on the shared process (a failed one took
                // `-errno` in place and never left the run). Linux's
                // `de_thread` kills every other thread before installing a
                // new image; the commit requested that stop, so wait until
                // the last sibling has drained out of the thread list before
                // tearing down mappings a straggler could still be
                // translating or executing.
                thread.process().wait_exec_quiesce();
                let PreparedExec {
                    req,
                    parsed,
                    parsed_interp,
                } = thread
                    .process()
                    .take_exec_request()
                    .expect("an Execve exit reason always has a published image");

                let mut keep = vec![parsed.as_raw_fd()];
                if let Some(interp) = &parsed_interp {
                    keep.push(interp.as_raw_fd());
                }
                // The handler first: a descriptor table's close-on-exec flags
                // live in the table, invisible to the host-fd sweep below.
                // Its closed fds simply fail the sweep's F_GETFD and are
                // skipped.
                thread.process().handler.on_execve(&req.path);
                close_cloexec_fds(&keep)?;

                // Tear down the old image (unmapping its regions, so a fixed
                // ET_EXEC can reuse its addresses), then map the new one.
                thread.addr_space().reset();
                let main = map_elf(&parsed)?;
                let (rip, interp_base, interp) = match &parsed_interp {
                    Some(parsed_interp) => {
                        let interp = map_elf(parsed_interp)?;
                        (interp.entry, interp.base, Some(interp))
                    }
                    None => (main.entry, 0, None),
                };
                let (rsp, stack_start, stack_len) = build_stack(
                    &req.argv,
                    &req.envp,
                    req.path.as_os_str().as_bytes(),
                    &main,
                    interp_base,
                )?;
                record_regions(
                    &mut thread.addr_space(),
                    &main,
                    interp.as_ref(),
                    stack_start,
                    stack_len,
                );
                thread.enter(rip, rsp);
                // POSIX `execve` resets caught signals to their default
                // disposition (ignored stay ignored) and drops the alt stack.
                thread.signals_mut().on_execve();
            }
        }
        reason = thread.run()?;
    }
}

/// Close every fd flagged close-on-exec, the way the kernel does when an exec
/// commits. Chimera services `execve` by re-entering in place — no host exec
/// runs — so `FD_CLOEXEC` must be applied by hand: the flag sits on the host
/// fd (the guest's `O_CLOEXEC` and `F_SETFD` pass straight through), but the
/// kernel honors it only at a real `execve`. Left open, a close-on-exec fd
/// leaks into the new image — git's exec-failure notify pipe, for one, whose
/// parent then waits forever for the EOF that close-on-exec was meant to
/// deliver. `keep` names the runtime's own fds that must survive the install
/// (the replacement image's files, still to be mapped); any runtime fd held
/// across an exec must be listed there, since Rust opens files `O_CLOEXEC`.
fn close_cloexec_fds(keep: &[RawFd]) -> Result<(), Error> {
    let entries = fs::read_dir("/proc/self/fd")
        .map_err(|e| Error::io("execve: listing /proc/self/fd".to_string(), e))?;
    // Collect before closing: the directory walk holds an fd of its own, and
    // closing entries out from under it would corrupt the walk. By the time
    // the sweep runs the iterator has been dropped, so its fd fails the
    // `F_GETFD` below and is skipped.
    let fds: Vec<RawFd> = entries
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
        .collect();
    for fd in fds {
        if keep.contains(&fd) {
            continue;
        }
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags >= 0 && flags & libc::FD_CLOEXEC != 0 {
            unsafe { libc::close(fd) };
        }
    }
    Ok(())
}

/// A validated, parsed `execve` replacement image: everything needed to
/// install it without touching the requesting thread again. Owned data
/// throughout, so it can be published on the shared `Process` by whichever
/// thread's `execve` committed and consumed later on the main host thread.
pub struct PreparedExec {
    req: ExecRequest,
    parsed: ParsedElf,
    parsed_interp: Option<ParsedElf>,
}

/// Read an `execve`/`execveat` request out of guest memory and parse the
/// named image and its interpreter, without mapping anything. Runs in the
/// calling thread, the way the kernel sequences an exec: an error here means
/// the caller takes `-errno` (see [`exec_errno`]) and resumes, with no other
/// thread disturbed.
pub fn prepare_exec(
    number: u64,
    args: &[u64; 6],
    handler: &dyn SystemCalls,
) -> Result<PreparedExec, Error> {
    let req = read_request(number, args, handler)?;
    let parsed = parse_elf(&req.path)?;
    let parsed_interp = match &parsed.interp {
        Some(interp_path) => Some(parse_elf(&resolve_image_path(interp_path, handler)?)?),
        None => None,
    };
    Ok(PreparedExec {
        req,
        parsed,
        parsed_interp,
    })
}

/// The host file backing a guest-visible image path. `PT_INTERP` names the
/// dynamic interpreter by its guest path, and only the handler's namespace
/// can say which host file serves it — under an image-rooted filesystem the
/// guest's `/lib64/ld-linux-x86-64.so.2` is the image's, not the host's, and
/// loading the host's mixes two libcs into one process. A handler with no
/// resolver ([`Passthrough`]) reads the path as-is.
fn resolve_image_path(path: &Path, handler: &dyn SystemCalls) -> Result<PathBuf, Error> {
    match handler.resolve_exec(AT_FDCWD, path.as_os_str().as_bytes(), 0) {
        Some(Ok(host)) => Ok(host),
        Some(Err(errno)) => Err(Error::io(
            format!("interpreter {}", path.display()),
            std::io::Error::from_raw_os_error(errno),
        )),
        None => Ok(path.to_path_buf()),
    }
}

/// The errno a failed [`prepare_exec`] reports to the guest. `None` for
/// runtime-fatal errors that are not a guest-visible exec failure.
pub fn exec_errno(err: &Error) -> Option<i32> {
    match err {
        Error::Io { source, .. } => Some(source.raw_os_error().unwrap_or(libc::EIO)),
        Error::BadBinary(_) | Error::Link(_) | Error::Unsupported(_) => Some(libc::ENOEXEC),
        Error::BadAccess(_) => Some(libc::EFAULT),
        Error::CodeCacheExhausted | Error::Translate(_) => None,
    }
}

/// Format a `KEY=VALUE` environment entry (no trailing NUL; `build_stack` adds
/// it).
fn env_pair(key: &OsStr, value: &OsStr) -> Vec<u8> {
    let mut s = key.as_bytes().to_vec();
    s.push(b'=');
    s.extend_from_slice(value.as_bytes());
    s
}

/// Record an image's mappings and its stack as regions of the address space,
/// so they are unmapped when the address space is torn down.
fn record_regions(
    addr_space: &mut AddressSpace,
    main: &LoadedElf,
    interp: Option<&LoadedElf>,
    stack_start: usize,
    stack_len: usize,
) {
    addr_space.set_program_break(current_program_break());
    for &(start, len) in &main.regions {
        addr_space.add_region(start as usize, len as usize);
    }
    if let Some(interp) = interp {
        for &(start, len) in &interp.regions {
            addr_space.add_region(start as usize, len as usize);
        }
    }
    addr_space.add_region(stack_start, stack_len);
}

fn current_program_break() -> usize {
    // Guest `brk` is forwarded to the host kernel, so guest-region bookkeeping
    // has to start from the process's live break rather than the ELF image's
    // nominal end-of-data address.
    unsafe { libc::syscall(libc::SYS_brk, 0) as usize }
}

/// A decoded `execve`/`execveat` request, copied out of guest memory.
struct ExecRequest {
    path: PathBuf,
    argv: Vec<Vec<u8>>,
    envp: Vec<Vec<u8>>,
}

/// Read an `execve`/`execveat` request out of guest memory. Chimera and the
/// guest share an address space, but none of the guest's pointers are
/// trusted: every read is a fault-safe kernel copy, bounded the way the
/// kernel bounds it, so a bad pathname, argv, or envp pointer reports
/// `EFAULT` to the caller — and an oversized string `ENAMETOOLONG`/`E2BIG` —
/// instead of crashing the runtime. The target pathname is resolved through
/// the handler's namespace (`resolve_exec`) when it has one, so a confined
/// guest's exec stays inside its root; otherwise the runtime's own
/// `/proc/self/fd` handling stands in for [`Passthrough`].
fn read_request(
    number: u64,
    args: &[u64; 6],
    handler: &dyn SystemCalls,
) -> Result<ExecRequest, Error> {
    let (dirfd, rawptr, flags, argv_ptr, envp_ptr) = if number == libc::SYS_execveat as u64 {
        (args[0] as i32, args[1], args[4] as i32, args[2], args[3])
    } else {
        (AT_FDCWD, args[0], 0, args[1], args[2])
    };
    let raw = read_guest_cstr(rawptr, libc::PATH_MAX as usize, libc::ENAMETOOLONG)?;
    let path = match handler.resolve_exec(dirfd, &raw, flags) {
        Some(Ok(path)) => path,
        Some(Err(errno)) => {
            return Err(Error::io(
                "execve",
                std::io::Error::from_raw_os_error(errno),
            ));
        }
        None => resolve_at(dirfd, &raw, flags, handler),
    };
    Ok(ExecRequest {
        path,
        argv: read_guest_ptr_array(argv_ptr)?,
        envp: read_guest_ptr_array(envp_ptr)?,
    })
}

/// The kernel's limit for a single argv/envp string (`MAX_ARG_STRLEN`,
/// 32 pages); a longer one fails the exec with `E2BIG`.
const MAX_ARG_STRLEN: usize = 32 * PAGE_SIZE as usize;

/// Cap on argv/envp entry counts. The kernel bounds the whole argument block
/// by the stack limit (`ARG_MAX` = stack rlimit / 4), and every entry costs
/// at least a stack pointer, so more entries than `ARG_MAX / 8` could never
/// fit on the guest stack anyway; reject them with `E2BIG` up front.
const MAX_ARG_COUNT: usize = STACK_SIZE / 4 / 8;

/// A guest-memory read failed or overran its bound while decoding an exec
/// request. Carried as an [`Error::Io`] so [`exec_errno`] hands the errno to
/// the caller exactly as the kernel's user-copy path would.
fn arg_error(errno: i32, what: &str) -> Error {
    Error::io(
        format!("execve: {what}"),
        std::io::Error::from_raw_os_error(errno),
    )
}

/// Copy a NUL-terminated string out of guest memory without trusting the
/// pointer (see `copy_from_guest`), in chunks that stop at page boundaries so
/// a string ending just before an unmapped page is not failed by over-reading
/// past it. `cap` is the kernel's limit for this kind of string: filling it
/// without a NUL fails with `toolong` (`ENAMETOOLONG` for a pathname, `E2BIG`
/// for an argv/envp entry), and an unreadable byte fails with `EFAULT`, the
/// way the kernel's user-copy would.
fn read_guest_cstr(ptr: u64, cap: usize, toolong: i32) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    let mut addr = ptr;
    while out.len() < cap {
        let page_left = (PAGE_SIZE - (addr % PAGE_SIZE)) as usize;
        let mut chunk = vec![0u8; page_left.min(cap - out.len())];
        if !copy_from_guest(addr, &mut chunk) {
            return Err(arg_error(libc::EFAULT, "unreadable string"));
        }
        if let Some(nul) = chunk.iter().position(|&b| b == 0) {
            chunk.truncate(nul);
            out.extend_from_slice(&chunk);
            return Ok(out);
        }
        addr += chunk.len() as u64;
        out.extend_from_slice(&chunk);
    }
    Err(arg_error(toolong, "string exceeds its limit"))
}

/// Copy a NULL-terminated array of C-string pointers (an argv or envp) and
/// its strings out of guest memory, trusting none of the pointers. A null
/// array pointer is Linux's "no arguments" degenerate case, not a fault.
fn read_guest_ptr_array(ptr: u64) -> Result<Vec<Vec<u8>>, Error> {
    if ptr == 0 {
        return Ok(Vec::new());
    }
    let mut out: Vec<Vec<u8>> = Vec::new();
    loop {
        if out.len() >= MAX_ARG_COUNT {
            return Err(arg_error(libc::E2BIG, "argument list too long"));
        }
        let mut raw = [0u8; 8];
        if !copy_from_guest(ptr + (out.len() as u64) * 8, &mut raw) {
            return Err(arg_error(libc::EFAULT, "unreadable argument pointer"));
        }
        let entry = u64::from_ne_bytes(raw);
        if entry == 0 {
            return Ok(out);
        }
        out.push(read_guest_cstr(entry, MAX_ARG_STRLEN, libc::E2BIG)?);
    }
}

/// Default resolution of an exec pathname against its `dirfd` and flags, used
/// when the handler does not resolve it itself. An absolute path or `AT_FDCWD`
/// is used directly; `AT_EMPTY_PATH` with an empty pathname names the open fd
/// itself (`fexecve`); otherwise the path is relative to the directory the fd
/// refers to. The fd cases route through `/proc/self/fd`, so a `dirfd` the
/// handler virtualizes is first mapped back to its host fd.
fn resolve_at(dirfd: i32, raw: &[u8], flags: i32, handler: &dyn SystemCalls) -> PathBuf {
    let host_fd = handler.resolve_fd(dirfd).unwrap_or(dirfd);
    if raw.is_empty() && flags & AT_EMPTY_PATH != 0 {
        return PathBuf::from(format!("/proc/self/fd/{host_fd}"));
    }
    let path = Path::new(OsStr::from_bytes(raw));
    if path.is_absolute() || dirfd == AT_FDCWD {
        return path.to_path_buf();
    }
    PathBuf::from(format!("/proc/self/fd/{host_fd}")).join(path)
}

unsafe fn push_bytes(p: &mut u64, b: &[u8]) -> u64 {
    *p -= b.len() as u64;
    unsafe {
        ptr::copy_nonoverlapping(b.as_ptr(), *p as *mut u8, b.len());
    }
    *p
}

/// Push a string and its NUL terminator onto the descending stack, returning
/// the address of the first byte.
unsafe fn push_str(p: &mut u64, s: &[u8]) -> u64 {
    unsafe {
        push_bytes(p, &[0]);
        push_bytes(p, s)
    }
}

/// Build the initial stack from already-formed `argv` and `envp` (each entry
/// carrying no trailing NUL) and the `execfn` path for `AT_EXECFN`. Returns the
/// guest `rsp` and the `(start, len)` of the stack mapping.
fn build_stack(
    argv: &[Vec<u8>],
    envp: &[Vec<u8>],
    execfn: &[u8],
    main: &LoadedElf,
    interp_base: u64,
) -> Result<(u64, usize, usize), Error> {
    let stack = unsafe {
        libc::mmap(
            ptr::null_mut(),
            STACK_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_STACK,
            -1,
            0,
        )
    };
    if stack == libc::MAP_FAILED {
        return Err(Error::last_os_error("stack mmap"));
    }
    let mut p = (stack as u64) + STACK_SIZE as u64;

    let random = [0xa5u8; 16];

    let mut argv_addrs: Vec<u64> = argv
        .iter()
        .rev()
        .map(|s| unsafe { push_str(&mut p, s) })
        .collect();
    argv_addrs.reverse();

    let mut envp_addrs: Vec<u64> = envp
        .iter()
        .rev()
        .map(|s| unsafe { push_str(&mut p, s) })
        .collect();
    envp_addrs.reverse();

    let at_platform = unsafe { push_bytes(&mut p, b"x86_64\0") };
    let at_execfn = unsafe { push_str(&mut p, execfn) };
    let at_random = unsafe { push_bytes(&mut p, &random) };

    let hwcap = unsafe { libc::getauxval(libc::AT_HWCAP) };
    let hwcap2 = unsafe { libc::getauxval(libc::AT_HWCAP2) };
    let clktck = unsafe { libc::getauxval(libc::AT_CLKTCK) };
    let sysinfo_ehdr = unsafe { libc::getauxval(libc::AT_SYSINFO_EHDR) };

    let mut auxv: Vec<(u64, u64)> = vec![
        (libc::AT_PHDR, main.phdr_addr),
        (libc::AT_PHENT, main.ehdr.e_phentsize as u64),
        (libc::AT_PHNUM, main.ehdr.e_phnum as u64),
        (libc::AT_PAGESZ, PAGE_SIZE),
        (libc::AT_BASE, interp_base),
        (libc::AT_FLAGS, 0),
        (libc::AT_ENTRY, main.entry),
        (libc::AT_UID, unsafe { libc::getuid() } as u64),
        (libc::AT_EUID, unsafe { libc::geteuid() } as u64),
        (libc::AT_GID, unsafe { libc::getgid() } as u64),
        (libc::AT_EGID, unsafe { libc::getegid() } as u64),
        (libc::AT_PLATFORM, at_platform),
        (libc::AT_HWCAP, hwcap),
        (libc::AT_CLKTCK, clktck),
        (libc::AT_SECURE, 0),
        (libc::AT_RANDOM, at_random),
        (libc::AT_HWCAP2, hwcap2),
        (libc::AT_EXECFN, at_execfn),
    ];
    if sysinfo_ehdr != 0 {
        auxv.push((libc::AT_SYSINFO_EHDR, sysinfo_ehdr));
    }
    auxv.push((libc::AT_NULL, 0));

    let argc = argv.len() as u64;
    let fixed_size =
        8u64 + (argv.len() as u64 + 1) * 8 + (envp.len() as u64 + 1) * 8 + (auxv.len() as u64) * 16;

    let target_rsp = (p - fixed_size) & !15;
    p = target_rsp;

    unsafe {
        ptr::write(p as *mut u64, argc);
        p += 8;
        for a in &argv_addrs {
            ptr::write(p as *mut u64, *a);
            p += 8;
        }
        ptr::write(p as *mut u64, 0);
        p += 8;
        for e in &envp_addrs {
            ptr::write(p as *mut u64, *e);
            p += 8;
        }
        ptr::write(p as *mut u64, 0);
        p += 8;
        for (k, v) in &auxv {
            ptr::write(p as *mut u64, *k);
            ptr::write((p + 8) as *mut u64, *v);
            p += 16;
        }
    }

    Ok((target_rsp, stack as usize, STACK_SIZE))
}
