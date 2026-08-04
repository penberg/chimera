//! Guest runtime entry on Darwin: build the initial frame, hand control to the
//! dispatcher, and service `execve` by loading the new image and re-entering.
//!
//! Chimera does not hand control to Apple's dyld for the guest — see `dyld.rs`
//! for the reason (the cached-dyld `__DATA` collision). This path maps the
//! Mach-O itself, links a dynamic image in-process, and dispatches straight to
//! its `LC_MAIN`/`LC_UNIXTHREAD` entry with the calling convention dyld would
//! have used: `x0=argc`, `x1=argv`, `x2=envp`, `x3=apple`, and `x30` set to the
//! dispatcher's return sentinel so `main`'s `ret` lands on the clean-exit path.

use std::{
    ffi::{OsStr, OsString},
    fs,
    os::{fd::RawFd, unix::ffi::OsStrExt},
    path::{Path, PathBuf},
    ptr,
    sync::Arc,
};

use crate::{
    Error, SystemCalls,
    arch::dispatch::{self, ExitReason, Thread},
    process::Process,
    sys::darwin::{
        dyld,
        macho::{LoadedMachO, load_macho},
    },
    sys::mmap::{read_guest_cstr, read_guest_ptr_array},
};

const STACK_SIZE: usize = 8 * 1024 * 1024;

/// Darwin's whole-argument-block limit (`ARG_MAX`, 1 MiB) bounds any single
/// argv/envp string too.
const MAX_ARG_STRLEN: usize = 1024 * 1024;

/// Cap on argv/envp entry counts: every entry costs at least a stack pointer,
/// so more entries than could fit on the guest stack are `E2BIG` up front.
const MAX_ARG_COUNT: usize = STACK_SIZE / 4 / 8;

pub fn execv(
    program: &Path,
    args: &[OsString],
    envs: Option<&[(OsString, OsString)]>,
    handler: Box<dyn SystemCalls>,
    code_cache_size: usize,
) -> Result<i32, Error> {
    // The first image's argv and envp come from the embedder: argv[0] is the
    // program path, then the supplied args; the environment is the explicit
    // set if one was given, otherwise the host's.
    let mut argv: Vec<Vec<u8>> = Vec::with_capacity(args.len() + 1);
    argv.push(program.as_os_str().as_bytes().to_vec());
    for a in args {
        argv.push(a.as_bytes().to_vec());
    }
    let envp: Vec<Vec<u8>> = match envs {
        Some(over) => over.iter().map(|(k, v)| env_pair(k, v)).collect(),
        None => std::env::vars_os().map(|(k, v)| env_pair(&k, &v)).collect(),
    };

    let image = load_macho(program)?;
    // A dynamic image carries imports and rebases that must be applied before
    // it runs; Chimera links it in-process rather than handing control to
    // Apple's dyld (see dyld.rs). A static image is dispatched as mapped.
    let linked = if image.is_dynamic {
        Some(dyld::link(&image, program)?)
    } else {
        None
    };
    let frame = build_main_frame(&argv, &envp, program.as_os_str().as_bytes())?;

    super::set_executable_path(program);
    super::set_image_slide(image.slide);
    super::set_guest_args(frame.argc as i32, frame.argv, frame.envp, frame.apple);
    let process = Arc::new(Process::new(handler, code_cache_size)?);
    super::fault::set_process(&process);
    super::callback::set_process(&process);
    start_profiler(&process);
    let mut thread = Thread::new(process, image.entry, frame.sp)?;
    record_regions(&mut thread, &image, &frame);
    enter_frame(&mut thread, &frame);
    if let Some(linked) = &linked {
        run_initializers(&mut thread, &linked.initializers, &frame)?;
    }

    let reason = thread.run()?;
    drive(&mut thread, reason)
}

/// Drive a main thread to process termination: install each committed
/// execve's published image and re-enter the run, until the guest exits;
/// returns the process exit status. Two callers: [`execv`] above, with the
/// initial thread's first run reason, and a guest thread promoted to main by
/// a fork in its thread (see `Thread::reset_after_fork`) — that child's host
/// thread is all its new process has, so it must drive itself.
pub fn drive(thread: &mut Thread, mut reason: ExitReason) -> Result<i32, Error> {
    loop {
        match reason {
            ExitReason::Exited(code) => return Ok(code),
            ExitReason::Execve => {
                // A committed execve: the calling thread — main or a sibling —
                // already validated, parsed, and mapped the image and
                // published it on the shared process (a failed one took
                // `-errno` in place and never left the run). Wait until the
                // last sibling has drained out of the thread list before
                // tearing down mappings a straggler could still be
                // translating or executing.
                thread.process().wait_exec_quiesce();
                let PreparedExec {
                    path,
                    argv,
                    envp,
                    image,
                } = thread
                    .process()
                    .take_exec_request()
                    .expect("an Execve exit reason always has a published image");

                close_cloexec_fds()?;
                // POSIX: the new image starts with no exit handlers, and the
                // old image's are about to be unmapped anyway. The malloc
                // zone the old image registered dies with it too.
                thread.process().clear_atexit();
                // Data the old image left buffered in the shared stdio dies
                // with it, exactly as a native exec discards it.
                super::purge_guest_stdio();
                // Replaced before the teardown below: the old image may have
                // pointed the shared libSystem's `environ` into its own
                // memory (bash and `env` both assign it), and the in-process
                // link of the new image reads the environment before
                // anything else re-syncs it.
                sync_host_environ(&envp);
                reset_getopt();
                super::clear_guest_zones();
                // Tear down the old image: its recorded regions (image
                // reservation, stack, guest mmaps) are unmapped and every
                // stale translation dropped. The replacement was mapped at
                // prepare time into its own reservation, so it is untouched.
                thread.addr_space().reset();
                // The old image's exports die with it: linking the new image
                // must not resolve a bind against them.
                dyld::reset_images();
                let linked = if image.is_dynamic {
                    Some(dyld::link(&image, &path)?)
                } else {
                    None
                };
                super::set_executable_path(&path);
                super::set_image_slide(image.slide);
                let frame = build_main_frame(&argv, &envp, path.as_os_str().as_bytes())?;
                super::set_guest_args(frame.argc as i32, frame.argv, frame.envp, frame.apple);
                record_regions(thread, &image, &frame);
                thread.enter(image.entry, frame.sp);
                enter_frame(thread, &frame);
                // POSIX `execve` resets caught signals to their default
                // disposition (ignored stay ignored) and drops the alt stack.
                thread.signals_mut().on_execve();
                if let Some(linked) = &linked {
                    run_initializers(thread, &linked.initializers, &frame)?;
                }
            }
        }
        reason = thread.run()?;
    }
}

/// Start the guest sampling profiler, if `CHIMERA_PROFILE` asked for one: a
/// runtime thread that reads every guest thread's current PC on a timer and
/// reports the hottest addresses when the run ends. It never touches guest
/// state, so it cannot perturb what it measures beyond the sampling itself.
pub fn start_profiler(process: &Arc<Process>) {
    if !crate::trace::profile() {
        return;
    }
    let process = Arc::clone(process);
    std::thread::spawn(move || {
        let pid = unsafe { libc::getpid() };
        eprintln!(
            "chimera: profile[{pid}] — guest image slid by {:#x} (subtract to get file offsets)",
            crate::sys::darwin::image_slide()
        );
        let mut counts: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        let mut samples = 0u64;
        let mut last_report = std::time::Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(1));
            for pc in process.sample_guest_pcs() {
                *counts.entry(pc).or_default() += 1;
                samples += 1;
            }
            if last_report.elapsed() >= std::time::Duration::from_secs(5) {
                let mut top: Vec<(u64, u64)> = counts.iter().map(|(&p, &n)| (p, n)).collect();
                top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
                eprintln!("chimera: profile[{pid}] — {samples} samples, hottest guest PCs:");
                for (pc, n) in top.iter().take(10) {
                    eprintln!("  {:5.1}%  {pc:#x}", 100.0 * *n as f64 / samples as f64);
                }
                // Instruction words around the hottest PCs, so a spin loop can
                // be identified (and its image fingerprinted) from the report
                // alone — the addresses are useless once the process is gone.
                for (pc, _) in top.iter().take(8) {
                    let mut code = [0u8; 32];
                    if crate::sys::mmap::copy_from_guest(pc - 16, &mut code) {
                        let words: Vec<String> = code
                            .chunks(4)
                            .map(|w| format!("{:08x}", u32::from_ne_bytes(w.try_into().unwrap())))
                            .collect();
                        eprintln!("  code at {pc:#x}-16: {}", words.join(" "));
                    }
                    if let Some((base, name)) = find_image(*pc) {
                        eprintln!(
                            "  image of {pc:#x}: base {base:#x} + {:#x}  {name}",
                            pc - base
                        );
                    }
                }
                last_report = std::time::Instant::now();
            }
        }
    });
}

/// Find the Mach-O image containing a guest PC, by walking down page by page
/// to its header, and name it from `LC_ID_DYLIB` (or the recorded executable
/// path when the header has none). Best-effort, for the profiler's reports.
fn find_image(pc: u64) -> Option<(u64, String)> {
    const MH_MAGIC_64: u32 = 0xfeed_facf;
    let mut base = pc & !0x3fff;
    for _ in 0..4096 {
        let mut word = [0u8; 4];
        if !crate::sys::mmap::copy_from_guest(base, &mut word) {
            return None;
        }
        if u32::from_ne_bytes(word) == MH_MAGIC_64 {
            break;
        }
        base = base.checked_sub(0x4000)?;
    }
    let mut header = [0u8; 32];
    if !crate::sys::mmap::copy_from_guest(base, &mut header) {
        return None;
    }
    let ncmds = u32::from_ne_bytes(header[16..20].try_into().unwrap());
    let mut cmd_addr = base + 32;
    for _ in 0..ncmds.min(256) {
        let mut cmd = [0u8; 24];
        if !crate::sys::mmap::copy_from_guest(cmd_addr, &mut cmd) {
            return None;
        }
        let (kind, size) = (
            u32::from_ne_bytes(cmd[0..4].try_into().unwrap()),
            u32::from_ne_bytes(cmd[4..8].try_into().unwrap()),
        );
        if kind == 0xd {
            // LC_ID_DYLIB: the name's offset within the command, then bytes.
            let name_off = u32::from_ne_bytes(cmd[8..12].try_into().unwrap()) as u64;
            let mut name = vec![0u8; (size as u64 - name_off).min(256) as usize];
            if crate::sys::mmap::copy_from_guest(cmd_addr + name_off, &mut name) {
                let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
                return Some((base, String::from_utf8_lossy(&name[..end]).into_owned()));
            }
        }
        cmd_addr += size.max(8) as u64;
    }
    Some((base, "<main image>".into()))
}

/// Record the image reservation and the guest stack as regions of the address
/// space, so an execve teardown unmaps them.
fn record_regions(thread: &mut Thread, image: &LoadedMachO, frame: &MainFrame) {
    let mut space = thread.addr_space();
    // The image is recorded as a plain guest region — never unmapped by an
    // execve teardown — because the shared libSystem can legitimately keep
    // pointers into it past the exec: the xlocale collate component observed
    // (a later `wcscoll` in the new image chased it into the unmapped hole),
    // and anything else a constructor or libc call registered from image
    // memory. Same resolution as for guest-dlopen'd images: leak the address
    // space, which the new image cannot observe. The stack is different —
    // nothing in the shared runtime may outlive it legitimately — so it
    // stays runtime-owned and is unmapped.
    space.add_region(image.region.0 as usize, image.region.1);
    space.add_runtime_region(frame.stack.0, frame.stack.1);
}

/// Run the image's static initializers as guest calls, after linking and
/// before `main` — dyld's slot for them, with dyld's `(argc, argv, envp,
/// apple)` argument convention.
fn run_initializers(thread: &mut Thread, funcs: &[u64], frame: &MainFrame) -> Result<(), Error> {
    // The guest-call loop runs only while the thread is; `Thread::run` will
    // set this again when the main entry is dispatched.
    thread.running = true;
    for &func in funcs {
        if crate::trace::trace() {
            eprintln!("chimera: initializer {func:#x}");
        }
        thread.run_guest_call4(func, [frame.argc, frame.argv, frame.envp, frame.apple])?;
    }
    Ok(())
}

/// Seed the entry registers for a freshly built main frame.
fn enter_frame(thread: &mut Thread, frame: &MainFrame) {
    thread.state.regs[dispatch::X0] = frame.argc;
    thread.state.regs[dispatch::X1] = frame.argv;
    thread.state.regs[dispatch::X2] = frame.envp;
    thread.state.regs[dispatch::X3] = frame.apple;
    // Seed the link register with the return sentinel: a top-level `ret` jumps
    // there and the run loop exits with `x0` as the status. A distinctive
    // non-canonical value (not 0) so a guest call to a null pointer stays a
    // fault instead of being mistaken for a clean return.
    thread.state.regs[30] = dispatch::RETURN_SENTINEL;
}

/// Close every fd flagged close-on-exec, the way the kernel does when an exec
/// commits. Chimera services `execve` by re-entering in place — no host exec
/// runs — so `FD_CLOEXEC` must be applied by hand: the flag sits on the host
/// fd (the guest's `O_CLOEXEC` and `F_SETFD` pass straight through), but the
/// kernel honors it only at a real `execve`. Unlike Linux, no runtime fd needs
/// to survive the install: the prepared image was fully read and mapped at
/// prepare time.
fn close_cloexec_fds() -> Result<(), Error> {
    let entries =
        fs::read_dir("/dev/fd").map_err(|e| Error::io("execve: listing /dev/fd".to_string(), e))?;
    // Collect before closing: the directory walk holds an fd of its own, and
    // closing entries out from under it would corrupt the walk. By the time
    // the sweep runs the iterator has been dropped, so its fd fails the
    // `F_GETFD` below and is skipped.
    let fds: Vec<RawFd> = entries
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
        .collect();
    for fd in fds {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags >= 0 && flags & libc::FD_CLOEXEC != 0 {
            unsafe { libc::close(fd) };
        }
    }
    Ok(())
}

/// A validated `execve` replacement: the request read out of guest memory and
/// the new image, already mapped into its own reservation (everything slides,
/// so it coexists with the old image until the install unmaps the old one).
/// Owned data throughout, so it can be published on the shared [`Process`] by
/// whichever thread's `execve` committed and consumed later on the main host
/// thread.
///
/// [`Process`]: crate::process::Process
pub struct PreparedExec {
    path: PathBuf,
    argv: Vec<Vec<u8>>,
    envp: Vec<Vec<u8>>,
    image: LoadedMachO,
}

impl PreparedExec {
    /// Unmap the prepared image. For a commit that lost the publish race (a
    /// sibling's exec or an `exit_group` is already dissolving the group) —
    /// the caller never observes a result, but the mapping must not leak.
    pub fn discard(self) {
        if self.image.region.1 != 0 {
            unsafe {
                libc::munmap(
                    self.image.region.0 as *mut libc::c_void,
                    self.image.region.1,
                )
            };
        }
    }
}

/// Read an `execve` request out of guest memory, then parse and map the named
/// image. Runs in the calling thread, the way the kernel sequences an exec: an
/// error here means the caller takes `-errno` (see [`exec_errno`]) and
/// resumes, with no other thread disturbed.
pub fn prepare_exec(args: &[u64; 8]) -> Result<PreparedExec, Error> {
    let raw = read_guest_cstr(args[0], libc::PATH_MAX as usize, libc::ENAMETOOLONG)?;
    let path = PathBuf::from(OsStr::from_bytes(&raw));
    let argv = read_guest_ptr_array(args[1], MAX_ARG_COUNT, MAX_ARG_STRLEN)?;
    let envp = read_guest_ptr_array(args[2], MAX_ARG_COUNT, MAX_ARG_STRLEN)?;
    let image = load_macho(&path)?;
    Ok(PreparedExec {
        path,
        argv,
        envp,
        image,
    })
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

/// What `main(argc, argv, envp, apple)` finds in registers and on the stack when
/// called `LC_MAIN`-style: the entry stack pointer, the four argument values,
/// and the stack mapping itself.
struct MainFrame {
    sp: u64,
    argc: u64,
    argv: u64,
    envp: u64,
    apple: u64,
    stack: (usize, usize),
}

/// Copy `b` onto the descending stack and return the address of its first byte.
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

unsafe extern "C" {
    fn _NSGetEnviron() -> *mut *mut *mut libc::c_char;
    static mut optreset: libc::c_int;
    static mut optind: libc::c_int;
    static mut opterr: libc::c_int;
    static mut optopt: libc::c_int;
    static mut optarg: *mut libc::c_char;
}

/// Return the shared libSystem's `getopt` parser to its startup state.
///
/// An exec'd image starts with freshly initialised globals, but the runtime
/// and the guest share one libSystem whose `__DATA` survives the in-process
/// execve — so a previous image's option parse leaks into the new one's.
/// `env -i prog -x` is the concrete case: `env` consumes `-i` leaving
/// `optind` at 2, and `prog`'s own `getopt` then starts past `-x`, silently
/// dropping it (`sw_vers` run from Homebrew's `brew.sh` printed the full
/// version listing where a bare number was expected). `optreset` is BSD's
/// sanctioned way to make the next `getopt` call also discard its private
/// intra-argument cursor.
fn reset_getopt() {
    unsafe {
        optreset = 1;
        optind = 1;
        opterr = 1;
        optopt = 0;
        optarg = ptr::null_mut();
    }
}

/// Make the shared libSystem's `environ` match the environment the image
/// being installed was given.
///
/// The runtime and the guest share one libSystem, so `getenv` answers from
/// *its* `environ` — the one dyld set up for Chimera — not from the `envp`
/// the runtime hands the guest's `main`. A guest that execs with a new
/// environment and then calls `getenv` would read the old one. (This is
/// also why `_NSGetEnviron` is deliberately not intercepted — one
/// environment, one owner.)
///
/// `environ` is *assigned* a freshly allocated array, never patched up
/// through `setenv`/`unsetenv`: the old image may have pointed the global
/// (or individual entries) into its own memory — bash and `env` both assign
/// it — and no sequence of libSystem calls is guaranteed to re-own the
/// array itself (an exec into an empty environment adds and removes
/// nothing), so anything short of wholesale replacement can leave `environ`
/// dangling into memory the teardown unmaps. Assigning a foreign array is
/// the pattern those same guests use natively; libSystem's next `setenv`
/// copies it and takes over maintenance. The replaced array is deliberately
/// leaked — the old image owns it, and it either dies with the teardown or
/// was already leaked to stay safely reachable.
fn sync_host_environ(envp: &[Vec<u8>]) {
    unsafe {
        let arr = libc::malloc((envp.len() + 1) * std::mem::size_of::<*mut libc::c_char>())
            as *mut *mut libc::c_char;
        if arr.is_null() {
            return;
        }
        let mut n = 0;
        for entry in envp {
            // A guest string cannot carry an interior NUL, but the copy is
            // measured against it anyway rather than trusting the length.
            let s = libc::strndup(entry.as_ptr() as *const libc::c_char, entry.len());
            if !s.is_null() {
                *arr.add(n) = s;
                n += 1;
            }
        }
        *arr.add(n) = ptr::null_mut();
        *_NSGetEnviron() = arr;
    }
}

/// Build the initial guest stack from already-formed `argv` and `envp` (each
/// entry carrying no trailing NUL) and the executable path for the `apple[]`
/// array: the strings near the top, then the three NULL-terminated pointer
/// arrays, with the stack pointer left 16-byte aligned. Every write targets
/// the freshly `mmap`'d stack directly, so no fault-safe guest copy is
/// involved.
fn build_main_frame(
    argv: &[Vec<u8>],
    envp: &[Vec<u8>],
    exec_path: &[u8],
) -> Result<MainFrame, Error> {
    let stack = unsafe {
        libc::mmap(
            ptr::null_mut(),
            STACK_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if stack == libc::MAP_FAILED {
        return Err(Error::last_os_error("stack mmap"));
    }
    let mut p = (stack as u64) + STACK_SIZE as u64;

    // The `apple[]` array is dyld's per-process metadata channel. The only entry
    // synthesized here is `executable_path=`, which `_NSGetExecutablePath` and a
    // handful of other libSystem queries consult.
    let mut exec_entry = b"executable_path=".to_vec();
    exec_entry.extend_from_slice(exec_path);

    // Push the strings top-down.
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

    let apple_addrs: Vec<u64> = vec![unsafe { push_str(&mut p, &exec_entry) }];

    // Then the three NULL-terminated pointer arrays, laid out contiguously below
    // the strings; each array's base is the value passed in x1/x2/x3.
    let pointers =
        argv_addrs.len() as u64 + 1 + envp_addrs.len() as u64 + 1 + apple_addrs.len() as u64 + 1;
    let arrays_start = (p - pointers * 8) & !15;
    p = arrays_start;

    let argv_base = p;
    unsafe {
        for a in &argv_addrs {
            ptr::write(p as *mut u64, *a);
            p += 8;
        }
        ptr::write(p as *mut u64, 0);
        p += 8;
        let envp_base = p;
        for e in &envp_addrs {
            ptr::write(p as *mut u64, *e);
            p += 8;
        }
        ptr::write(p as *mut u64, 0);
        p += 8;
        let apple_base = p;
        for a in &apple_addrs {
            ptr::write(p as *mut u64, *a);
            p += 8;
        }
        ptr::write(p as *mut u64, 0);

        Ok(MainFrame {
            sp: arrays_start & !15,
            argc: argv.len() as u64,
            argv: argv_base,
            envp: envp_base,
            apple: apple_base,
            stack: (stack as usize, STACK_SIZE),
        })
    }
}

/// Format a `KEY=VALUE` environment entry (no trailing NUL; the frame builder
/// adds it).
fn env_pair(key: &OsStr, value: &OsStr) -> Vec<u8> {
    let mut s = key.as_bytes().to_vec();
    s.push(b'=');
    s.extend_from_slice(value.as_bytes());
    s
}
