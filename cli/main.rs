mod fs;
// A merged view is served over FUSE, which only the Linux kernel offers.
#[cfg(target_os = "linux")]
mod fuser;
#[cfg(target_os = "linux")]
mod mount;
mod opts;
mod prompt;
mod run;

use std::process::ExitCode;

use mimalloc::MiMalloc;

use opts::{Command, Opts};

/// Route every Chimera-side allocation through mimalloc, whose segments are
/// `mmap`-backed and never touch `brk`, keeping Chimera's heap traffic out of
/// the process's `brk` segment. The guest's own `brk` is virtualized by the
/// runtime, so the real break belongs to the host libc alone.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// The runtime parks its backing descriptors — and a session its hold on the
/// filesystem and its copy of stderr — above the guest's descriptor space, at
/// 512 and up, so the guest keeps the low, dense numbers POSIX promises. A
/// process on macOS starts with a soft limit of 256 open files, below that
/// floor, and an `F_DUPFD` to a number past the limit is `EINVAL`; lift the
/// soft limit to what the hard limit allows before anything is opened. On
/// macOS the soft limit may not exceed `OPEN_MAX` even when the hard limit
/// is unbounded.
fn raise_open_file_limit() {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return;
    }
    // `OPEN_MAX` (`<sys/syslimits.h>`); the libc crate has no binding.
    #[cfg(target_os = "macos")]
    let ceiling = limit.rlim_max.min(10240);
    #[cfg(not(target_os = "macos"))]
    let ceiling = limit.rlim_max;
    if limit.rlim_cur < ceiling {
        limit.rlim_cur = ceiling;
        unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) };
    }
}

/// A guest's `posix_spawn` on Darwin is served as a fork of this
/// multithreaded process that links the new image in place, never a fresh
/// image. The Objective-C runtime refuses `+initialize` in such a child for
/// any class another thread was initialising at the fork — it aborts the child
/// — unless `OBJC_DISABLE_INITIALIZE_FORK_SAFETY` is set, and libobjc reads
/// that from the environment once, when the process starts. So it has to be
/// in place before `chimera` is: re-execute this process with it set. The
/// guest inherits the variable, which is also what its own fork children need.
#[cfg(target_os = "macos")]
fn ensure_fork_safety_override() {
    use std::{
        ffi::CString,
        os::unix::ffi::{OsStrExt, OsStringExt},
    };

    const VAR: &str = "OBJC_DISABLE_INITIALIZE_FORK_SAFETY";
    if std::env::var_os(VAR).is_some() {
        return;
    }
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let Ok(exe) = CString::new(exe.as_os_str().as_bytes()) else {
        return;
    };
    let argv: Vec<CString> = std::env::args_os()
        .filter_map(|a| CString::new(a.as_bytes()).ok())
        .collect();
    let mut envp: Vec<CString> = std::env::vars_os()
        .filter_map(|(k, v)| {
            let mut e = k.into_vec();
            e.push(b'=');
            e.extend(v.as_bytes());
            CString::new(e).ok()
        })
        .collect();
    envp.push(CString::new(format!("{VAR}=YES")).expect("no interior NUL"));
    let mut argv_ptrs: Vec<*const libc::c_char> = argv.iter().map(|a| a.as_ptr()).collect();
    argv_ptrs.push(std::ptr::null());
    let mut envp_ptrs: Vec<*const libc::c_char> = envp.iter().map(|e| e.as_ptr()).collect();
    envp_ptrs.push(std::ptr::null());
    // A failed exec leaves this process to run as it is; the spawn children
    // then take their chances with the check.
    unsafe { libc::execve(exe.as_ptr(), argv_ptrs.as_ptr(), envp_ptrs.as_ptr()) };
}

fn main() -> ExitCode {
    #[cfg(target_os = "macos")]
    ensure_fork_safety_override();
    raise_open_file_limit();
    let opts: Opts = argh::from_env();
    match opts.command {
        Command::Run(cmd) => run::run(cmd),
        #[cfg(target_os = "linux")]
        Command::Mount(cmd) => mount::command(cmd),
        Command::Version(_) => version(),
        Command::Fs(cmd) => fs::command(cmd.action),
    }
}

fn version() -> ExitCode {
    println!(
        "chimera version {} linux x86-64 {}",
        env!("CARGO_PKG_VERSION"),
        if chimera::mpk_enabled() {
            "mpk"
        } else {
            "nompk"
        }
    );
    ExitCode::SUCCESS
}
