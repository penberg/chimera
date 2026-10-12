mod fs;
mod fuser;
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

fn main() -> ExitCode {
    raise_open_file_limit();
    let opts: Opts = argh::from_env();
    match opts.command {
        Command::Run(cmd) => run::run(cmd),
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
