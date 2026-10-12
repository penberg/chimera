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

fn main() -> ExitCode {
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
        "chimera version {} {} {} {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        if chimera::mpk_enabled() {
            "mpk"
        } else {
            "nompk"
        }
    );
    ExitCode::SUCCESS
}
