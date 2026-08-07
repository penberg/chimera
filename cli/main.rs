mod fs;
mod opts;
mod prompt;
mod run;

use std::process::ExitCode;

use mimalloc::MiMalloc;

use opts::{Command, Opts};

/// Route every Chimera-side allocation through mimalloc, whose segments are
/// `mmap`-backed and never touch `brk`. This keeps Chimera's heap clear of the
/// guest libc's `brk`-managed `main_arena`, which shares the one process-wide
/// program break.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> ExitCode {
    let opts: Opts = argh::from_env();
    match opts.command {
        Command::Run(cmd) => run::run(cmd),
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
