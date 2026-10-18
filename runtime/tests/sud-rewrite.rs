//! Under syscall user dispatch, a `syscall` instruction that traps is
//! rewritten into `call rax` when the host lets Chimera map the trampoline at
//! address 0 and protect it with a key, or, without strict null faults, map
//! it at all, and is left alone otherwise.
//!
//! The guest shares the test's address space, so a handler that sees the
//! guest's `exit_group` compares every private executable file mapping in
//! memory with the file it came from. The only difference allowed is a
//! rewritten system call, and there must be one exactly when the host can
//! host the trampoline.

use std::{
    fs::{self, File},
    os::unix::fs::FileExt,
    process::ExitCode,
    ptr, slice,
    sync::Mutex,
};

use chimera::{Backend, Sandbox, SystemCall, SystemCalls};
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Rewritten sites and any other differences between text and its file.
static FOUND: Mutex<(usize, usize)> = Mutex::new((0, 0));

struct Inspect;

impl SystemCalls for Inspect {
    fn pre_syscall(&self, call: &SystemCall) {
        if call.number == libc::SYS_exit_group as u64 {
            *FOUND.lock().unwrap() = compare_text();
        }
    }
}

fn compare_text() -> (usize, usize) {
    let (mut rewritten, mut other) = (0, 0);
    let maps = fs::read_to_string("/proc/self/maps").expect("maps");
    for line in maps.lines() {
        let fields: Vec<&str> = line.split_ascii_whitespace().collect();
        if fields.len() < 6 || fields[1] != "r-xp" || !fields[5].starts_with('/') {
            continue;
        }
        let (lo, hi) = fields[0].split_once('-').unwrap();
        let lo = usize::from_str_radix(lo, 16).unwrap();
        let hi = usize::from_str_radix(hi, 16).unwrap();
        let offset = u64::from_str_radix(fields[2], 16).unwrap();
        let Ok(file) = File::open(fields[5]) else {
            continue;
        };
        let mut disk = vec![0; hi - lo];
        let n = file.read_at(&mut disk, offset).unwrap_or(0);
        let memory = unsafe { slice::from_raw_parts(lo as *const u8, n) };
        let mut i = 0;
        while i < n {
            if memory[i] == disk[i] {
                i += 1;
            } else if disk[i..].starts_with(&[0x0f, 0x05]) && memory[i..].starts_with(&[0xff, 0xd0])
            {
                rewritten += 1;
                i += 2;
            } else {
                other += 1;
                i += 1;
            }
        }
    }
    (rewritten, other)
}

/// Whether this process may map address 0, which rewriting needs.
fn page_zero_mappable() -> bool {
    let page = unsafe {
        libc::mmap(
            ptr::null_mut(),
            4096,
            libc::PROT_READ,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE,
            -1,
            0,
        )
    };
    if !page.is_null() {
        return false;
    }
    unsafe { libc::munmap(page, 4096) };
    true
}

/// Run a guest that makes the same system calls repeatedly, and return the
/// rewritten sites its text holds by the end.
fn run(strict_null: bool) -> Result<usize, String> {
    *FOUND.lock().unwrap() = (0, 0);
    let mut sandbox = Sandbox::new("/bin/sh").expect("sandbox");
    sandbox.backend(Backend::SyscallUserDispatch);
    sandbox.strict_null(strict_null);
    sandbox.system_calls(Inspect);
    sandbox.args([
        "-c",
        "for i in 1 2 3; do read -r line < /proc/self/maps; done",
    ]);
    let status = sandbox.run().expect("run");
    if status.code() != 0 {
        return Err(format!("the guest exited {}", status.code()));
    }
    let (rewritten, other) = *FOUND.lock().unwrap();
    if other != 0 {
        return Err(format!(
            "{other} bytes of text differ from their files other than by a rewrite"
        ));
    }
    Ok(rewritten)
}

fn main() -> ExitCode {
    let mappable = page_zero_mappable();
    // Strict null faults need a protection key to make the trampoline
    // execute-only; without them any host that can map it will do.
    for (strict_null, expected) in [
        (true, mappable && chimera::mpk_enabled()),
        (false, mappable),
    ] {
        match run(strict_null) {
            Ok(rewritten) if expected == (rewritten > 0) => {}
            Ok(rewritten) => {
                eprintln!(
                    "strict_null {strict_null}: rewriting expected: {expected}, \
                     sites rewritten: {rewritten}"
                );
                return ExitCode::FAILURE;
            }
            Err(err) => {
                eprintln!("strict_null {strict_null}: {err}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
