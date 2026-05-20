//! Guest runtime entry on Darwin.
//!
//! Chimera does *not* hand control to Apple's dyld for the guest — see
//! `dyld.rs` for the reason (cached-dyld `__DATA` collision). Instead, we
//! map the guest's Mach-O, link it in-process with our own `dyld::link`,
//! and dispatch straight to its `LC_MAIN` entry with the calling
//! convention `dyld` would have used: `x0=argc, x1=argv, x2=envp,
//! x3=apple`, `x30=0` (so `main`'s ret falls into the dispatcher's exit
//! path).

use std::{ffi::OsString, os::unix::ffi::OsStrExt, path::Path, ptr};

use crate::{Error, SystemCalls, arch::dispatch};

use super::{dyld, macho::load_macho};

pub fn execv(
    program: &Path,
    args: &[OsString],
    envs: Option<&[(OsString, OsString)]>,
    handler: Box<dyn SystemCalls>,
) -> Result<i32, Error> {
    let image = load_macho(program)?;
    let link = dyld::link(&image)?;
    let stack = build_main_frame(program, args, envs)?;

    let mut state = dispatch::InitialState::new(link.entry, stack.sp);
    state.regs[dispatch::X0] = stack.argc;
    state.regs[dispatch::X1] = stack.argv;
    state.regs[dispatch::X2] = stack.envp;
    state.regs[dispatch::X3] = stack.apple;
    // x30 = 0 is our "main returned" sentinel — the dispatcher catches a
    // subsequent `pc == 0` and exits with `x0` as the return code.
    state.regs[30] = 0;

    dispatch::start_thread(state, handler)
}

/// What `main(argc, argv, envp, apple)` needs to find in registers and on
/// the stack when called LC_MAIN-style.
struct MainFrame {
    sp: u64,
    argc: u64,
    argv: u64,
    envp: u64,
    apple: u64,
}

unsafe fn push_bytes(p: &mut u64, b: &[u8]) -> u64 {
    *p -= b.len() as u64;
    unsafe {
        ptr::copy_nonoverlapping(b.as_ptr(), *p as *mut u8, b.len());
    }
    *p
}

fn build_main_frame(
    program: &Path,
    args: &[OsString],
    envs_override: Option<&[(OsString, OsString)]>,
) -> Result<MainFrame, Error> {
    const STACK_SIZE: usize = 8 * 1024 * 1024;
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

    // argv: argv[0] is the program path itself, followed by the rest.
    let mut argv_strs: Vec<Vec<u8>> = Vec::with_capacity(args.len() + 1);
    let mut argv0 = program.as_os_str().as_bytes().to_vec();
    argv0.push(0);
    argv_strs.push(argv0);
    for a in args {
        let mut b = a.as_bytes().to_vec();
        b.push(0);
        argv_strs.push(b);
    }

    let mut envp_strs: Vec<Vec<u8>> = Vec::new();
    if let Some(over) = envs_override {
        for (k, v) in over {
            let mut s = k.as_bytes().to_vec();
            s.push(b'=');
            s.extend_from_slice(v.as_bytes());
            s.push(0);
            envp_strs.push(s);
        }
    } else {
        for (k, v) in std::env::vars_os() {
            let mut s = k.as_bytes().to_vec();
            s.push(b'=');
            s.extend_from_slice(v.as_bytes());
            s.push(0);
            envp_strs.push(s);
        }
    }

    // The `apple[]` array is dyld's per-process metadata channel. The only
    // entry we synthesise for v0 is `executable_path=`, which `_NSGetExecutablePath`
    // and friends consult.
    let mut exec_path_str = b"executable_path=".to_vec();
    exec_path_str.extend_from_slice(program.as_os_str().as_bytes());
    exec_path_str.push(0);

    // Push strings first, top-down.
    let mut argv_addrs: Vec<u64> = argv_strs
        .iter()
        .rev()
        .map(|s| unsafe { push_bytes(&mut p, s) })
        .collect();
    argv_addrs.reverse();

    let mut envp_addrs: Vec<u64> = envp_strs
        .iter()
        .rev()
        .map(|s| unsafe { push_bytes(&mut p, s) })
        .collect();
    envp_addrs.reverse();

    let exec_path_addr = unsafe { push_bytes(&mut p, &exec_path_str) };
    let apple_addrs: Vec<u64> = vec![exec_path_addr];

    // Now the pointer arrays: argv[] + NULL, envp[] + NULL, apple[] + NULL.
    // Each array's base address is what we pass in x1/x2/x3. We lay them
    // out contiguous-but-separately so each can be passed independently.
    let pointers =
        argv_addrs.len() as u64 + 1 + envp_addrs.len() as u64 + 1 + apple_addrs.len() as u64 + 1;
    let array_size = pointers * 8;
    let arrays_start = (p - array_size) & !15;
    p = arrays_start;

    let argv_ptr = p;
    unsafe {
        for a in &argv_addrs {
            ptr::write(p as *mut u64, *a);
            p += 8;
        }
        ptr::write(p as *mut u64, 0);
        p += 8;
        let envp_ptr_local = p;
        for e in &envp_addrs {
            ptr::write(p as *mut u64, *e);
            p += 8;
        }
        ptr::write(p as *mut u64, 0);
        p += 8;
        let apple_ptr_local = p;
        for a in &apple_addrs {
            ptr::write(p as *mut u64, *a);
            p += 8;
        }
        ptr::write(p as *mut u64, 0);
        // Final sp is just below the arrays we wrote, aligned to 16.
        let sp = arrays_start & !15;
        Ok(MainFrame {
            sp,
            argc: argv_strs.len() as u64,
            argv: argv_ptr,
            envp: envp_ptr_local,
            apple: apple_ptr_local,
        })
    }
}
