//! Guest runtime entry: build the initial stack and hand control to the dispatcher.

use std::{ffi::OsString, os::unix::ffi::OsStrExt, path::Path, ptr};

use crate::{Error, SystemCalls, arch::dispatch};

use super::elf::{LoadedElf, PAGE_SIZE, load_elf};

pub fn execv(
    program: &Path,
    args: &[OsString],
    envs: Option<&[(OsString, OsString)]>,
    handler: Box<dyn SystemCalls>,
) -> Result<i32, Error> {
    let main = load_elf(program)?;

    let (rip, interp_base) = if let Some(interp_path) = &main.interp {
        let interp = load_elf(interp_path)?;
        (interp.entry, interp.base)
    } else {
        (main.entry, 0)
    };

    let rsp = build_stack(program, args, envs, &main, interp_base)?;

    dispatch::start_thread(rip, rsp, handler)
}

unsafe fn push_bytes(p: &mut u64, b: &[u8]) -> u64 {
    *p -= b.len() as u64;
    unsafe {
        ptr::copy_nonoverlapping(b.as_ptr(), *p as *mut u8, b.len());
    }
    *p
}

fn build_stack(
    program: &Path,
    args: &[OsString],
    envs_override: Option<&[(OsString, OsString)]>,
    main: &LoadedElf,
    interp_base: u64,
) -> Result<u64, Error> {
    const STACK_SIZE: usize = 8 * 1024 * 1024;
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

    let mut execfn = program.as_os_str().as_bytes().to_vec();
    execfn.push(0);

    let random = [0xa5u8; 16];

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

    let at_platform = unsafe { push_bytes(&mut p, b"x86_64\0") };
    let at_execfn = unsafe { push_bytes(&mut p, &execfn) };
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

    let argc = argv_strs.len() as u64;
    let fixed_size = 8u64
        + (argv_strs.len() as u64 + 1) * 8
        + (envp_strs.len() as u64 + 1) * 8
        + (auxv.len() as u64) * 16;

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

    Ok(target_rsp)
}
