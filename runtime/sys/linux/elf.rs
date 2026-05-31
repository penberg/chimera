//! ELF parsing and loading.

use std::{
    ffi::OsStr,
    fs,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    ptr,
};

use crate::Error;

pub const PAGE_SIZE: u64 = 4096;

const ET_EXEC: u16 = 2;
const ET_DYN: u16 = 3;
const EM_X86_64: u16 = 62;

const PT_LOAD: u32 = 1;
const PT_INTERP: u32 = 3;

const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Ehdr {
    pub e_ident: [u8; 16],
    pub e_type: u16,
    pub e_machine: u16,
    pub e_version: u32,
    pub e_entry: u64,
    pub e_phoff: u64,
    pub e_shoff: u64,
    pub e_flags: u32,
    pub e_ehsize: u16,
    pub e_phentsize: u16,
    pub e_phnum: u16,
    pub e_shentsize: u16,
    pub e_shnum: u16,
    pub e_shstrndx: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Phdr {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
}

pub struct LoadedElf {
    pub ehdr: Ehdr,
    pub base: u64,
    pub entry: u64,
    pub phdr_addr: u64,
    pub interp: Option<PathBuf>,
    /// Host mappings owned by the loader for this image.
    ///
    /// `ET_EXEC` images record the individual `PT_LOAD` mappings created for
    /// each segment. `ET_DYN` images reserve one contiguous PROT_NONE span up
    /// front and then map their segments into it, so that single reservation
    /// remains the owned region to unmap later.
    pub regions: Vec<(u64, u64)>,
}

/// Read, validate, and map an ELF image: [`parse_elf`] followed by [`map_elf`].
pub fn load_elf(path: &Path) -> Result<LoadedElf, Error> {
    map_elf(&parse_elf(path)?)
}

/// An ELF image read and validated, but not yet mapped. Producing one touches
/// no memory, so a malformed image is reported as a recoverable error rather
/// than after the loader has started committing mappings.
pub struct ParsedElf {
    bytes: Vec<u8>,
    ehdr: Ehdr,
    phdrs: Vec<Phdr>,
    pub interp: Option<PathBuf>,
    lo: u64,
    hi: u64,
}

/// Read an ELF image and validate that Chimera can run it, without mapping
/// anything.
pub fn parse_elf(path: &Path) -> Result<ParsedElf, Error> {
    let bytes = fs::read(path).map_err(|e| Error::io(format!("reading {}", path.display()), e))?;
    if bytes.len() < std::mem::size_of::<Ehdr>() {
        return Err(Error::BadBinary(format!(
            "{} is too short for an ELF",
            path.display()
        )));
    }
    if &bytes[..4] != b"\x7fELF" {
        return Err(Error::BadBinary(format!(
            "{} is not an ELF file",
            path.display()
        )));
    }
    if bytes[4] != 2 {
        return Err(Error::BadBinary(format!("{} is not ELF64", path.display())));
    }
    if bytes[5] != 1 {
        return Err(Error::BadBinary(format!(
            "{} is not little-endian",
            path.display()
        )));
    }

    let ehdr: Ehdr = unsafe { ptr::read_unaligned(bytes.as_ptr() as *const Ehdr) };
    if ehdr.e_machine != EM_X86_64 {
        return Err(Error::BadBinary(format!(
            "{} is not x86-64",
            path.display()
        )));
    }
    if ehdr.e_type != ET_EXEC && ehdr.e_type != ET_DYN {
        return Err(Error::BadBinary(format!(
            "{} is not ET_EXEC or ET_DYN",
            path.display()
        )));
    }

    let phoff = ehdr.e_phoff as usize;
    let phentsize = ehdr.e_phentsize as usize;
    let phnum = ehdr.e_phnum as usize;
    if phoff + phentsize * phnum > bytes.len() {
        return Err(Error::BadBinary(format!(
            "{}: program headers exceed file",
            path.display()
        )));
    }

    let mut phdrs = Vec::with_capacity(phnum);
    for i in 0..phnum {
        let off = phoff + i * phentsize;
        let p: Phdr = unsafe { ptr::read_unaligned(bytes[off..].as_ptr() as *const Phdr) };
        phdrs.push(p);
    }

    let mut interp = None;
    for ph in &phdrs {
        if ph.p_type == PT_INTERP {
            let start = ph.p_offset as usize;
            let end = start + ph.p_filesz as usize;
            let slice = &bytes[start..end];
            let slice = slice.strip_suffix(b"\0").unwrap_or(slice);
            interp = Some(PathBuf::from(OsStr::from_bytes(slice)));
            break;
        }
    }

    let (lo, hi) = load_range(&phdrs);
    Ok(ParsedElf {
        bytes,
        ehdr,
        phdrs,
        interp,
        lo,
        hi,
    })
}

/// Map a parsed ELF image into memory. The commit phase: the reservation and
/// `PT_LOAD` mappings it makes cannot be rolled back.
pub fn map_elf(parsed: &ParsedElf) -> Result<LoadedElf, Error> {
    let ParsedElf {
        bytes,
        ehdr,
        phdrs,
        interp,
        lo,
        hi,
    } = parsed;
    let (lo, hi) = (*lo, *hi);

    let base = if ehdr.e_type == ET_DYN {
        let total = (hi - lo) as usize;
        let reservation = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if reservation == libc::MAP_FAILED {
            return Err(Error::last_os_error(format!("reserving {total} bytes")));
        }
        (reservation as u64).wrapping_sub(lo)
    } else {
        0
    };

    let mut phdr_addr = 0u64;
    let mut regions = if ehdr.e_type == ET_DYN {
        vec![(base.wrapping_add(lo), hi - lo)]
    } else {
        Vec::new()
    };
    for ph in phdrs {
        if ph.p_type != PT_LOAD {
            continue;
        }

        let vaddr = ph.p_vaddr.wrapping_add(base);
        let vstart = vaddr & !(PAGE_SIZE - 1);
        let vend = (vaddr + ph.p_memsz + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        let len = (vend - vstart) as usize;

        let mut prot = 0i32;
        if ph.p_flags & PF_R != 0 {
            prot |= libc::PROT_READ;
        }
        if ph.p_flags & PF_W != 0 {
            prot |= libc::PROT_WRITE;
        }
        // W^X: Chimera never executes guest pages natively — the dispatcher
        // reads them and runs translated blocks from the code cache — so an
        // executable segment is mapped read-only rather than `PROT_EXEC`. This
        // mirrors the `PROT_EXEC` stripping the syscall driver applies to the
        // libraries the dynamic linker maps later; here it covers the
        // executable and interpreter images the runtime maps itself, which
        // never pass through that path. `PROT_READ` keeps them translatable.
        if ph.p_flags & PF_X != 0 {
            prot |= libc::PROT_READ;
        }

        let map_flags = if ehdr.e_type == ET_EXEC {
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE
        } else {
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED
        };

        let p = unsafe {
            libc::mmap(
                vstart as *mut libc::c_void,
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                map_flags,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(Error::last_os_error(format!(
                "mmap PT_LOAD at {:#x}",
                vstart
            )));
        }
        if p as u64 != vstart {
            return Err(Error::BadBinary(format!(
                "mmap returned {:p}, expected {:#x}",
                p, vstart
            )));
        }
        if ehdr.e_type == ET_EXEC {
            regions.push((vstart, len as u64));
        }

        if ph.p_filesz > 0 {
            unsafe {
                ptr::copy_nonoverlapping(
                    bytes.as_ptr().add(ph.p_offset as usize),
                    vaddr as *mut u8,
                    ph.p_filesz as usize,
                );
            }
        }

        if unsafe { libc::mprotect(vstart as *mut libc::c_void, len, prot) } != 0 {
            return Err(Error::last_os_error(format!("mprotect at {:#x}", vstart)));
        }

        let phdr_off = ehdr.e_phoff;
        let phdr_end = phdr_off + (ehdr.e_phentsize as u64) * (ehdr.e_phnum as u64);
        if ph.p_offset <= phdr_off && phdr_end <= ph.p_offset + ph.p_filesz {
            phdr_addr = vaddr + (phdr_off - ph.p_offset);
        }
    }

    Ok(LoadedElf {
        ehdr: *ehdr,
        base,
        entry: ehdr.e_entry.wrapping_add(base),
        phdr_addr,
        interp: interp.clone(),
        regions,
    })
}

fn load_range(phdrs: &[Phdr]) -> (u64, u64) {
    let mut lo = u64::MAX;
    let mut hi = 0u64;
    for p in phdrs {
        if p.p_type != PT_LOAD {
            continue;
        }
        let s = p.p_vaddr & !(PAGE_SIZE - 1);
        let e = (p.p_vaddr + p.p_memsz + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        if s < lo {
            lo = s;
        }
        if e > hi {
            hi = e;
        }
    }
    (lo, hi)
}
