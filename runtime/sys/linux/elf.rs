//! ELF parsing and loading.

use std::{
    ffi::OsStr,
    fs,
    os::{
        fd::{AsRawFd, RawFd},
        unix::{ffi::OsStrExt, fs::FileExt},
    },
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
    ehdr: Ehdr,
    phdrs: Vec<Phdr>,
    pub interp: Option<PathBuf>,
    lo: u64,
    hi: u64,
    /// The image, held open since [`parse_elf`] read it, for [`map_elf`] to
    /// map `PT_LOAD` segments file-backed — a guest `madvise(MADV_DONTNEED)`
    /// on such a segment then re-reads the file (as it would natively),
    /// instead of zero-filling an anonymous copy and silently losing the
    /// segment's data. Holding the fd rather than the path pins the inode the
    /// headers were validated against, so a file swapped in at the same path
    /// later is never mapped, and the execve commit phase resolves no paths
    /// after the old image is torn down. Overwriting the inode in place still
    /// mutates the running image — the hazard every shared library has
    /// natively; a userspace loader cannot take `execve`'s `ETXTBSY` lock.
    file: fs::File,
}

impl AsRawFd for ParsedElf {
    fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

/// Read an ELF image and validate that Chimera can run it, without mapping
/// anything.
pub fn parse_elf(path: &Path) -> Result<ParsedElf, Error> {
    let file =
        fs::File::open(path).map_err(|e| Error::io(format!("opening {}", path.display()), e))?;
    // A short read is a malformed image (recoverable ENOEXEC), not an I/O
    // failure; likewise for the header-table and PT_INTERP reads below.
    let mut ehdr_bytes = [0u8; std::mem::size_of::<Ehdr>()];
    file.read_exact_at(&mut ehdr_bytes, 0).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::BadBinary(format!("{} is too short for an ELF", path.display()))
        } else {
            Error::io(format!("reading {}", path.display()), e)
        }
    })?;
    if &ehdr_bytes[..4] != b"\x7fELF" {
        return Err(Error::BadBinary(format!(
            "{} is not an ELF file",
            path.display()
        )));
    }
    if ehdr_bytes[4] != 2 {
        return Err(Error::BadBinary(format!("{} is not ELF64", path.display())));
    }
    if ehdr_bytes[5] != 1 {
        return Err(Error::BadBinary(format!(
            "{} is not little-endian",
            path.display()
        )));
    }

    let ehdr: Ehdr = unsafe { ptr::read_unaligned(ehdr_bytes.as_ptr() as *const Ehdr) };
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

    // Linux likewise accepts no other entry size; it is also what lets the
    // table be read below as a packed array of `Phdr`.
    if ehdr.e_phentsize as usize != std::mem::size_of::<Phdr>() {
        return Err(Error::BadBinary(format!(
            "{}: e_phentsize is not the size of a program header",
            path.display()
        )));
    }
    let phnum = ehdr.e_phnum as usize;
    let mut table = vec![0u8; phnum * std::mem::size_of::<Phdr>()];
    file.read_exact_at(&mut table, ehdr.e_phoff).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            Error::BadBinary(format!("{}: program headers exceed file", path.display()))
        } else {
            Error::io(format!("reading program headers of {}", path.display()), e)
        }
    })?;

    let mut phdrs = Vec::with_capacity(phnum);
    for entry in table.chunks_exact(std::mem::size_of::<Phdr>()) {
        let p: Phdr = unsafe { ptr::read_unaligned(entry.as_ptr() as *const Phdr) };
        // `map_elf` places each `PT_LOAD` by mapping the file at
        // `p_offset - (p_vaddr % PAGE_SIZE)`, which lands the bytes at
        // `p_vaddr` only when the two offsets share a page residue. Reject
        // the malformed case here, in the recoverable parse phase, so a guest
        // `execve` of such an image fails with `ENOEXEC` instead of dying
        // mid-commit after the old image is gone.
        if p.p_type == PT_LOAD && p.p_offset % PAGE_SIZE != p.p_vaddr % PAGE_SIZE {
            return Err(Error::BadBinary(format!(
                "{}: PT_LOAD vaddr {:#x} and file offset {:#x} disagree modulo the page size",
                path.display(),
                p.p_vaddr,
                p.p_offset
            )));
        }
        phdrs.push(p);
    }

    let mut interp = None;
    for ph in &phdrs {
        if ph.p_type == PT_INTERP {
            // Linux bounds the interpreter path by PATH_MAX; without the cap a
            // malformed p_filesz would drive the allocation here.
            if ph.p_filesz == 0 || ph.p_filesz > PAGE_SIZE {
                return Err(Error::BadBinary(format!(
                    "{}: PT_INTERP size {:#x}",
                    path.display(),
                    ph.p_filesz
                )));
            }
            let mut buf = vec![0u8; ph.p_filesz as usize];
            file.read_exact_at(&mut buf, ph.p_offset).map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    Error::BadBinary(format!("{}: PT_INTERP exceeds file", path.display()))
                } else {
                    Error::io(format!("reading PT_INTERP of {}", path.display()), e)
                }
            })?;
            let slice = buf.strip_suffix(b"\0").unwrap_or(&buf);
            interp = Some(PathBuf::from(OsStr::from_bytes(slice)));
            break;
        }
    }

    let (lo, hi) = load_range(&phdrs);
    Ok(ParsedElf {
        ehdr,
        phdrs,
        interp,
        lo,
        hi,
        file,
    })
}

/// Map a parsed ELF image into memory. The commit phase: the reservation and
/// `PT_LOAD` mappings it makes cannot be rolled back.
pub fn map_elf(parsed: &ParsedElf) -> Result<LoadedElf, Error> {
    map_elf_at(parsed, None)
}

/// Map a parsed ELF image for native execution behind syscall user dispatch:
/// executable segments keep `PROT_EXEC` (there is no translator to run them
/// for the guest), and an `ET_DYN` image is placed at `hint`, which the SUD
/// driver draws from its low guest arena so the image sits outside the
/// dispatch-exempt region (see [`crate::sys::linux::sud`]).
pub fn map_elf_native(parsed: &ParsedElf, hint: u64) -> Result<LoadedElf, Error> {
    map_elf_at(parsed, Some(hint))
}

fn map_elf_at(parsed: &ParsedElf, native_hint: Option<u64>) -> Result<LoadedElf, Error> {
    let ParsedElf {
        ehdr,
        phdrs,
        interp,
        lo,
        hi,
        file,
    } = parsed;
    let (lo, hi) = (*lo, *hi);

    let base = if ehdr.e_type == ET_DYN {
        let total = (hi - lo) as usize;
        let (addr, place) = match native_hint {
            // The address matters, so a taken hint is an error, not a silent
            // relocation into the exempt region.
            Some(hint) => (hint as *mut libc::c_void, libc::MAP_FIXED_NOREPLACE),
            None => (ptr::null_mut(), 0),
        };
        let reservation = unsafe {
            libc::mmap(
                addr,
                total,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | place,
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
        // W^X: under translation Chimera never executes guest pages natively —
        // the dispatcher reads them and runs translated blocks from the code
        // cache — so an executable segment is mapped read-only rather than
        // `PROT_EXEC`. This mirrors the `PROT_EXEC` stripping the syscall
        // driver applies to the libraries the dynamic linker maps later; here
        // it covers the executable and interpreter images the runtime maps
        // itself, which never pass through that path. `PROT_READ` keeps them
        // translatable. Under syscall user dispatch the CPU itself executes
        // the segment, so the native loader keeps `PROT_EXEC`.
        if ph.p_flags & PF_X != 0 {
            prot |= if native_hint.is_some() {
                libc::PROT_EXEC | libc::PROT_READ
            } else {
                libc::PROT_READ
            };
        }

        let fixed = if ehdr.e_type == ET_EXEC {
            libc::MAP_FIXED_NOREPLACE
        } else {
            libc::MAP_FIXED
        };

        // Map the file-backed part [vstart, file_map_end) from the image, at the
        // page-aligned file offset. [`parse_elf`] verified the segment's file and
        // virtual offsets share a page residue, so this places the bytes at
        // `vaddr`. The last partial page's tail past `p_filesz` is zero-filled by
        // the kernel. Mapped writable up front; `mprotect` sets the final prot.
        let file_end = vaddr + ph.p_filesz;
        let file_map_end = (file_end + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        let file_off = ph.p_offset - (vaddr - vstart);
        if ph.p_filesz > 0 {
            let flen = (file_map_end - vstart) as usize;
            let p = unsafe {
                libc::mmap(
                    vstart as *mut libc::c_void,
                    flen,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | fixed,
                    file.as_raw_fd(),
                    file_off as libc::off_t,
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
            // Bytes past `p_filesz` on the last file-backed page belong to `.bss`
            // (or nothing); zero them so `[p_filesz, p_memsz)` reads as zero.
            if ph.p_memsz > ph.p_filesz && file_end < file_map_end {
                unsafe {
                    ptr::write_bytes(file_end as *mut u8, 0, (file_map_end - file_end) as usize)
                };
            }
        }

        // Map any whole `.bss` pages beyond the file-backed part as anonymous.
        let anon_start = if ph.p_filesz > 0 {
            file_map_end
        } else {
            vstart
        };
        if vend > anon_start {
            let p = unsafe {
                libc::mmap(
                    anon_start as *mut libc::c_void,
                    (vend - anon_start) as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | fixed,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED || p as u64 != anon_start {
                return Err(Error::last_os_error(format!(
                    "mmap bss at {:#x}",
                    anon_start
                )));
            }
        }

        if ehdr.e_type == ET_EXEC {
            regions.push((vstart, len as u64));
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
