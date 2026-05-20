//! Mach-O parsing and loading.
//!
//! Mirrors the ELF flow in `sys/linux/elf.rs`: parse the executable, map its
//! `LC_SEGMENT_64` segments, and identify the dynamic linker
//! (`LC_LOAD_DYLINKER`). If a linker is named, the caller loads it as a
//! second image and dispatches from *its* entry — exactly the way the ELF
//! side hands control to `ld-linux.so` and lets it do the dynamic-linking
//! work as ordinary translated code.
//!
//! The loader handles fat (universal) binaries by extracting the arm64 (or
//! arm64e) slice. It does not interpret `LC_DYLD_CHAINED_FIXUPS`,
//! `LC_DYLD_INFO_ONLY`, or any other binding metadata — that is dyld's job,
//! not ours.

use std::{
    ffi::OsStr,
    fs,
    os::unix::ffi::OsStrExt,
    path::{Path, PathBuf},
    ptr,
};

use crate::Error;

pub const PAGE_SIZE: u64 = 16 * 1024; // macOS arm64 uses 16K pages.

const MH_MAGIC_64: u32 = 0xfeedfacf;
const FAT_MAGIC: u32 = 0xcafebabe;
const FAT_CIGAM: u32 = 0xbebafeca;
const FAT_MAGIC_64: u32 = 0xcafebabf;
const FAT_CIGAM_64: u32 = 0xbfbafeca;

const MH_EXECUTE: u32 = 2;
const MH_DYLINKER: u32 = 7;
const MH_PIE: u32 = 0x200000;

const CPU_TYPE_ARM64: i32 = 0x0100000c;
const CPU_SUBTYPE_ARM64_ALL: i32 = 0;
const CPU_SUBTYPE_MASK: i32 = 0x00ffffff;

const LC_SEGMENT_64: u32 = 0x19;
const LC_UNIXTHREAD: u32 = 0x5;
const LC_LOAD_DYLINKER: u32 = 0xe;
const LC_MAIN: u32 = 0x80000028;

const ARM_THREAD_STATE64: u32 = 6;

const VM_PROT_READ: u32 = 0x1;
const VM_PROT_WRITE: u32 = 0x2;
const VM_PROT_EXECUTE: u32 = 0x4;

#[repr(C)]
#[derive(Clone, Copy)]
struct FatHeader {
    magic: u32,
    nfat_arch: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FatArch {
    cputype: i32,
    cpusubtype: i32,
    offset: u32,
    size: u32,
    align: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FatArch64 {
    cputype: i32,
    cpusubtype: i32,
    offset: u64,
    size: u64,
    align: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MachHeader64 {
    magic: u32,
    cputype: i32,
    cpusubtype: i32,
    filetype: u32,
    ncmds: u32,
    sizeofcmds: u32,
    flags: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LoadCommand {
    cmd: u32,
    cmdsize: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SegmentCommand64 {
    cmd: u32,
    cmdsize: u32,
    segname: [u8; 16],
    vmaddr: u64,
    vmsize: u64,
    fileoff: u64,
    filesize: u64,
    maxprot: u32,
    initprot: u32,
    nsects: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct EntryPointCommand {
    cmd: u32,
    cmdsize: u32,
    entryoff: u64,
    stacksize: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DylinkerCommand {
    cmd: u32,
    cmdsize: u32,
    name_offset: u32,
    // followed by the path string padded to cmdsize.
}

pub struct LoadedMachO {
    /// Slide applied to the binary's virtual addresses. Zero for non-PIE
    /// binaries that were mapped at their requested vmaddr.
    pub slide: u64,
    /// `__TEXT.vmaddr + slide` — the loaded address of the Mach header.
    /// The in-process linker (`dyld.rs`) reads load commands from this
    /// address and resolves `LC_MAIN.entryoff` against it.
    pub mh_addr: u64,
}

pub fn load_macho(path: &Path) -> Result<LoadedMachO, Error> {
    let bytes = fs::read(path).map_err(|e| Error::io(format!("reading {}", path.display()), e))?;
    let (slice_start, slice_len) = pick_macho_slice(&bytes, path)?;
    load_macho_slice(&bytes[slice_start..slice_start + slice_len], path)
}

fn pick_macho_slice(bytes: &[u8], path: &Path) -> Result<(usize, usize), Error> {
    if bytes.len() < 4 {
        return Err(Error::BadBinary(format!(
            "{}: file too short",
            path.display()
        )));
    }
    let magic = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

    // Plain (thin) 64-bit Mach-O — use the whole file.
    if magic == MH_MAGIC_64 {
        return Ok((0, bytes.len()));
    }

    // Fat / universal binary — pick the arm64(e) slice. We prefer plain
    // arm64; if only arm64e is present (as with `/usr/lib/dyld`) we accept
    // that — every instruction we have to translate has the same encoding,
    // we just have to handle PAC ops in the translator.
    let swap = match magic {
        FAT_MAGIC | FAT_MAGIC_64 => false,
        FAT_CIGAM | FAT_CIGAM_64 => true,
        _ => {
            return Err(Error::BadBinary(format!(
                "{} is not a Mach-O (magic {:#x})",
                path.display(),
                magic
            )));
        }
    };
    let is_64 = matches!(magic, FAT_MAGIC_64 | FAT_CIGAM_64);

    if bytes.len() < std::mem::size_of::<FatHeader>() {
        return Err(Error::BadBinary(format!(
            "{}: truncated fat header",
            path.display()
        )));
    }
    let fh: FatHeader = unsafe { ptr::read_unaligned(bytes.as_ptr() as *const _) };
    let n = if swap {
        fh.nfat_arch.swap_bytes()
    } else {
        fh.nfat_arch
    };

    let entry_size = if is_64 {
        std::mem::size_of::<FatArch64>()
    } else {
        std::mem::size_of::<FatArch>()
    };
    let table = std::mem::size_of::<FatHeader>();
    if table + (n as usize) * entry_size > bytes.len() {
        return Err(Error::BadBinary(format!(
            "{}: truncated fat arch table",
            path.display()
        )));
    }

    let mut chosen: Option<(usize, usize, i32)> = None;
    for i in 0..n as usize {
        let off = table + i * entry_size;
        let (cputype, cpusubtype, offset, size) = if is_64 {
            let a: FatArch64 = unsafe { ptr::read_unaligned(bytes.as_ptr().add(off) as *const _) };
            let s = |v: i32| if swap { v.swap_bytes() } else { v };
            let u = |v: u64| if swap { v.swap_bytes() } else { v };
            (
                s(a.cputype),
                s(a.cpusubtype),
                u(a.offset) as usize,
                u(a.size) as usize,
            )
        } else {
            let a: FatArch = unsafe { ptr::read_unaligned(bytes.as_ptr().add(off) as *const _) };
            let s = |v: i32| if swap { v.swap_bytes() } else { v };
            let u = |v: u32| if swap { v.swap_bytes() } else { v };
            (
                s(a.cputype),
                s(a.cpusubtype),
                u(a.offset) as usize,
                u(a.size) as usize,
            )
        };
        if cputype != CPU_TYPE_ARM64 {
            continue;
        }
        let subtype = cpusubtype & CPU_SUBTYPE_MASK;
        let prefer = subtype == CPU_SUBTYPE_ARM64_ALL;
        if chosen.is_none() || prefer {
            chosen = Some((offset, size, subtype));
            if prefer {
                break;
            }
        }
    }
    chosen.map(|(off, size, _)| (off, size)).ok_or_else(|| {
        Error::BadBinary(format!("{}: no arm64 slice in fat binary", path.display()))
    })
}

fn load_macho_slice(bytes: &[u8], path: &Path) -> Result<LoadedMachO, Error> {
    if bytes.len() < std::mem::size_of::<MachHeader64>() {
        return Err(Error::BadBinary(format!(
            "{} is too short for a Mach-O",
            path.display()
        )));
    }
    let header: MachHeader64 = unsafe { ptr::read_unaligned(bytes.as_ptr() as *const _) };
    if header.magic != MH_MAGIC_64 {
        return Err(Error::BadBinary(format!(
            "{} is not a 64-bit Mach-O (magic {:#x})",
            path.display(),
            header.magic
        )));
    }
    if header.cputype != CPU_TYPE_ARM64 {
        return Err(Error::BadBinary(format!(
            "{} is not arm64 (cputype {:#x})",
            path.display(),
            header.cputype
        )));
    }
    if header.filetype != MH_EXECUTE && header.filetype != MH_DYLINKER {
        return Err(Error::Unsupported(format!(
            "{}: Mach-O filetype {}",
            path.display(),
            header.filetype
        )));
    }
    let is_pie = (header.flags & MH_PIE) != 0;

    // First pass: collect segments and find entry / dylinker.
    let mut off = std::mem::size_of::<MachHeader64>();
    let cmds_end = off + header.sizeofcmds as usize;
    if cmds_end > bytes.len() {
        return Err(Error::BadBinary(format!(
            "{}: load commands exceed file",
            path.display()
        )));
    }

    let mut segments: Vec<SegmentCommand64> = Vec::new();
    let mut entry_pc: Option<u64> = None;
    let mut entry_file_offset: Option<u64> = None;
    let mut dylinker: Option<PathBuf> = None;

    for _ in 0..header.ncmds {
        if off + std::mem::size_of::<LoadCommand>() > cmds_end {
            return Err(Error::BadBinary(format!(
                "{}: truncated load command at offset {}",
                path.display(),
                off
            )));
        }
        let lc: LoadCommand = unsafe { ptr::read_unaligned(bytes.as_ptr().add(off) as *const _) };
        if lc.cmdsize == 0 || off + lc.cmdsize as usize > cmds_end {
            return Err(Error::BadBinary(format!(
                "{}: malformed load command at offset {}",
                path.display(),
                off
            )));
        }
        match lc.cmd {
            LC_SEGMENT_64 => {
                let seg: SegmentCommand64 =
                    unsafe { ptr::read_unaligned(bytes.as_ptr().add(off) as *const _) };
                segments.push(seg);
            }
            LC_UNIXTHREAD => {
                let body = off + std::mem::size_of::<LoadCommand>();
                if body + 8 > cmds_end {
                    return Err(Error::BadBinary(format!(
                        "{}: truncated LC_UNIXTHREAD",
                        path.display()
                    )));
                }
                let flavor: u32 =
                    unsafe { ptr::read_unaligned(bytes.as_ptr().add(body) as *const u32) };
                if flavor != ARM_THREAD_STATE64 {
                    return Err(Error::Unsupported(format!(
                        "{}: LC_UNIXTHREAD flavor {} on arm64",
                        path.display(),
                        flavor
                    )));
                }
                let state = body + 8;
                let pc_off = state + (29 + 3) * 8;
                if pc_off + 8 > cmds_end {
                    return Err(Error::BadBinary(format!(
                        "{}: truncated thread state",
                        path.display()
                    )));
                }
                let pc: u64 =
                    unsafe { ptr::read_unaligned(bytes.as_ptr().add(pc_off) as *const u64) };
                entry_pc = Some(pc);
            }
            LC_MAIN => {
                let ep: EntryPointCommand =
                    unsafe { ptr::read_unaligned(bytes.as_ptr().add(off) as *const _) };
                entry_file_offset = Some(ep.entryoff);
            }
            LC_LOAD_DYLINKER => {
                let dc: DylinkerCommand =
                    unsafe { ptr::read_unaligned(bytes.as_ptr().add(off) as *const _) };
                let name_off = off + dc.name_offset as usize;
                if name_off >= off + lc.cmdsize as usize {
                    return Err(Error::BadBinary(format!(
                        "{}: LC_LOAD_DYLINKER name offset out of range",
                        path.display()
                    )));
                }
                let max = off + lc.cmdsize as usize - name_off;
                let mut n = 0usize;
                while n < max && bytes[name_off + n] != 0 {
                    n += 1;
                }
                let s = &bytes[name_off..name_off + n];
                dylinker = Some(PathBuf::from(OsStr::from_bytes(s)));
            }
            _ => {}
        }
        off += lc.cmdsize as usize;
    }

    let (vm_lo, vm_hi) = segments_range(&segments);
    let total = (vm_hi - vm_lo) as usize;
    let needs_slide = is_pie || vm_lo == 0 || header.filetype == MH_DYLINKER;
    let slide = if needs_slide && total > 0 {
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
            return Err(Error::last_os_error(format!(
                "reserving {} bytes for {}",
                total,
                path.display(),
            )));
        }
        (reservation as u64).wrapping_sub(vm_lo)
    } else {
        0
    };

    let mut mh_addr: Option<u64> = None;
    for seg in &segments {
        if seg_name(seg) == "__PAGEZERO" {
            continue;
        }
        if seg.vmsize == 0 {
            continue;
        }
        let vaddr = seg.vmaddr.wrapping_add(slide);
        let vstart = vaddr & !(PAGE_SIZE - 1);
        let vend = (vaddr + seg.vmsize + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        let len = (vend - vstart) as usize;

        let mut prot = 0i32;
        if seg.initprot & VM_PROT_READ != 0 {
            prot |= libc::PROT_READ;
        }
        if seg.initprot & VM_PROT_WRITE != 0 {
            prot |= libc::PROT_WRITE;
        }
        if seg.initprot & VM_PROT_EXECUTE != 0 {
            prot |= libc::PROT_EXEC;
        }

        let p = unsafe {
            libc::mmap(
                vstart as *mut libc::c_void,
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(Error::last_os_error(format!(
                "mmap segment {} at {:#x}",
                seg_name(seg),
                vstart,
            )));
        }
        if seg.filesize > 0 {
            unsafe {
                ptr::copy_nonoverlapping(
                    bytes.as_ptr().add(seg.fileoff as usize),
                    vaddr as *mut u8,
                    seg.filesize as usize,
                );
            }
        }
        if unsafe { libc::mprotect(vstart as *mut libc::c_void, len, prot) } != 0 {
            return Err(Error::last_os_error(format!(
                "mprotect {} at {:#x}",
                seg_name(seg),
                vstart,
            )));
        }
        if seg_name(seg) == "__TEXT" {
            mh_addr = Some(vaddr);
        }
    }

    let mh_addr = mh_addr
        .ok_or_else(|| Error::BadBinary(format!("{}: no __TEXT segment", path.display())))?;
    let _ = (entry_pc, entry_file_offset, dylinker, is_pie);

    Ok(LoadedMachO { slide, mh_addr })
}

fn segments_range(segs: &[SegmentCommand64]) -> (u64, u64) {
    let mut lo = u64::MAX;
    let mut hi = 0u64;
    for s in segs {
        if seg_name(s) == "__PAGEZERO" {
            continue;
        }
        if s.vmsize == 0 {
            continue;
        }
        let start = s.vmaddr & !(PAGE_SIZE - 1);
        let end = (s.vmaddr + s.vmsize + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        if start < lo {
            lo = start;
        }
        if end > hi {
            hi = end;
        }
    }
    if lo == u64::MAX { (0, 0) } else { (lo, hi) }
}

fn seg_name(s: &SegmentCommand64) -> &str {
    let end = s
        .segname
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(s.segname.len());
    std::str::from_utf8(&s.segname[..end]).unwrap_or("")
}
