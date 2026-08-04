//! Chimera's in-process Mach-O dynamic linker.
//!
//! Chimera and a Darwin guest cannot share Apple's dyld: the kernel maps
//! the shared cache (containing dyld + libSystem + everything) at one
//! system-wide VA, and Chimera's own libsystem startup already initialises
//! that cached dyld for itself. By the time the guest tries to bring up
//! its own dyld, the cached `__DATA` already says "initialized" and points
//! at Chimera's heap. See [[project-darwin-arm64-port-state]].
//!
//! The workaround is to skip dyld entirely. This module reads the guest's
//! Mach-O, walks its `LC_DYLD_CHAINED_FIXUPS` *or* `LC_DYLD_INFO_ONLY`
//! metadata, resolves imports against the host's loaded libraries (via
//! `dlsym(RTLD_DEFAULT, ...)`), patches the binding pointers, performs
//! rebases, sets up thread-local-variable thunks, and returns the entry
//! address for the dispatcher to jump into. dyld is never executed for the
//! guest.
//!
//! **Scope (v1)**:
//! - `DYLD_CHAINED_PTR_64` and `DYLD_CHAINED_PTR_64_OFFSET` (chained fixups
//!   — newer clang output).
//! - `DYLD_CHAINED_PTR_ARM64E` / `_USERLAND` / `_USERLAND24` (arm64e chained
//!   fixups — stock system binaries like `/bin/zsh`). Auth pointers are written
//!   plain (key/diversity dropped); the translator strips PAC on use, so
//!   unsigned pointers round-trip.
//! - `LC_DYLD_INFO_ONLY` rebase + bind + lazy-bind opcode streams (classic
//!   format — what rustc currently emits).
//! - `LC_MAIN`-style entry points.
//! - Thread-local-variable initialization: we walk the `S_THREAD_LOCAL_*`
//!   sections, allocate a pthread key per image, install a Chimera-provided
//!   thunk into each descriptor's `thunk` slot, and stash the TLS image
//!   template so first-touch threads can allocate fresh per-thread storage.

use std::{collections::HashMap, ffi::CString, path::Path, ptr, sync::Mutex};

use crate::Error;

use super::macho::{LoadedMachO, load_macho};

// === Mach-O load command IDs ===

const LC_SEGMENT_64: u32 = 0x19;
const LC_LOAD_DYLIB: u32 = 0xc;
const LC_LOAD_WEAK_DYLIB: u32 = 0x80000018;
const LC_REEXPORT_DYLIB: u32 = 0x8000001f;
const LC_RPATH: u32 = 0x8000001c;
const LC_DYLD_INFO: u32 = 0x22;
const LC_DYLD_INFO_ONLY: u32 = 0x80000022;
const LC_DYLD_CHAINED_FIXUPS: u32 = 0x80000034;

// === Chained-fixup pointer formats ===

// Values are from <mach-o/fixup-chains.h>. NB the userland formats are 9 and 12,
// not 7 and 8 (7 is ARM64E_KERNEL, 8 is 64_KERNEL_CACHE) — a stock arm64e binary
// like /bin/zsh uses ARM64E_USERLAND24 (12 = 0xc).
const DYLD_CHAINED_PTR_ARM64E: u16 = 1;
const DYLD_CHAINED_PTR_64: u16 = 2;
const DYLD_CHAINED_PTR_64_OFFSET: u16 = 6;
const DYLD_CHAINED_PTR_ARM64E_USERLAND: u16 = 9;
const DYLD_CHAINED_PTR_ARM64E_USERLAND24: u16 = 12;

// === Imports formats ===

const DYLD_CHAINED_IMPORT: u32 = 1;
const DYLD_CHAINED_IMPORT_ADDEND: u32 = 2;
const DYLD_CHAINED_IMPORT_ADDEND64: u32 = 3;

const DYLD_CHAINED_PTR_START_NONE: u16 = 0xFFFF;

// === Classic dyld_info opcodes (loader.h) ===

const REBASE_OPCODE_MASK: u8 = 0xF0;
const REBASE_IMMEDIATE_MASK: u8 = 0x0F;
const REBASE_OPCODE_DONE: u8 = 0x00;
const REBASE_OPCODE_SET_TYPE_IMM: u8 = 0x10;
const REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB: u8 = 0x20;
const REBASE_OPCODE_ADD_ADDR_ULEB: u8 = 0x30;
const REBASE_OPCODE_ADD_ADDR_IMM_SCALED: u8 = 0x40;
const REBASE_OPCODE_DO_REBASE_IMM_TIMES: u8 = 0x50;
const REBASE_OPCODE_DO_REBASE_ULEB_TIMES: u8 = 0x60;
const REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB: u8 = 0x70;
const REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB: u8 = 0x80;

const REBASE_TYPE_POINTER: u8 = 1;

const BIND_OPCODE_MASK: u8 = 0xF0;
const BIND_IMMEDIATE_MASK: u8 = 0x0F;
const BIND_OPCODE_DONE: u8 = 0x00;
const BIND_OPCODE_SET_DYLIB_ORDINAL_IMM: u8 = 0x10;
const BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB: u8 = 0x20;
const BIND_OPCODE_SET_DYLIB_SPECIAL_IMM: u8 = 0x30;
const BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM: u8 = 0x40;
const BIND_OPCODE_SET_TYPE_IMM: u8 = 0x50;
const BIND_OPCODE_SET_ADDEND_SLEB: u8 = 0x60;
const BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB: u8 = 0x70;
const BIND_OPCODE_ADD_ADDR_ULEB: u8 = 0x80;
const BIND_OPCODE_DO_BIND: u8 = 0x90;
const BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB: u8 = 0xA0;
const BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED: u8 = 0xB0;
const BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB: u8 = 0xC0;

const BIND_TYPE_POINTER: u8 = 1;

// === Section flags / types (mach-o/loader.h) ===

const SECTION_TYPE_MASK: u32 = 0x000000ff;
const S_THREAD_LOCAL_REGULAR: u32 = 0x11;
const S_THREAD_LOCAL_ZEROFILL: u32 = 0x12;
const S_THREAD_LOCAL_VARIABLES: u32 = 0x13;
const S_THREAD_LOCAL_INIT_FUNCTION_POINTERS: u32 = 0x15;

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
struct Section64 {
    sectname: [u8; 16],
    segname: [u8; 16],
    addr: u64,
    size: u64,
    offset: u32,
    align: u32,
    reloff: u32,
    nreloc: u32,
    flags: u32,
    reserved1: u32,
    reserved2: u32,
    reserved3: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LinkeditDataCommand {
    cmd: u32,
    cmdsize: u32,
    dataoff: u32,
    datasize: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DyldInfoCommand {
    cmd: u32,
    cmdsize: u32,
    rebase_off: u32,
    rebase_size: u32,
    bind_off: u32,
    bind_size: u32,
    weak_bind_off: u32,
    weak_bind_size: u32,
    lazy_bind_off: u32,
    lazy_bind_size: u32,
    export_off: u32,
    export_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DyldChainedFixupsHeader {
    fixups_version: u32,
    starts_offset: u32,
    imports_offset: u32,
    symbols_offset: u32,
    imports_count: u32,
    imports_format: u32,
    symbols_format: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DyldChainedStartsInSegment {
    size: u32,
    page_size: u16,
    pointer_format: u16,
    segment_offset: u64,
    max_valid_pointer: u32,
    page_count: u16,
    // page_start: [u16; page_count] follows immediately
}

/// Result of linking the guest image.
pub struct LinkOutcome {
    /// The image's static initializers (C++ constructors, `__attribute__
    /// ((constructor))`), in section order — guest calls the caller owes the
    /// image after linking and before `main`, the way dyld runs them. Left
    /// unrun, any C++ guest whose ctors configure load-bearing state
    /// misbehaves later in ways that no longer point here (`ld` livelocked
    /// in its arena allocator because the zone limits its ctor was to set
    /// read as zero).
    pub initializers: Vec<u64>,
}

/// Link the guest in-process: apply chained fixups or classic dyld_info
/// bind/rebase opcodes, and set up TLV thunks. The `image` must already be
/// mapped (segments at `vmaddr + slide`); we read the load commands
/// directly from that mapping. `image_path` is where the image was loaded
/// from, which its `@loader_path`/`@executable_path` dependencies resolve
/// against. The linked image joins the registry that symbol resolution
/// falls back to, so a later image can bind to this one's exports.
pub fn link(image: &LoadedMachO, image_path: &Path) -> Result<LinkOutcome, Error> {
    let cmds = LoadCommands::parse(image)?;

    // Make every dylib the guest links against visible to `dlsym(RTLD_DEFAULT)`.
    // Chimera's own process is a Rust CLI — it doesn't pull in
    // CoreFoundation, Security, AppKit, etc. on its own — but the guest
    // expects to find symbols from whatever it linked against. `dlopen`
    // makes the host's copy of each library available for lookup.
    load_dependent_dylibs(&cmds, image_path);

    // Registered before binding: an image may bind to its own exports (a
    // weak C++ guard variable coalesces against the image that defines it).
    register_image(image);

    if let Some(fixups) = &cmds.fixups {
        apply_chained_fixups(image, &cmds, fixups)?;
    } else if let Some(info) = &cmds.dyld_info {
        apply_dyld_info(image, &cmds, info)?;
    }

    setup_tlv(image, &cmds)?;

    Ok(LinkOutcome {
        initializers: collect_initializers(image, &cmds),
    })
}

const S_MOD_INIT_FUNC_POINTERS: u32 = 0x9;
const S_INIT_FUNC_OFFSETS: u32 = 0x16;

/// The image's static initializers: `__mod_init_func` absolute pointers
/// (read after the fixup walk rebased them) and the modern `__init_offsets`
/// form, 32-bit offsets from the mach header.
fn collect_initializers(image: &LoadedMachO, cmds: &LoadCommands) -> Vec<u64> {
    let mut funcs = Vec::new();
    for s in &cmds.sections {
        match s.section_type {
            S_MOD_INIT_FUNC_POINTERS => {
                for i in 0..s.size / 8 {
                    let f = unsafe { ptr::read_unaligned((s.runtime_addr + i * 8) as *const u64) };
                    if f != 0 {
                        funcs.push(f);
                    }
                }
            }
            S_INIT_FUNC_OFFSETS => {
                for i in 0..s.size / 4 {
                    let off =
                        unsafe { ptr::read_unaligned((s.runtime_addr + i * 4) as *const u32) };
                    funcs.push(image.mh_addr.wrapping_add(off as u64));
                }
            }
            _ => {}
        }
    }
    funcs
}

fn load_dependent_dylibs(cmds: &LoadCommands, image_path: &Path) {
    let trace = crate::trace::dyld_trace();
    for path in &cmds.dylibs {
        let mut loaded = false;
        for candidate in dylib_candidates(path, &cmds.rpaths, image_path) {
            let Ok(c) = CString::new(candidate.as_str()) else {
                continue;
            };
            let handle = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_LAZY | libc::RTLD_GLOBAL) };
            if trace {
                eprintln!("chimera: dlopen {} -> {:p}", candidate, handle);
            }
            if !handle.is_null() {
                loaded = true;
                break;
            }
        }
        // Failing to load a dependency is not fatal here: the guest may never
        // reference a symbol from it, and if it does, the bind path reports
        // the unresolved symbol by name.
        if trace && !loaded {
            eprintln!("chimera: dlopen {} -> not found", path);
        }
    }
}

/// The paths to try for a dependency, in dyld's order. A dependency recorded
/// with one of the `@` prefixes is relative by design — the install name says
/// *how* to find it, not where it is:
///
/// - `@rpath/x` expands against each `LC_RPATH` entry of the loading image,
///   which is how a program ships private libraries beside itself (a `rustc`
///   binary finding its `librustc_driver`, for one);
/// - `@loader_path/x` and `@executable_path/x` resolve against the directory
///   of the image being linked, which for a main executable is the same
///   directory.
///
/// An `LC_RPATH` entry may itself start with `@loader_path`/`@executable_path`,
/// so those are expanded first. A plain absolute path is used as-is, with the
/// bare leaf name as a last resort so the standard search paths still get a
/// chance.
fn dylib_candidates(install_name: &str, rpaths: &[String], image_path: &Path) -> Vec<String> {
    let dir = image_path
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string());
    let expand = |p: &str| -> String {
        for prefix in ["@loader_path", "@executable_path"] {
            if let Some(rest) = p.strip_prefix(prefix) {
                return format!("{dir}{rest}");
            }
        }
        p.to_string()
    };

    let mut out = Vec::new();
    if let Some(rest) = install_name.strip_prefix("@rpath/") {
        for rpath in rpaths {
            out.push(format!("{}/{}", expand(rpath).trim_end_matches('/'), rest));
        }
        out.push(format!("{dir}/{rest}"));
        out.push(rest.to_string());
    } else {
        out.push(expand(install_name));
    }
    out
}

// === Load-command walking ===

struct FixupsLocation {
    /// Runtime address of the chained-fixups header (in `__LINKEDIT`).
    header_addr: u64,
    /// Bytes available at that address.
    size: u64,
}

struct DyldInfoLocation {
    /// Runtime address of the rebase opcode stream (NULL if rebase_size=0).
    rebase_addr: u64,
    rebase_size: u64,
    bind_addr: u64,
    bind_size: u64,
    lazy_bind_addr: u64,
    lazy_bind_size: u64,
}

struct LoadCommands {
    /// Segments laid out in load-command order. Index matters: chained
    /// fixups and classic bind opcodes identify each segment by its
    /// zero-based ordinal here.
    segments: Vec<Segment>,
    /// Each `LC_LOAD_DYLIB` entry's name (kept for diagnostics; we bind
    /// via `dlsym(RTLD_DEFAULT, ...)` and don't need per-dylib lookup).
    #[allow(dead_code)]
    dylibs: Vec<String>,
    /// Each `LC_RPATH` entry, in order — the search list an `@rpath/…`
    /// dependency is resolved against.
    rpaths: Vec<String>,
    /// `LC_DYLD_CHAINED_FIXUPS` location once translated to a runtime
    /// address.
    fixups: Option<FixupsLocation>,
    /// `LC_DYLD_INFO[_ONLY]` location once translated to runtime addresses.
    dyld_info: Option<DyldInfoLocation>,
    /// All sections, in load-command order. Used for finding TLV sections.
    sections: Vec<Section>,
}

struct Segment {
    #[allow(dead_code)]
    name: String,
    vmaddr: u64,
}

struct Section {
    #[allow(dead_code)]
    sectname: String,
    #[allow(dead_code)]
    segname: String,
    /// Runtime address of the section (vmaddr + slide).
    runtime_addr: u64,
    size: u64,
    /// Section type — the low byte of `flags`.
    section_type: u32,
}

impl LoadCommands {
    fn parse(image: &LoadedMachO) -> Result<Self, Error> {
        let mh_addr = image.mh_addr;
        let header: MachHeader64 = unsafe { ptr::read_unaligned(mh_addr as *const _) };

        let mut linkedit = None;
        let mut segments = Vec::new();
        let mut sections = Vec::new();
        let mut dylibs = Vec::new();
        let mut rpaths = Vec::new();
        let mut fixups_dataoff = None;
        let mut fixups_datasize = 0u32;
        let mut dyld_info: Option<DyldInfoCommand> = None;

        let mut cursor = mh_addr + std::mem::size_of::<MachHeader64>() as u64;
        for _ in 0..header.ncmds {
            let lc: LoadCommand = unsafe { ptr::read_unaligned(cursor as *const _) };
            match lc.cmd {
                LC_SEGMENT_64 => {
                    let seg: SegmentCommand64 = unsafe { ptr::read_unaligned(cursor as *const _) };
                    let seg_name = c_str(&seg.segname);
                    if seg_name == "__LINKEDIT" {
                        linkedit = Some((seg.fileoff, seg.vmaddr));
                    }
                    // Read the nsects Section64 records that follow.
                    let sects_base = cursor + std::mem::size_of::<SegmentCommand64>() as u64;
                    for i in 0..seg.nsects as u64 {
                        let sec_addr = sects_base + i * std::mem::size_of::<Section64>() as u64;
                        let sec: Section64 = unsafe { ptr::read_unaligned(sec_addr as *const _) };
                        let _ = sec.offset; // file offset unused — sections are mapped already
                        sections.push(Section {
                            sectname: c_str(&sec.sectname),
                            segname: c_str(&sec.segname),
                            runtime_addr: sec.addr.wrapping_add(image.slide),
                            size: sec.size,
                            section_type: sec.flags & SECTION_TYPE_MASK,
                        });
                    }
                    segments.push(Segment {
                        name: seg_name,
                        vmaddr: seg.vmaddr,
                    });
                }
                LC_LOAD_DYLIB | LC_LOAD_WEAK_DYLIB | LC_REEXPORT_DYLIB => {
                    let name_off: u32 = unsafe { ptr::read_unaligned((cursor + 8) as *const u32) };
                    let path_addr = cursor + name_off as u64;
                    let max = (lc.cmdsize as u64).saturating_sub(name_off as u64);
                    dylibs.push(read_cstr(path_addr, max));
                }
                LC_RPATH => {
                    let path_off: u32 = unsafe { ptr::read_unaligned((cursor + 8) as *const u32) };
                    let path_addr = cursor + path_off as u64;
                    let max = (lc.cmdsize as u64).saturating_sub(path_off as u64);
                    rpaths.push(read_cstr(path_addr, max));
                }
                LC_DYLD_CHAINED_FIXUPS => {
                    let ld: LinkeditDataCommand =
                        unsafe { ptr::read_unaligned(cursor as *const _) };
                    fixups_dataoff = Some(ld.dataoff as u64);
                    fixups_datasize = ld.datasize;
                }
                LC_DYLD_INFO | LC_DYLD_INFO_ONLY => {
                    let di: DyldInfoCommand = unsafe { ptr::read_unaligned(cursor as *const _) };
                    dyld_info = Some(di);
                }
                _ => {}
            }
            cursor = cursor.wrapping_add(lc.cmdsize as u64);
        }

        // Translate file offsets into runtime addresses for the linkedit
        // metadata blobs. Use the image's known slide rather than
        // recomputing it from segments[0], because segments[0] is typically
        // __PAGEZERO (vmaddr=0) and would give a bogus slide.
        let fixups = match (fixups_dataoff, linkedit) {
            (Some(off), Some((le_off, le_vmaddr))) => {
                let header_addr = (le_vmaddr + (off - le_off)).wrapping_add(image.slide);
                Some(FixupsLocation {
                    header_addr,
                    size: fixups_datasize as u64,
                })
            }
            _ => None,
        };

        let dyld_info_loc = match (dyld_info, linkedit) {
            (Some(di), Some((le_off, le_vmaddr))) => {
                let translate = |off: u32| -> u64 {
                    if off == 0 {
                        0
                    } else {
                        (le_vmaddr + (off as u64 - le_off)).wrapping_add(image.slide)
                    }
                };
                Some(DyldInfoLocation {
                    rebase_addr: translate(di.rebase_off),
                    rebase_size: di.rebase_size as u64,
                    bind_addr: translate(di.bind_off),
                    bind_size: di.bind_size as u64,
                    lazy_bind_addr: translate(di.lazy_bind_off),
                    lazy_bind_size: di.lazy_bind_size as u64,
                })
            }
            _ => None,
        };

        Ok(Self {
            segments,
            dylibs,
            rpaths,
            fixups,
            dyld_info: dyld_info_loc,
            sections,
        })
    }
}

fn c_str(bytes: &[u8; 16]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).unwrap_or("").to_string()
}

fn read_cstr(addr: u64, max: u64) -> String {
    let mut out = Vec::new();
    for i in 0..max {
        let b = unsafe { *((addr + i) as *const u8) };
        if b == 0 {
            break;
        }
        out.push(b);
    }
    String::from_utf8_lossy(&out).into_owned()
}

// === Chained-fixups application ===

fn apply_chained_fixups(
    image: &LoadedMachO,
    cmds: &LoadCommands,
    fixups: &FixupsLocation,
) -> Result<(), Error> {
    if fixups.size < std::mem::size_of::<DyldChainedFixupsHeader>() as u64 {
        return Err(Error::Link("LC_DYLD_CHAINED_FIXUPS data too small".into()));
    }
    let header: DyldChainedFixupsHeader =
        unsafe { ptr::read_unaligned(fixups.header_addr as *const _) };

    if header.fixups_version != 0 {
        return Err(Error::Unsupported(format!(
            "chained-fixups version {}",
            header.fixups_version
        )));
    }
    let imports = parse_imports(fixups, &header)?;
    let bound: Vec<u64> = imports
        .iter()
        .map(|imp| {
            // A missing weak import binds to NULL by design — the image
            // tests the pointer before use.
            match resolve_symbol(&imp.name) {
                Ok(addr) => Ok(addr.wrapping_add(imp.addend as u64)),
                Err(_) if imp.weak => Ok(0),
                Err(e) => Err(e),
            }
        })
        .collect::<Result<_, _>>()?;

    let starts_in_image = fixups.header_addr + header.starts_offset as u64;
    let seg_count: u32 = unsafe { ptr::read_unaligned(starts_in_image as *const u32) };
    let seg_info_offsets = starts_in_image + 4;

    for seg_idx in 0..seg_count {
        let seg_info_off: u32 =
            unsafe { ptr::read_unaligned((seg_info_offsets + (seg_idx * 4) as u64) as *const u32) };
        if seg_info_off == 0 {
            continue;
        }
        let seg_starts_addr = starts_in_image + seg_info_off as u64;
        let segment = cmds.segments.get(seg_idx as usize).ok_or_else(|| {
            Error::Link(format!(
                "chained-fixups references out-of-range segment {}",
                seg_idx
            ))
        })?;
        walk_segment_chains(image, segment, seg_starts_addr, &bound)?;
    }

    Ok(())
}

struct Import {
    name: String,
    weak: bool,
    /// From the ADDEND import formats; always 0 for `DYLD_CHAINED_IMPORT`.
    /// Applied on top of the resolved symbol — a chain entry's own inline
    /// addend, when its pointer format carries one, still adds separately.
    addend: i64,
}

fn parse_imports(
    fixups: &FixupsLocation,
    header: &DyldChainedFixupsHeader,
) -> Result<Vec<Import>, Error> {
    let imports_base = fixups.header_addr + header.imports_offset as u64;
    let symbols_base = fixups.header_addr + header.symbols_offset as u64;
    let mut imports = Vec::with_capacity(header.imports_count as usize);
    for i in 0..header.imports_count as u64 {
        // The three formats share one shape — a lib-ordinal/weak/name-offset
        // record, optionally followed by an addend — at different widths:
        //   DYLD_CHAINED_IMPORT           u32 { lib_ordinal:8,  weak:1, name_offset:23 }
        //   DYLD_CHAINED_IMPORT_ADDEND    the same u32, then i32 addend
        //   DYLD_CHAINED_IMPORT_ADDEND64  u64 { lib_ordinal:16, weak:1, reserved:15,
        //                                       name_offset:32 }, then u64 addend
        let (name_off, weak, addend) = match header.imports_format {
            DYLD_CHAINED_IMPORT => {
                let raw: u32 = unsafe { ptr::read_unaligned((imports_base + i * 4) as *const u32) };
                ((raw >> 9) as u64, raw & 0x100 != 0, 0i64)
            }
            DYLD_CHAINED_IMPORT_ADDEND => {
                let raw: u32 = unsafe { ptr::read_unaligned((imports_base + i * 8) as *const u32) };
                let addend: i32 =
                    unsafe { ptr::read_unaligned((imports_base + i * 8 + 4) as *const i32) };
                ((raw >> 9) as u64, raw & 0x100 != 0, addend as i64)
            }
            DYLD_CHAINED_IMPORT_ADDEND64 => {
                let raw: u64 =
                    unsafe { ptr::read_unaligned((imports_base + i * 16) as *const u64) };
                let addend: u64 =
                    unsafe { ptr::read_unaligned((imports_base + i * 16 + 8) as *const u64) };
                (raw >> 32, raw & 0x1_0000 != 0, addend as i64)
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "chained-fixups imports format {}",
                    other
                )));
            }
        };
        let name_addr = symbols_base + name_off;
        let remaining = fixups.size - (name_addr - fixups.header_addr);
        imports.push(Import {
            name: read_cstr(name_addr, remaining),
            weak,
            addend,
        });
    }
    Ok(imports)
}

fn walk_segment_chains(
    image: &LoadedMachO,
    segment: &Segment,
    seg_starts_addr: u64,
    bound: &[u64],
) -> Result<(), Error> {
    let starts: DyldChainedStartsInSegment =
        unsafe { ptr::read_unaligned(seg_starts_addr as *const _) };
    let pointer_format = starts.pointer_format;
    let page_size = starts.page_size as u64;
    let segment_runtime_base = segment.vmaddr.wrapping_add(image.slide);
    let trace = crate::trace::dyld_trace();
    if trace {
        eprintln!(
            "chimera: walk seg name={:?} vmaddr={:#x} slide={:#x} runtime_base={:#x} fmt={} pages={}",
            segment.name,
            segment.vmaddr,
            image.slide,
            segment_runtime_base,
            pointer_format,
            starts.page_count
        );
    }

    // page_start[] sits immediately after `page_count` in the C layout —
    // at byte offset 22 from the struct base. We can't use Rust's
    // `size_of` here: with `#[repr(C)]` the compiler rounds the struct up
    // to 24 bytes for u64 alignment, but the actual on-disk format packs
    // `page_start[0]` at offset 22 with no padding.
    let page_starts = seg_starts_addr + 22;

    for page_idx in 0..starts.page_count as u64 {
        let page_start: u16 =
            unsafe { ptr::read_unaligned((page_starts + page_idx * 2) as *const u16) };
        if page_start == DYLD_CHAINED_PTR_START_NONE {
            continue;
        }
        let page_base = segment_runtime_base + page_idx * page_size;
        let chain_start = page_base + page_start as u64;
        if trace {
            eprintln!(
                "chimera:   page {}: page_start={:#x} chain_start={:#x}",
                page_idx, page_start, chain_start
            );
        }
        walk_chain(image, pointer_format, chain_start, bound, trace)?;
    }

    Ok(())
}

fn walk_chain(
    image: &LoadedMachO,
    pointer_format: u16,
    chain_start: u64,
    bound: &[u64],
    trace: bool,
) -> Result<(), Error> {
    let mut ptr_addr = chain_start;
    loop {
        let raw: u64 = unsafe { ptr::read_unaligned(ptr_addr as *const u64) };
        let next_delta;
        match pointer_format {
            DYLD_CHAINED_PTR_64 | DYLD_CHAINED_PTR_64_OFFSET => {
                let is_bind = (raw >> 63) & 1 != 0;
                next_delta = ((raw >> 51) & 0xFFF) as u64;

                let resolved = if is_bind {
                    let ordinal = (raw & 0xFF_FFFF) as usize;
                    let addend = ((raw >> 24) & 0xFF) as i64;
                    let addend = if addend >= 0x80 {
                        addend - 0x100
                    } else {
                        addend
                    };
                    let base = *bound.get(ordinal).ok_or_else(|| {
                        Error::Link(format!("bind ordinal {} out of range", ordinal))
                    })?;
                    base.wrapping_add(addend as u64)
                } else {
                    let target = raw & 0xF_FFFF_FFFF;
                    let high8 = ((raw >> 36) & 0xFF) << 56;
                    match pointer_format {
                        DYLD_CHAINED_PTR_64 => (target | high8).wrapping_add(image.slide),
                        DYLD_CHAINED_PTR_64_OFFSET => (target | high8).wrapping_add(image.mh_addr),
                        _ => unreachable!(),
                    }
                };

                if trace {
                    eprintln!(
                        "chimera:     ptr@{:#x}: raw={:#018x} {} → {:#x}, next={}",
                        ptr_addr,
                        raw,
                        if is_bind { "bind" } else { "rebase" },
                        resolved,
                        next_delta
                    );
                }
                unsafe { ptr::write_unaligned(ptr_addr as *mut u64, resolved) };
            }
            DYLD_CHAINED_PTR_ARM64E
            | DYLD_CHAINED_PTR_ARM64E_USERLAND
            | DYLD_CHAINED_PTR_ARM64E_USERLAND24 => {
                // arm64e chain entry: bit 63 = auth, bit 62 = bind, bits 51..61
                // = next (11-bit, 8-byte stride). We resolve targets but write
                // PLAIN pointers — the auth key/diversity are dropped, and the
                // translator strips PAC on use (retaa/braa → xpaci), so unsigned
                // pointers round-trip correctly.
                let is_auth = (raw >> 63) & 1 != 0;
                let is_bind = (raw >> 62) & 1 != 0;
                next_delta = (raw >> 51) & 0x7FF;

                let resolved = if is_bind {
                    // USERLAND24 carries a 24-bit import ordinal; the older
                    // formats a 16-bit one. Auth binds have no addend (those
                    // bits hold diversity); plain binds carry a signed 19-bit
                    // addend at bits 32..50.
                    let ordinal = if pointer_format == DYLD_CHAINED_PTR_ARM64E_USERLAND24 {
                        (raw & 0xFF_FFFF) as usize
                    } else {
                        (raw & 0xFFFF) as usize
                    };
                    let base = *bound.get(ordinal).ok_or_else(|| {
                        Error::Link(format!("bind ordinal {} out of range", ordinal))
                    })?;
                    if is_auth {
                        base
                    } else {
                        let addend = ((raw >> 32) & 0x7_FFFF) as i64;
                        let addend = if addend & 0x4_0000 != 0 {
                            addend - 0x8_0000
                        } else {
                            addend
                        };
                        base.wrapping_add(addend as u64)
                    }
                } else if is_auth {
                    // Auth rebase: 32-bit target. USERLAND targets are offsets
                    // from the mach header.
                    (raw & 0xFFFF_FFFF).wrapping_add(image.mh_addr)
                } else {
                    // Plain rebase: 43-bit target + 8-bit high tag. USERLAND
                    // targets are mach-header offsets; the original ARM64E
                    // format encodes them as vmaddrs (add the slide instead).
                    let target = raw & 0x7FF_FFFF_FFFF;
                    let high8 = ((raw >> 43) & 0xFF) << 56;
                    let value = target | high8;
                    if pointer_format == DYLD_CHAINED_PTR_ARM64E {
                        value.wrapping_add(image.slide)
                    } else {
                        value.wrapping_add(image.mh_addr)
                    }
                };

                if trace {
                    eprintln!(
                        "chimera:     ptr@{:#x}: raw={:#018x} {}{} → {:#x}, next={}",
                        ptr_addr,
                        raw,
                        if is_auth { "auth " } else { "" },
                        if is_bind { "bind" } else { "rebase" },
                        resolved,
                        next_delta
                    );
                }
                unsafe { ptr::write_unaligned(ptr_addr as *mut u64, resolved) };
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "chained pointer format {:#x}",
                    other
                )));
            }
        }

        if next_delta == 0 {
            break;
        }
        // The `next` delta counts strides: 8 bytes for arm64e chains, 4 for the
        // 64-bit formats.
        let stride = match pointer_format {
            DYLD_CHAINED_PTR_ARM64E
            | DYLD_CHAINED_PTR_ARM64E_USERLAND
            | DYLD_CHAINED_PTR_ARM64E_USERLAND24 => 8,
            _ => 4,
        };
        ptr_addr += next_delta * stride;
    }
    Ok(())
}

// === Classic `LC_DYLD_INFO[_ONLY]` opcode interpretation ===

fn apply_dyld_info(
    image: &LoadedMachO,
    cmds: &LoadCommands,
    info: &DyldInfoLocation,
) -> Result<(), Error> {
    let trace = crate::trace::dyld_trace();
    if info.rebase_size > 0 {
        apply_rebase_stream(image, cmds, info.rebase_addr, info.rebase_size, trace)?;
    }
    if info.bind_size > 0 {
        apply_bind_stream(image, cmds, info.bind_addr, info.bind_size, false, trace)?;
    }
    // Eagerly process lazy binds too. We have no `dyld_stub_binder` to
    // fall back to, so any lazy stub the guest ends up taking would panic.
    if info.lazy_bind_size > 0 {
        apply_bind_stream(
            image,
            cmds,
            info.lazy_bind_addr,
            info.lazy_bind_size,
            true,
            trace,
        )?;
    }
    Ok(())
}

fn read_uleb(stream: &[u8], pos: &mut usize) -> Result<u64, Error> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        if *pos >= stream.len() {
            return Err(Error::Link("ULEB128 ran off end of stream".into()));
        }
        let b = stream[*pos];
        *pos += 1;
        result |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(Error::Link("ULEB128 too large".into()));
        }
    }
    Ok(result)
}

fn read_sleb(stream: &[u8], pos: &mut usize) -> Result<i64, Error> {
    let mut result = 0i64;
    let mut shift = 0u32;
    let last_byte;
    loop {
        if *pos >= stream.len() {
            return Err(Error::Link("SLEB128 ran off end of stream".into()));
        }
        let byte = stream[*pos];
        *pos += 1;
        result |= ((byte & 0x7f) as i64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            last_byte = byte;
            break;
        }
        if shift >= 64 {
            return Err(Error::Link("SLEB128 too large".into()));
        }
    }
    // Sign-extend if the sign bit of the last 7-bit group is set.
    if shift < 64 && (last_byte & 0x40) != 0 {
        result |= -(1i64) << shift;
    }
    Ok(result)
}

fn segment_runtime_base(cmds: &LoadCommands, seg_idx: usize) -> Result<u64, Error> {
    cmds.segments
        .get(seg_idx)
        .map(|s| s.vmaddr)
        .ok_or_else(|| Error::Link(format!("segment ordinal {} out of range", seg_idx)))
}

fn apply_rebase_stream(
    image: &LoadedMachO,
    cmds: &LoadCommands,
    addr: u64,
    size: u64,
    trace: bool,
) -> Result<(), Error> {
    let stream = unsafe { std::slice::from_raw_parts(addr as *const u8, size as usize) };
    let mut pos = 0usize;
    let mut rebase_type = REBASE_TYPE_POINTER;
    let mut cursor: u64 = 0; // runtime address of next pointer to rebase
    const POINTER_SIZE: u64 = 8;

    let do_rebase_one = |cursor: u64, rebase_type: u8| -> Result<(), Error> {
        if rebase_type != REBASE_TYPE_POINTER {
            return Err(Error::Unsupported(format!(
                "rebase type {} (only POINTER)",
                rebase_type
            )));
        }
        let raw: u64 = unsafe { ptr::read_unaligned(cursor as *const u64) };
        let slid = raw.wrapping_add(image.slide);
        unsafe { ptr::write_unaligned(cursor as *mut u64, slid) };
        Ok(())
    };

    while pos < stream.len() {
        let byte = stream[pos];
        pos += 1;
        let opcode = byte & REBASE_OPCODE_MASK;
        let imm = byte & REBASE_IMMEDIATE_MASK;
        match opcode {
            REBASE_OPCODE_DONE => break,
            REBASE_OPCODE_SET_TYPE_IMM => {
                rebase_type = imm;
            }
            REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB => {
                let seg_off = read_uleb(stream, &mut pos)?;
                let base = segment_runtime_base(cmds, imm as usize)?;
                cursor = base.wrapping_add(image.slide).wrapping_add(seg_off);
                if trace {
                    eprintln!(
                        "chimera: rebase set_seg seg={} off={:#x} cursor={:#x}",
                        imm, seg_off, cursor
                    );
                }
            }
            REBASE_OPCODE_ADD_ADDR_ULEB => {
                let delta = read_uleb(stream, &mut pos)?;
                cursor = cursor.wrapping_add(delta);
            }
            REBASE_OPCODE_ADD_ADDR_IMM_SCALED => {
                cursor = cursor.wrapping_add(imm as u64 * POINTER_SIZE);
            }
            REBASE_OPCODE_DO_REBASE_IMM_TIMES => {
                for _ in 0..imm {
                    do_rebase_one(cursor, rebase_type)?;
                    cursor = cursor.wrapping_add(POINTER_SIZE);
                }
            }
            REBASE_OPCODE_DO_REBASE_ULEB_TIMES => {
                let n = read_uleb(stream, &mut pos)?;
                for _ in 0..n {
                    do_rebase_one(cursor, rebase_type)?;
                    cursor = cursor.wrapping_add(POINTER_SIZE);
                }
            }
            REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB => {
                let delta = read_uleb(stream, &mut pos)?;
                do_rebase_one(cursor, rebase_type)?;
                cursor = cursor.wrapping_add(POINTER_SIZE).wrapping_add(delta);
            }
            REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB => {
                let count = read_uleb(stream, &mut pos)?;
                let skip = read_uleb(stream, &mut pos)?;
                for _ in 0..count {
                    do_rebase_one(cursor, rebase_type)?;
                    cursor = cursor.wrapping_add(POINTER_SIZE).wrapping_add(skip);
                }
            }
            other => {
                return Err(Error::Unsupported(format!("rebase opcode {:#x}", other)));
            }
        }
    }
    Ok(())
}

fn apply_bind_stream(
    image: &LoadedMachO,
    cmds: &LoadCommands,
    addr: u64,
    size: u64,
    is_lazy: bool,
    trace: bool,
) -> Result<(), Error> {
    let stream = unsafe { std::slice::from_raw_parts(addr as *const u8, size as usize) };
    let mut pos = 0usize;
    let mut symbol_name = String::new();
    let mut bind_type = BIND_TYPE_POINTER;
    let mut addend: i64 = 0;
    let mut cursor: u64 = 0;
    const POINTER_SIZE: u64 = 8;

    let do_bind_one =
        |symbol_name: &str, bind_type: u8, addend: i64, cursor: u64| -> Result<(), Error> {
            if symbol_name.is_empty() {
                return Err(Error::Link("bind without symbol name".into()));
            }
            if bind_type != BIND_TYPE_POINTER {
                return Err(Error::Unsupported(format!(
                    "bind type {} (only POINTER)",
                    bind_type
                )));
            }
            let target = resolve_symbol(symbol_name)?.wrapping_add(addend as u64);
            unsafe { ptr::write_unaligned(cursor as *mut u64, target) };
            if crate::trace::dyld_trace() {
                eprintln!(
                    "chimera: bind {} @ {:#x} -> {:#x}",
                    symbol_name, cursor, target
                );
            }
            Ok(())
        };

    while pos < stream.len() {
        let byte = stream[pos];
        pos += 1;
        let opcode = byte & BIND_OPCODE_MASK;
        let imm = byte & BIND_IMMEDIATE_MASK;
        match opcode {
            BIND_OPCODE_DONE => {
                // In lazy bind streams, DONE just terminates one stub —
                // there are more stubs to follow in the same blob.
                if !is_lazy {
                    break;
                }
            }
            BIND_OPCODE_SET_DYLIB_ORDINAL_IMM
            | BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB
            | BIND_OPCODE_SET_DYLIB_SPECIAL_IMM => {
                // We resolve everything via dlsym(RTLD_DEFAULT,...) and
                // ignore which dylib a symbol nominally belongs to.
                if opcode == BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB {
                    let _ = read_uleb(stream, &mut pos)?;
                }
            }
            BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM => {
                let _flags = imm;
                let mut s = Vec::new();
                while pos < stream.len() && stream[pos] != 0 {
                    s.push(stream[pos]);
                    pos += 1;
                }
                if pos >= stream.len() {
                    return Err(Error::Link("symbol name ran off bind stream".into()));
                }
                pos += 1; // skip NUL
                symbol_name = String::from_utf8_lossy(&s).into_owned();
            }
            BIND_OPCODE_SET_TYPE_IMM => {
                bind_type = imm;
            }
            BIND_OPCODE_SET_ADDEND_SLEB => {
                addend = read_sleb(stream, &mut pos)?;
            }
            BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB => {
                let seg_off = read_uleb(stream, &mut pos)?;
                let base = segment_runtime_base(cmds, imm as usize)?;
                cursor = base.wrapping_add(image.slide).wrapping_add(seg_off);
                if trace {
                    eprintln!(
                        "chimera: bind set_seg seg={} off={:#x} cursor={:#x}",
                        imm, seg_off, cursor
                    );
                }
            }
            BIND_OPCODE_ADD_ADDR_ULEB => {
                let delta = read_uleb(stream, &mut pos)?;
                cursor = cursor.wrapping_add(delta);
            }
            BIND_OPCODE_DO_BIND => {
                do_bind_one(&symbol_name, bind_type, addend, cursor)?;
                cursor = cursor.wrapping_add(POINTER_SIZE);
            }
            BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB => {
                let delta = read_uleb(stream, &mut pos)?;
                do_bind_one(&symbol_name, bind_type, addend, cursor)?;
                cursor = cursor.wrapping_add(POINTER_SIZE).wrapping_add(delta);
            }
            BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED => {
                do_bind_one(&symbol_name, bind_type, addend, cursor)?;
                cursor = cursor.wrapping_add(POINTER_SIZE * (1 + imm as u64));
            }
            BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB => {
                let count = read_uleb(stream, &mut pos)?;
                let skip = read_uleb(stream, &mut pos)?;
                for _ in 0..count {
                    do_bind_one(&symbol_name, bind_type, addend, cursor)?;
                    cursor = cursor.wrapping_add(POINTER_SIZE).wrapping_add(skip);
                }
            }
            other => {
                return Err(Error::Unsupported(format!("bind opcode {:#x}", other)));
            }
        }
    }
    Ok(())
}

// === Thread-local-variable setup ===

/// Per-TLV-image state kept alive for the lifetime of the process. Indexed
/// by the pthread key we hand to each descriptor — the thunk callback
/// finds the image via `pthread_getspecific`'s key and reads the template
/// from this map to materialise a fresh per-thread block on first touch.
struct TlvImage {
    /// Bytes to memcpy into a freshly-allocated per-thread block. Length =
    /// `total_size`. Concatenates the contents of all `S_THREAD_LOCAL_REGULAR`
    /// sections (initialised data), padded with zeros where
    /// `S_THREAD_LOCAL_ZEROFILL` sections sit.
    template: Vec<u8>,
    /// The image's `S_THREAD_LOCAL_INIT_FUNCTION_POINTERS` entries — guest
    /// functions (C++ `thread_local` constructors) dyld would call once per
    /// thread, right after materialising that thread's block. Slid guest
    /// addresses: the section contents were rebased with the rest of the
    /// image before `setup_tlv` read them.
    init_funcs: Vec<u64>,
}

static TLV_IMAGES: Mutex<Option<HashMap<libc::pthread_key_t, TlvImage>>> = Mutex::new(None);

#[repr(C)]
struct TlvDescriptor {
    thunk: u64,
    key: libc::pthread_key_t,
    offset: u64,
}

/// The TLV slot lookup behind the thunk: hand back the address of the
/// thread-local slot, allocating and initialising the per-thread block on the
/// thread's first access to this image. On that first touch the image's
/// initializer functions are returned to the caller, which must run them as
/// guest code before the guest resumes — the block is installed in the TSD
/// slot first, exactly as dyld installs before initialising, so an
/// initializer's own thread-local accesses find it rather than recursing.
/// What the dispatcher must do to finish installing a freshly allocated
/// thread-local block: call `pthread_setspecific(key, block)` *as guest code*.
/// `(function, key, block)`, or `None` when the block already existed.
pub type TlvInstall = Option<(u64, u64, u64)>;

fn tlv_slot(desc: &TlvDescriptor) -> (*mut u8, Vec<u64>, TlvInstall) {
    // The block is kept in the *guest* thread's TSD array, not through
    // `pthread_setspecific`, which would put it in the host thread's. The two
    // differ for every spawned guest thread (the guest's libpthread owns its
    // own pthread struct, reached through the virtualized `TPIDRRO_EL0`), and
    // it is the guest's array that its own teardown walks — so this is what
    // makes libpthread's TSD cleanup reach `chimera_tlv_finalize` and run the
    // thread's destructors at the right moment.
    let slot = guest_tsd_slot(desc.key);
    // Read the guest's TSD array without trusting it. A thread far enough
    // into teardown has had its pthread struct unmapped while translated code
    // still reaches the TLV thunk — the guest survives that (the fault is
    // delivered to its handler, as it would be natively), but a raw
    // dereference here faults the *runtime*, on a thread with no fault
    // handling of its own, and takes the process down.
    let mut raw = [0u8; 8];
    if !crate::sys::mmap::copy_from_guest(slot as u64, &mut raw) {
        return (ptr::null_mut(), Vec::new(), None);
    }
    let mut block = u64::from_ne_bytes(raw) as *mut u8;
    let mut init_funcs = Vec::new();
    let mut install = None;
    if block.is_null() {
        let guard = TLV_IMAGES.lock().unwrap();
        let images = guard.as_ref().expect("TLV image map not initialised");
        let image = images
            .get(&desc.key)
            .expect("TLV access for unregistered key");
        let len = image.template.len();
        // Allocate the per-thread block from the system allocator. The
        // thread's own TSD cleanup clears the slot at exit; the block itself
        // is deliberately not freed, since a destructor the guest registered
        // may still be about to run against it.
        let layout = std::alloc::Layout::from_size_align(len, 16).unwrap();
        block = unsafe { std::alloc::alloc(layout) };
        if block.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        unsafe {
            ptr::copy_nonoverlapping(image.template.as_ptr(), block, len);
        }
        // Write the slot first so a thread-local access made by the installer
        // or an initializer finds the block instead of recursing; the call
        // below is what makes libpthread track it. Fault-safe for the same
        // reason as the read.
        if !crate::sys::mmap::copy_to_guest(slot as u64, &(block as u64).to_ne_bytes()) {
            return (ptr::null_mut(), Vec::new(), None);
        }
        install = Some((setspecific_addr(), desc.key, block as u64));
        init_funcs = image.init_funcs.clone();
    }
    (
        unsafe { block.add(desc.offset as usize) },
        init_funcs,
        install,
    )
}

/// The guest-callable `pthread_setspecific`, resolved once.
fn setspecific_addr() -> u64 {
    const RTLD_DEFAULT: *mut core::ffi::c_void = (-2isize) as *mut core::ffi::c_void;
    unsafe { libc::dlsym(RTLD_DEFAULT, c"pthread_setspecific".as_ptr()) as u64 }
}

/// First-touch handler for a TLV access. Each `__thread_vars` descriptor's
/// `thunk` field is overwritten to point here during link-time setup; the
/// guest's `blr x8` (where `x8 = desc->thunk`) targets this routine with the
/// descriptor pointer in `x0` per the Darwin TLV ABI. The branch never
/// actually lands here: the dispatcher recognizes the target address and
/// services it through [`run_tlv_thunk`], which also carries the initializer
/// list this C signature cannot.
unsafe extern "C" fn chimera_tlv_get_addr(desc: *const TlvDescriptor) -> *mut u8 {
    tlv_slot(unsafe { &*desc }).0
}

/// Address of this guest thread's TSD slot for `key`.
///
/// Guest threads reach their thread-local storage through the `guest_tsd`
/// base the dispatcher keeps per thread — the guest's own pthread struct for a
/// spawned thread, and the host thread's for the initial one, which is the
/// same base its translated code sees when it reads `TPIDRRO_EL0`.
fn guest_tsd_slot(key: libc::pthread_key_t) -> *mut u64 {
    let ctx = crate::arch::dispatch::current_ctx();
    let base = if ctx.is_null() {
        read_host_tsd_base()
    } else {
        unsafe { (*ctx).guest_tsd }
    };
    unsafe { (base as *mut u64).add(key as usize) }
}

/// The calling host thread's own TSD base, for a lookup made outside any
/// guest context.
fn read_host_tsd_base() -> u64 {
    let v: u64;
    unsafe { std::arch::asm!("mrs {v}, tpidrro_el0", v = out(reg) v, options(nomem, nostack)) };
    v & !7
}

/// Marker for "this thread's guest thread-local destructors are due now".
///
/// It is registered as the TLV key's `pthread_key_create` destructor, so the
/// guest's own teardown reaches it: `_pthread_exit` runs the per-thread TSD
/// cleanup — as translated guest code — before it marks the thread finished
/// and lets a joiner go. The dispatcher recognizes a call to this address and
/// runs the guest's registered destructors there (see `Thread::escape`), which
/// is the only point late enough that the thread is really exiting and early
/// enough that `pthread_join` has not yet returned to the other side.
///
/// Natively it does nothing. The only caller that reaches the real function is
/// the host thread's own libpthread teardown, after the guest is gone and the
/// destructor list is already drained.
unsafe extern "C" fn chimera_tlv_finalize(_block: *mut libc::c_void) {}

/// Address of [`chimera_tlv_finalize`], for the dispatcher to match against.
pub fn tlv_finalize_addr() -> u64 {
    chimera_tlv_finalize as unsafe extern "C" fn(*mut libc::c_void) as usize as u64
}

/// Native address of the TLV first-touch thunk. `setup_tlv` rewrites every TLV
/// descriptor to call this (see the `thunk` binding there), so the guest reaches
/// it with `blr desc->thunk`. It is Chimera's own compiled Rust — the arm64
/// dispatcher recognizes a guest branch to this address and runs it natively
/// rather than translating Chimera's runtime as guest code (which would corrupt
/// the returned thread-local pointer).
pub fn tlv_thunk_addr() -> u64 {
    chimera_tlv_get_addr as unsafe extern "C" fn(*const TlvDescriptor) -> *mut u8 as usize as u64
}

/// Run the TLV first-touch thunk natively for a guest call: `desc` is the
/// guest's `x0` (the descriptor pointer); returns the thread-local slot
/// address the guest expects back in `x0`, plus the image's initializer
/// functions when this call materialised the thread's block — the dispatcher
/// runs those as guest calls before resuming the guest.
pub fn run_tlv_thunk(desc: u64) -> (u64, Vec<u64>, TlvInstall) {
    let (slot, init_funcs, install) = tlv_slot(unsafe { &*(desc as *const TlvDescriptor) });
    (slot as u64, init_funcs, install)
}

fn setup_tlv(image: &LoadedMachO, cmds: &LoadCommands) -> Result<(), Error> {
    // First, build the TLV image template by walking the thread-local
    // *data* sections in address order. Each descriptor's `offset` field
    // (set by the static linker) is the byte offset into this template.
    let mut tls_sections: Vec<&Section> = cmds
        .sections
        .iter()
        .filter(|s| {
            s.section_type == S_THREAD_LOCAL_REGULAR || s.section_type == S_THREAD_LOCAL_ZEROFILL
        })
        .collect();
    tls_sections.sort_by_key(|s| s.runtime_addr);

    let mut total_size = 0u64;
    if let (Some(first), Some(last)) = (tls_sections.first(), tls_sections.last()) {
        // Total spans from the first section's start to the last section's
        // end. The descriptor's `offset` field is relative to the first
        // section, so we anchor at `first.runtime_addr`.
        total_size = last.runtime_addr + last.size - first.runtime_addr;
    }

    let mut template = vec![0u8; total_size as usize];
    let base = tls_sections.first().map(|s| s.runtime_addr).unwrap_or(0);
    for s in &tls_sections {
        if s.section_type == S_THREAD_LOCAL_REGULAR && s.size > 0 {
            // Read initial data from the loaded segment image — it was
            // already memcpy'd by macho.rs from the file at load time.
            let off = (s.runtime_addr - base) as usize;
            let bytes =
                unsafe { std::slice::from_raw_parts(s.runtime_addr as *const u8, s.size as usize) };
            template[off..off + s.size as usize].copy_from_slice(bytes);
        }
        // ZEROFILL sections are already zero in `template`.
    }

    // Find __thread_vars sections (descriptors). For each descriptor:
    //   1. Allocate one pthread key for the image (shared across all
    //      descriptors — they distinguish themselves by `offset`).
    //   2. Set desc->thunk = chimera_tlv_get_addr.
    //   3. Set desc->key = pthread_key.
    //   (`desc->offset` is already correct — the static linker set it.)
    let var_sections: Vec<&Section> = cmds
        .sections
        .iter()
        .filter(|s| s.section_type == S_THREAD_LOCAL_VARIABLES)
        .collect();
    if var_sections.is_empty() {
        return Ok(());
    }

    // The per-thread initializer functions (C++ `thread_local` constructors),
    // read out of the mapped section after the fixup walk rebased them.
    let mut init_funcs: Vec<u64> = Vec::new();
    for s in cmds
        .sections
        .iter()
        .filter(|s| s.section_type == S_THREAD_LOCAL_INIT_FUNCTION_POINTERS)
    {
        for i in 0..s.size / 8 {
            let f = unsafe { ptr::read_unaligned((s.runtime_addr + i * 8) as *const u64) };
            if f != 0 {
                init_funcs.push(f);
            }
        }
    }

    // The key carries a destructor so libpthread's per-thread TSD cleanup
    // calls it during the guest's own exit path — the moment dyld would have
    // run this image's thread-local destructors, and crucially *before*
    // libpthread releases a joiner. See [`chimera_tlv_finalize`].
    let mut key: libc::pthread_key_t = 0;
    let rc = unsafe { libc::pthread_key_create(&mut key, Some(chimera_tlv_finalize)) };
    if rc != 0 {
        return Err(Error::io(
            "pthread_key_create",
            std::io::Error::from_raw_os_error(rc),
        ));
    }

    {
        let mut guard = TLV_IMAGES.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        map.insert(
            key,
            TlvImage {
                template,
                init_funcs,
            },
        );
    }

    let thunk = chimera_tlv_get_addr as unsafe extern "C" fn(*const TlvDescriptor) -> *mut u8
        as usize as u64;

    for vs in var_sections {
        let count = vs.size / std::mem::size_of::<TlvDescriptor>() as u64;
        for i in 0..count {
            let desc_addr = vs.runtime_addr + i * std::mem::size_of::<TlvDescriptor>() as u64;
            unsafe {
                ptr::write_unaligned(desc_addr as *mut u64, thunk);
                ptr::write_unaligned((desc_addr + 8) as *mut libc::pthread_key_t, key);
                // offset is left as-is — already set by the static linker.
            }
            if crate::trace::dyld_trace() {
                let offset: u64 = unsafe { ptr::read_unaligned((desc_addr + 16) as *const u64) };
                eprintln!(
                    "chimera: TLV desc@{:#x} thunk={:#x} key={} offset={:#x}",
                    desc_addr, thunk, key, offset
                );
            }
        }
    }
    let _ = image; // future: track per-image cleanup
    Ok(())
}

// === Guest image registry and guest dlopen ===

/// Every image Chimera's own linker has loaded and linked — the main
/// executable and any guest-`dlopen`'d module the host dyld could not take.
/// Symbol resolution falls back to these after `dlsym`: the host dyld knows
/// nothing of images Chimera mapped, so a bind naming a symbol one of them
/// exports (a zsh module's flat-namespace reference to the `zsh` binary,
/// an image's bind to its own weak C++ guard variable) resolves here.
/// Reset at each execve install: the old image's exports die with it.
static IMAGES: Mutex<Vec<GuestImage>> = Mutex::new(Vec::new());

#[derive(Clone, Copy)]
struct GuestImage {
    mh_addr: u64,
    slide: u64,
}

fn register_image(image: &LoadedMachO) {
    let mut images = IMAGES.lock().unwrap();
    if !images.iter().any(|i| i.mh_addr == image.mh_addr) {
        images.push(GuestImage {
            mh_addr: image.mh_addr,
            slide: image.slide,
        });
    }
}

/// Forget every linked image. Called at an execve install, before the new
/// main image is linked: the request's binds must not resolve against the
/// image being replaced.
pub fn reset_images() {
    IMAGES.lock().unwrap().clear();
}

/// Whether `handle` is an image Chimera's linker loaded (its mach-header
/// address doubles as the `dlopen` handle the guest holds).
pub fn is_guest_handle(handle: u64) -> bool {
    IMAGES.lock().unwrap().iter().any(|i| i.mh_addr == handle)
}

/// Look `mach_name` up across every registered image, in load order — the
/// main executable first, dyld's flat-namespace search order.
fn lookup_guest_images(mach_name: &str) -> Option<u64> {
    let images = IMAGES.lock().unwrap().clone();
    images.iter().find_map(|i| image_symbol(i, mach_name))
}

/// Service a guest `dlsym(handle, name)` on a Chimera-loaded image. `name`
/// is the bare C name the caller passed; the Mach-O symbol table carries
/// the leading underscore.
pub fn dlsym_guest(handle: u64, name: &str) -> Option<u64> {
    let image = IMAGES
        .lock()
        .unwrap()
        .iter()
        .find(|i| i.mh_addr == handle)
        .copied()?;
    image_symbol(&image, &format!("_{name}")).or_else(|| image_symbol(&image, name))
}

/// A guest `dlsym(RTLD_DEFAULT, name)` that the host `dlsym` answered NULL
/// for: the name may be a guest image's export.
pub fn dlsym_guest_global(name: &str) -> Option<u64> {
    lookup_guest_images(&format!("_{name}")).or_else(|| lookup_guest_images(name))
}

const LC_SYMTAB: u32 = 0x2;

#[repr(C)]
#[derive(Clone, Copy)]
struct SymtabCommand {
    cmd: u32,
    cmdsize: u32,
    symoff: u32,
    nsyms: u32,
    stroff: u32,
    strsize: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Nlist64 {
    n_strx: u32,
    n_type: u8,
    n_sect: u8,
    n_desc: u16,
    n_value: u64,
}

const N_STAB: u8 = 0xe0;
const N_TYPE: u8 = 0x0e;
const N_SECT: u8 = 0x0e;
const N_EXT: u8 = 0x01;

/// Find `mach_name` among an image's external defined symbols, by scanning
/// its `LC_SYMTAB` in place. A linear scan per lookup is fine at this
/// altitude: the table is consulted only after the host `dlsym` has already
/// answered NULL, which for a healthy image is a handful of binds.
fn image_symbol(image: &GuestImage, mach_name: &str) -> Option<u64> {
    let header: MachHeader64 = unsafe { ptr::read_unaligned(image.mh_addr as *const _) };
    let mut symtab: Option<SymtabCommand> = None;
    let mut linkedit: Option<(u64, u64)> = None;
    let mut cursor = image.mh_addr + std::mem::size_of::<MachHeader64>() as u64;
    for _ in 0..header.ncmds {
        let lc: LoadCommand = unsafe { ptr::read_unaligned(cursor as *const _) };
        match lc.cmd {
            LC_SYMTAB => {
                symtab = Some(unsafe { ptr::read_unaligned(cursor as *const _) });
            }
            LC_SEGMENT_64 => {
                let seg: SegmentCommand64 = unsafe { ptr::read_unaligned(cursor as *const _) };
                if c_str(&seg.segname) == "__LINKEDIT" {
                    linkedit = Some((seg.fileoff, seg.vmaddr));
                }
            }
            _ => {}
        }
        cursor = cursor.wrapping_add(lc.cmdsize as u64);
    }
    let (symtab, (le_off, le_vmaddr)) = (symtab?, linkedit?);
    let translate = |off: u32| (le_vmaddr + (off as u64 - le_off)).wrapping_add(image.slide);
    let syms = translate(symtab.symoff);
    let strs = translate(symtab.stroff);
    for i in 0..symtab.nsyms as u64 {
        let n: Nlist64 = unsafe {
            ptr::read_unaligned((syms + i * std::mem::size_of::<Nlist64>() as u64) as *const _)
        };
        if n.n_type & N_STAB != 0 || n.n_type & N_EXT == 0 || n.n_type & N_TYPE != N_SECT {
            continue;
        }
        let name = read_cstr(
            strs + n.n_strx as u64,
            (symtab.strsize as u64).saturating_sub(n.n_strx as u64),
        );
        if name == mach_name {
            return Some(n.n_value.wrapping_add(image.slide));
        }
    }
    None
}

/// A guest-`dlopen`'d image Chimera's linker brought in.
pub struct LoadedModule {
    /// The mach-header address, doubling as the guest's `dlopen` handle.
    pub handle: u64,
    /// The image's static initializers — guest calls the caller owes the
    /// image before the guest resumes, dyld's contract for `dlopen`.
    pub initializers: Vec<u64>,
    /// The image's whole reservation, for the caller to record as a guest
    /// region.
    pub region: (u64, usize),
}

/// Service a guest `dlopen` the host dyld refused: load the image with
/// Chimera's own loader, link it against the host's libraries *and* the
/// registered guest images, and hand back the mach-header address as the
/// guest's handle. The image is never unloaded; `dlclose` on it is a
/// no-op, matching the leak-the-address-space rule the exec teardown
/// already follows.
pub fn dlopen_guest(path: &Path) -> Result<LoadedModule, Error> {
    let image = load_macho(path)?;
    let outcome = link(&image, path)?;
    Ok(LoadedModule {
        handle: image.mh_addr,
        initializers: outcome.initializers,
        region: image.region,
    })
}

// === Symbol resolution ===

fn resolve_symbol(mach_name: &str) -> Result<u64, Error> {
    // Mach-O symbols carry a leading underscore that the C ABI doesn't
    // use; `dlsym` wants the bare C name.
    let bare = mach_name.strip_prefix('_').unwrap_or(mach_name);
    let cname = CString::new(bare).map_err(|_| {
        Error::Link(format!(
            "symbol name {:?} contains an interior NUL",
            mach_name
        ))
    })?;

    const RTLD_DEFAULT: *mut libc::c_void = -2isize as *mut libc::c_void;
    let addr = unsafe { libc::dlsym(RTLD_DEFAULT, cname.as_ptr()) };
    if addr.is_null() {
        // `dyld_stub_binder` only exists inside dyld itself; it's not
        // visible to `dlsym`. Any classic-format binary names it for the
        // TLV thunk slots and for lazy-bind stubs. We rewrite the TLV
        // descriptors after binding (see `setup_tlv`), so a placeholder is
        // fine for them. Lazy binds are processed eagerly with this same
        // resolve, so a non-TLV slot landing on `dyld_stub_binder` would
        // be a real broken bind — but in practice the lazy-bind stream
        // names the actual target (e.g. `_printf`), not the binder helper.
        if mach_name == "dyld_stub_binder" || mach_name == "_dyld_stub_binder" {
            return Ok(0); // placeholder; `setup_tlv` overwrites the slot
        }
        // The host dyld cannot see images Chimera mapped, so a symbol a
        // guest image exports resolves here: the flat-namespace case (a zsh
        // module referencing the zsh binary) and self-binds (an image's own
        // coalesced weak definitions).
        if let Some(addr) = lookup_guest_images(mach_name) {
            return Ok(addr);
        }
        return Err(Error::Link(format!(
            "unresolved symbol {:?} (host dlsym returned NULL)",
            mach_name
        )));
    }
    Ok(addr as u64)
}
