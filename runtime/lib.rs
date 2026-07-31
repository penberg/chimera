//! Chimera: a light-weight sandboxing runtime using same-ISA dynamic
//! binary translation.

use std::{
    ffi::{OsStr, OsString},
    io,
    path::{Path, PathBuf},
};

mod arch;
mod process;
mod sys;
mod syscall;

pub use syscall::{ForkHold, Passthrough, SyscallResult, SystemCall, SystemCalls};

pub use sys::linux::syscall::host_syscall;

pub use sys::linux::{
    DirEntry, Errno, File, FileType, HostFs, Mode, MountFlags, Namespace, OpenFlags, OverlayFs,
    Personality, RenameFlags, Stat, StatFs, Timespec, Vfs, WriteResult,
};

/// The tooling side of the workspace delta format, for inspecting and
/// applying a delta from outside a sandbox: the marker predicates, the
/// origin record a copy-up leaves behind, and the applied-state writers
/// `fs apply` advances after each successful host operation. The format
/// itself is owned by this crate; the CLI's `chimera fs` subcommands
/// consume these rather than reimplementing the on-disk knowledge.
pub mod delta {
    pub use crate::sys::linux::{
        Origin, is_applied, is_opaque, is_whiteout, mark_applied, origin, record_origin,
    };
}

/// Default capacity of the translated-code cache, in bytes. One cache is
/// shared by every guest thread, so a multithreaded guest translates more
/// blocks into it than a single-threaded one and runs a higher risk of
/// exhausting it. Size it generously: the region is `mmap`'d lazily, so unused
/// capacity costs virtual address space, not resident memory. It stays well
/// under 2 GiB, so every intra-cache `rel32` displacement still fits in an
/// `i32`.
pub const DEFAULT_CODE_CACHE_SIZE: usize = 256 * 1024 * 1024;

/// Upper bound on the translated-code cache capacity, in bytes. Every
/// intra-cache `rel32` branch displacement is measured within the cache, so
/// the cache must stay under 2 GiB for the displacement to fit in an `i32`.
pub const MAX_CODE_CACHE_SIZE: usize = i32::MAX as usize;

/// Whether the host supports memory protection keys for the translated-code
/// cache.
pub fn mpk_enabled() -> bool {
    arch::mpk_enabled()
}

/// A sandboxed guest program, configured but not yet running.
pub struct Sandbox {
    program: PathBuf,
    args: Vec<OsString>,
    envs: Option<Vec<(OsString, OsString)>>,
    handler: Box<dyn SystemCalls>,
    code_cache_size: usize,
}

impl Sandbox {
    /// Create a new sandbox for the given program.
    pub fn new(program: impl AsRef<Path>) -> Result<Self, Error> {
        arch::init()?;
        sys::mmap::init()?;
        Ok(Self {
            program: program.as_ref().to_path_buf(),
            args: Vec::new(),
            envs: None,
            handler: Box::new(Passthrough),
            code_cache_size: DEFAULT_CODE_CACHE_SIZE,
        })
    }

    /// Append a single argument to the guest's argv.
    pub fn arg(&mut self, arg: impl AsRef<OsStr>) -> &mut Self {
        self.args.push(arg.as_ref().to_os_string());
        self
    }

    /// Append multiple arguments to the guest's argv.
    pub fn args<I, S>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        for a in args {
            self.args.push(a.as_ref().to_os_string());
        }
        self
    }

    /// Set an environment variable for the guest. The first call to `env`
    /// stops the guest from inheriting the host environment; subsequent
    /// calls add to the explicit set.
    pub fn env(&mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> &mut Self {
        self.envs
            .get_or_insert_with(Vec::new)
            .push((key.as_ref().to_os_string(), value.as_ref().to_os_string()));
        self
    }

    /// Set the translated-code cache capacity in bytes. Replaces the default
    /// [`DEFAULT_CODE_CACHE_SIZE`]. Must be nonzero and at most
    /// [`MAX_CODE_CACHE_SIZE`]; an out-of-range size is reported by
    /// [`Sandbox::run`].
    pub fn code_cache_size(&mut self, bytes: usize) -> &mut Self {
        self.code_cache_size = bytes;
        self
    }

    /// Install a system-call handler. Replaces the default [`Passthrough`].
    pub fn system_calls<H: SystemCalls + 'static>(&mut self, handler: H) -> &mut Self {
        self.handler = Box::new(handler);
        self
    }

    /// Run the guest. Returns when the guest issues `exit_group` (or `exit`),
    /// with the requested exit code; returns an error on setup failure.
    pub fn run(&mut self) -> Result<ExitStatus, Error> {
        // Reject a bad cache size before anything is mapped: the guest image
        // and stack go up before the thread (and its cache) exists and are
        // recorded for cleanup only afterwards, so a late rejection would
        // leak them in a long-lived embedder.
        if self.code_cache_size == 0 || self.code_cache_size > MAX_CODE_CACHE_SIZE {
            return Err(Error::io(
                "code cache size",
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "must be nonzero and under 2 GiB",
                ),
            ));
        }
        let handler = std::mem::replace(&mut self.handler, Box::new(Passthrough));
        let code = sys::exec::execv(
            &self.program,
            &self.args,
            self.envs.as_deref(),
            handler,
            self.code_cache_size,
        )?;
        Ok(ExitStatus { code })
    }
}

/// How the guest terminated.
pub struct ExitStatus {
    code: i32,
}

impl ExitStatus {
    pub fn code(&self) -> i32 {
        self.code
    }
    pub fn success(&self) -> bool {
        self.code == 0
    }
}

/// Errors that can occur when constructing or running a [`Sandbox`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The guest tried to execute unmapped (or otherwise unreadable) memory:
    /// translation could not read any bytes at this guest address. The run
    /// loop turns it into a guest `SIGSEGV` there — the fault the guest would
    /// have taken natively on the fetch.
    #[error("guest fetch fault at {0:#x}")]
    BadAccess(u64),

    /// The guest binary is malformed: bad magic, truncated header, an
    /// out-of-range offset, missing required segment, and so on.
    #[error("{0}")]
    BadBinary(String),

    /// The translated-code cache is full.
    #[error("code cache exhausted")]
    CodeCacheExhausted,

    /// Host-OS I/O failure (file read, `mmap`, `mprotect`, `arch_prctl`, …).
    /// `op` describes what Chimera was trying to do; `source` is the
    /// underlying `errno` mapped through `std::io::Error`.
    #[error("{op}: {source}")]
    Io {
        op: String,
        #[source]
        source: std::io::Error,
    },

    /// In-process dynamic linking failed: malformed fixup metadata,
    /// out-of-range ordinal, missing symbol, truncated LEB128, …
    #[error("link: {0}")]
    Link(String),

    /// Same-ISA dynamic binary translation failed for a guest block.
    #[error("translate: {0}")]
    Translate(String),

    /// The guest binary is well-formed but uses a feature Chimera doesn't
    /// yet implement (an unsupported load command, pointer format,
    /// bind/rebase opcode, …).
    #[error("unsupported guest feature: {0}")]
    Unsupported(String),
}

impl Error {
    /// Helper: wrap a `std::io::Error` with operation context.
    pub fn io(op: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            op: op.into(),
            source,
        }
    }

    /// Helper: tag the current `errno` with operation context.
    pub fn last_os_error(op: impl Into<String>) -> Self {
        Self::io(op, std::io::Error::last_os_error())
    }
}
