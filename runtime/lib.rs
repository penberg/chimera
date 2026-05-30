//! Chimera: a light-weight sandboxing runtime using same-ISA dynamic
//! binary translation.

use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

mod arch;
mod sys;
mod syscall;

pub use syscall::{Passthrough, SyscallResult, SystemCall, SystemCalls, syscall};

/// A sandboxed guest program, configured but not yet running.
pub struct Sandbox {
    program: PathBuf,
    args: Vec<OsString>,
    envs: Option<Vec<(OsString, OsString)>>,
    handler: Box<dyn SystemCalls>,
}

impl Sandbox {
    /// Create a new sandbox for the given program.
    pub fn new(program: impl AsRef<Path>) -> Result<Self, Error> {
        arch::init()?;
        Ok(Self {
            program: program.as_ref().to_path_buf(),
            args: Vec::new(),
            envs: None,
            handler: Box::new(Passthrough),
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

    /// Install a system-call handler. Replaces the default [`Passthrough`].
    pub fn system_calls<H: SystemCalls + 'static>(&mut self, handler: H) -> &mut Self {
        self.handler = Box::new(handler);
        self
    }

    /// Run the guest. Returns when the guest issues `exit_group` (or `exit`),
    /// with the requested exit code; returns an error on setup failure.
    pub fn run(&mut self) -> Result<ExitStatus, Error> {
        let handler = std::mem::replace(&mut self.handler, Box::new(Passthrough));
        let code = sys::exec::execv(&self.program, &self.args, self.envs.as_deref(), handler)?;
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

    /// Chimera is itself running on a host platform it doesn't support.
    #[error("unsupported host platform {os}-{arch}: Chimera runtime supports Linux and macOS")]
    UnsupportedHost {
        os: &'static str,
        arch: &'static str,
    },
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
