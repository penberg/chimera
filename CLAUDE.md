This is the Chimera repository — a light-weight sandboxing runtime that runs unmodified binaries through same-ISA dynamic binary translation: ELF on Linux/x86-64, and Mach-O on Darwin/arm64. Same-ISA means the guest always shares the host's instruction set; there is no cross-ISA emulation anywhere in the design. Every system call the guest makes is routed to an embedder-supplied handler (default `Passthrough`), so sandboxing policy lives in the embedder, not the runtime.

The full design lives in `ARCHITECTURE.md`. Read it before making architectural decisions.

## Building and Running Chimera

- **Build**: `make build` (or `cargo build`) — produces `chimera` in the cargo target directory.
- **Run a guest binary**: `chimera run <program> [args...]`, e.g. `cargo run -q -- run /bin/ls -l`.
  - `--code-cache-size <MiB>` sets the translated-code cache capacity (default 256).
- **Example embedders** (each has a README with details):
  - `cargo run --example strace -- <program> [args...]` — syscall tracer built on the `SystemCalls` trait.
  - `cargo run --example sandbox -- --allow '<regex>' [--allow ...] <program> [args...]` — allowlist syscall sandbox; anything unmatched returns `-EPERM`.

The cargo target directory may be shared across worktrees (`scripts/cargo-target-dir` prints it), so another checkout's build can clobber the `chimera` binary — rebuild before trusting a run.

## Project Layout

Cargo virtual workspace with two member crates:

- `runtime/` — the `chimera-runtime` package: the DBT runtime. `lib.rs` plus `dispatch.rs`, `elf.rs`, `exec.rs`, `syscall.rs`, `trampoline.rs`, `trampoline.S`, `translate.rs`. Example embedders in `runtime/examples/` (`sandbox/`, `strace/`). The lib name is `chimera`, so consumers write `use chimera::Sandbox` regardless of the package name.
- `cli/` — the `chimera-cli` package: thin front end (`main.rs`, `opts.rs`) producing the `chimera` binary. Intentionally thin — a stable harness for the conformance suite and performance work, not a product surface.
- `testing/` — conformance tests, driven by `testing/lit.py` (invokes `target/debug/chimera run`).
- `research/` — reference papers.

The workspace is shaped to accept a future `chimera-linux` crate alongside `chimera-runtime` — the natural home for userspace Linux semantics (`Vfs`, `Net`, …) layered on top of the DBT runtime.

## Tests

- **Conformance**: `make conformance` builds Chimera and runs each test under it; `make conformance-native` runs the same tests directly without Chimera.
- The runner is `testing/lit.py`, modeled after LLVM's LIT: each test source carries `// RUN:` directives the runner expands and executes. Tests live under `testing/conformance/`, organized by topic.
- **Where a test runs is decided by where it lives, never by a marker inside it.** Every directory is named `<topic>[-<condition>…]`, with conditions drawn from `linux`, `darwin`, `x86`, `arm64`: `signals/` runs everywhere, `signals-linux/` only on Linux, `isolation-linux-x86/` only on x86-64 Linux. Topic names carry no hyphen, so an unrecognized segment is an error rather than a directory that quietly runs everywhere. Adding a test that only makes sense on one host or one ISA means naming its directory, not annotating the test. The one marker that remains is `// XFAIL: <condition>…`, for a test that *should* pass and doesn't — a tracked bug, which no directory can express because the same directory runs both natively and under Chimera. See `testing/conformance/README.md`.
- **CI locally**: `make ci` runs the GitHub Actions workflow (`.github/workflows/ci.yml` — rustfmt, clippy, build, test, conformance) with [Agent CI](https://agent-ci.dev/). It uses a custom runner image, `.github/agent-ci.Dockerfile`; keep its pinned toolchain in sync with `rust-toolchain.toml`.

## Coding Style

### Imports

Group `use` statements by crate — `std`/`core` first, external crates next, local modules last — separated by blank lines. Collapse shared prefixes into a single `use crate::{a, b, c}` line:

```rust
use std::{ffi::OsString, path::PathBuf, process::ExitCode};

use chimera::Sandbox;

use opts::{Command, Opts};
```

### Comments

A comment must say something the code cannot: a kernel contract, an invariant, a non-obvious why. Say it once, on the item that owns it, and let every other site stand bare. Never restate a rationale at call sites, narrate what the next line does, or argue that the code is correct — that last kind is a review comment, and it goes stale the moment the review ends.

### Visibility

Don't use `pub(crate)`. Internal items live in private modules — `mod foo;`, not `pub mod foo;` — so plain `pub` there is already unreachable from outside the crate. If plain `pub` would genuinely widen the public API surface, move the item into a private module instead.

### Naming

The project name is **Chimera** in prose, comments, and doc comments. Lowercase `chimera` only inside backticks for code identifiers — the binary name, the crate path, `libchimera`, `chimera.h`, C-API symbols like `chimera_sandbox_t`.

### CLI Option Parsing

Options live in `opts.rs`, derived with [`argh`](https://github.com/google/argh). `Opts` carries a `Command` subcommand enum so new tools (`translate`, …) slot in alongside `run` without reshaping the top level. Do not switch to `clap` or another arg-parsing crate. Note: this version of `argh` accepts options only as `--name value`, not `--name=value`.

## Commit Messages

Write the subject as `<area>: <change>` when one subsystem owns the change (`runtime`, `cli`, `testing`, or `perf`). Keep the text after the colon lowercase, phrase it as what the patch does, and omit the trailing period. For example: `cli: run the current shell by default`. Use a concise unscoped imperative subject for a repository-wide change that has no honest subsystem prefix. Do not add a pull-request number; GitHub supplies that when it creates a merge commit.

Every commit must have a body that makes sense without the diff or review conversation. Follow the subject with a blank line and flowing prose, hard-wrapped at 78 columns. Begin with the concrete failure or limitation that motivated the patch, then explain the change and the behavior or invariant it establishes. Call out meaningful tradeoffs and finish with the regression coverage or other verification. Prefer specific paths, commands, error values, and before/after behavior over a generic summary of the diff. Never omit the body on the grounds that the change is small or self-evident: the purpose of the log is to preserve why the change was worth making.

## Writing for ARCHITECTURE.md

Flowing Sun/DEC-style prose paragraphs. Bullets are reserved for genuine enumerations — auxv field names, lifecycle stages, layout entries — never as a substitute for explanatory sentences. Use numerals for technical quantities (`64-byte`, `16 bytes`). Citations are short: full first names, italicized venue abbreviation (e.g., *VEE '12*), DOI on its own at the end.
