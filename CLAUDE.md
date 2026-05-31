# CLAUDE.md

Notes for Claude when working on this repository.

## What this is

Chimera is a light-weight sandboxing runtime that runs unmodified Linux x86-64 binaries through same-ISA dynamic binary translation. It is a sandbox: not a fault-injection tool, not a deterministic-execution harness, not a cross-ISA translator. The system-call layer is an embedder-supplied handler with a `Passthrough` default; Chimera itself bakes no policy in.

The full design lives in `ARCHITECTURE.md`. Read it before making architectural decisions.

## Project layout

The repository is a Cargo virtual workspace with two member crates. `runtime/` is the `chimera-runtime` package — Chimera's same-ISA DBT runtime, with `lib.rs` and its submodules (`dispatch.rs`, `elf.rs`, `exec.rs`, `syscall.rs`, `trampoline.rs`, `trampoline.S`, `translate.rs`) and the example embedders under `runtime/examples/` (`sandbox/`, `strace/`). The lib name is `chimera`, so consumers write `use chimera::Sandbox` regardless of the package name. `cli/` is the `chimera-cli` package — a thin front end (`main.rs`, `opts.rs`) that depends on `chimera-runtime` and produces the `chimera` binary. The binary is intentionally thin: its main job is to be a stable harness for the conformance suite and performance work, not a product surface. Conformance tests live under `testing/` and are driven by `testing/lit.py`, which invokes `target/debug/chimera run`. Reference papers are in `research/`. The workspace is shaped to accept a future `chimera-linux` crate alongside `chimera-runtime` — the natural home for userspace Linux semantics (`Vfs`, `Net`, …) layered on top of the DBT runtime.

## Coding style

### Imports

Group `use` statements by crate. Within a group, collapse paths sharing a prefix into a single `use crate::{a, b, c}` line. Separate groups with a blank line:

```rust
use std::{ffi::OsString, path::PathBuf, process::ExitCode};

use chimera::Sandbox;

use opts::{Command, Opts};
```

`std` and `core` first, external crates next, local modules last.

### Visibility

Don't use `pub(crate)`. Most internal items live in private modules — declared `mod foo;`, not `pub mod foo;` — so a plain `pub` inside one of those modules is already unreachable from outside the crate. The `(crate)` is noise. If plain `pub` would genuinely widen the public API surface, move the item into a private module rather than reach for `pub(crate)`.

### Naming

The project name is **Chimera** in prose, comments, and doc comments. Lowercase `chimera` is correct only inside backticks for code identifiers — the binary name, the crate path, `libchimera`, `chimera.h`, and C-API symbols like `chimera_sandbox_t`.

### CLI option parsing

CLI options live in `opts.rs` and are derived with [`argh`](https://github.com/google/argh). The top-level `Opts` carries a `Command` subcommand enum so new tools (`translate`, …) can slot in alongside `run` without reshaping the top level. Do not switch to `clap` or another arg-parsing crate; the surface is small and `argh` is the chosen tool. Note that this version of `argh` accepts options only as `--name value`, not `--name=value`.

## Writing for ARCHITECTURE.md

`ARCHITECTURE.md` is written as flowing Sun/DEC-style prose paragraphs. Bullets are reserved for genuine enumerations — auxv field names, named lifecycle stages, layout entries — never as a substitute for explanatory sentences. Use numerals for technical quantities (`64-byte`, `16 bytes`), not spelled-out forms. Citations are short: full first names, italicized venue abbreviation (e.g., *VEE '12*), DOI on its own at the end.

## Tests

`make conformance` builds Chimera and runs each test under it; `make conformance-native` runs the same tests directly without Chimera. The runner is `testing/lit.py`, modeled after LLVM's LIT: each test source carries one or more `// RUN:` directives that the runner expands and executes. Tests live under `testing/conformance/` and are organized by topic.

`make ci` runs the GitHub Actions workflow (`.github/workflows/ci.yml` — rustfmt, clippy, build, test, conformance) locally with [Agent CI](https://agent-ci.dev/), reproducing what GitHub runs on push. It uses a custom runner image, `.github/agent-ci.Dockerfile`, that adds the build toolchain and rustup the default minimal runner lacks; keep its pinned toolchain in sync with `rust-toolchain.toml`.
