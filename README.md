<p align="center">
  <img src=".github/assets/logo.png" alt="Chimera" width="320">
</p>

<h1 align="center">Chimera</h1>

<p align="center">
  <em><strong>Sandbox untrusted code with safe access to the host.</strong></em>
</p>

<p align="center">
  <a href="https://github.com/penberg/chimera/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/penberg/chimera/ci.yml?branch=main&style=flat-square&logo=github&label=CI"></a>
  <a href="https://www.rust-lang.org"><img alt="Built with Rust" src="https://img.shields.io/badge/built%20with-rust-orange?style=flat-square&logo=rust"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green?style=flat-square"></a>
</p>

---

Chimera is a userspace sandbox for code you don't fully trust — a natural fit
for coding agents that run arbitrary, generated commands on a machine they share
with you. It confines what a process can do to the host while still letting it
reach the host's files and tools, so the code stays useful without being free to
corrupt the system. Because it runs as an ordinary program, Chimera needs no VM,
no container, and no special hardware, kernel features, or privileges — it works
wherever your code already runs.

It achieves this by running unmodified binaries through same-ISA dynamic binary
translation, intercepting each guest system call so the sandbox decides what it
does. Chimera ships as two pieces: a `chimera` command-line tool for wrapping a
process at the shell, and a Rust library for embedding the same runtime in your
own program. Embedders supply a system-call handler that decides what every guest
syscall does — forward it to the host kernel, log it, or virtualize it.

Chimera currently supports **Linux/x86** target with work-in-progress port to **Darwin/arm64**.

## Getting started

Install the `chimera` command-line tool from GitHub:

```
cargo install --git https://github.com/penberg/chimera chimera-cli
```

Then wrap any command:

```
chimera run /bin/echo hello
```

The wrapped process sees the host filesystem exactly as it is, but every write,
rename, and delete it makes lands in a private change-set rather than on the
host. Every run keeps its change-set under a short id: inspect it with
`chimera fs diff`, resume it with `chimera run --in <id>`, branch it with
`--from <id>`, browse it as a FUSE mount with `chimera mount <id> <dir>`, or
adopt it onto the host with `chimera fs apply`. `--rm`
discards it on exit, and `--unsafe` runs without one, the process mutating the
host directly.

With no command, `chimera run` starts `/bin/bash` with a compact prompt naming
its filesystem id. `--unsafe` replaces the id with `unsafe`; pass
`--no-prompt` to leave the normal Bash prompt unchanged.

From a checkout, install with `cargo install --path cli` instead. The
[manual](MANUAL.md) covers the `chimera` command in full.

## Library

The `chimera` crate exposes the runtime the CLI is built on.
Implement the `SystemCalls` trait to decide how each guest syscall is
handled:

```rust
use chimera::{Sandbox, SystemCall, SystemCalls, host_syscall};

struct Tracer;

impl SystemCalls for Tracer {
    fn handle(&mut self, call: &mut SystemCall) {
        eprintln!("syscall {}", call.number);
        call.set_result(host_syscall(call));
    }
}

fn main() -> Result<(), chimera::Error> {
    let mut sandbox = Sandbox::new("/bin/echo")?;
    sandbox.args(["hello"]).system_calls(Tracer);
    sandbox.run()?;
    Ok(())
}
```

Worked examples live under [`runtime/examples/`](runtime/examples):

- [`sandbox`](runtime/examples/sandbox/README.md) — an allowlist-based
  system-call sandbox.
- [`strace`](runtime/examples/strace/README.md) — reimplements
  `strace(1)` on top of the same trait.

## Design

See [ARCHITECTURE.md](ARCHITECTURE.md).

## Publications

- Pekka Enberg and Ashwin Rao (2026). Towards Sandboxing Untrusted Agents in
  Userspace. _Tech report._ [[PDF](https://penberg.org/papers/penberg-chimera.pdf)]

## License

This project is licensed under the [MIT license].

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in Chimera by you, shall be licensed as MIT, without any additional
terms or conditions.

[MIT license]: LICENSE.md
