<p align="center">
  <img src=".github/assets/logo.png" alt="Chimera" width="320">
</p>

<h1 align="center">Chimera</h1>

<p align="center">
  <em><strong>Run any command in a zero-setup sandbox.</strong></em>
</p>

<p align="center">
  <a href="https://github.com/penberg/chimera/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/penberg/chimera/ci.yml?branch=main&style=flat-square&logo=github&label=CI"></a>
  <a href="https://www.rust-lang.org"><img alt="Built with Rust" src="https://img.shields.io/badge/built%20with-rust-orange?style=flat-square&logo=rust"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green?style=flat-square"></a>
</p>

---

Chimera runs unmodified binaries through same-ISA dynamic binary translation.
It ships as two pieces: a `chimera` command-line tool for wrapping a process
at the shell, and a Rust library for embedding the same runtime in your own
program. Library users specify a system-call handler that decide what every
guest syscall does: forward it to the host kernel, log it, or virtualize it.

Chimera currently supports **Linux/x86** target with work-in-progress port to **Darwin/arm64**.

## Command-line tool

Install:

```
cargo install --path cli
```

Run any command:

```
chimera run /bin/echo hello
```

The default handler forwards every syscall to the host kernel, so the
wrapped process behaves like a native one.

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
        let ret = host_syscall(call);
        call.set_return(ret);
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

## License

This project is licensed under the [MIT license].

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in Chimera by you, shall be licensed as MIT, without any additional
terms or conditions.

[MIT license]: LICENSE.md
