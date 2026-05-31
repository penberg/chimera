# chimera-runtime

The same-ISA dynamic binary translation runtime that powers Chimera.
This crate is the library; the `chimera` command-line tool in the
sibling [`cli/`](../cli) crate is one consumer of it.

Use it to build sandboxes, syscall tracers, fault injectors,
deterministic execution engines, and other tools that need to observe
or modify program behavior without source code or recompilation.

## Embedding

Construct a `Sandbox`, install a `SystemCalls` handler, and run:

```rust
use chimera::{Sandbox, SystemCall, SystemCalls, host_syscall};

struct Passthrough;

impl SystemCalls for Passthrough {
    fn do_syscall(&mut self, call: &mut SystemCall) {
        call.set_result(host_syscall(call));
    }
}

let mut sandbox = Sandbox::new("/bin/echo")?;
sandbox.args(["hello"]).system_calls(Passthrough);
sandbox.run()?;
```

`pre_syscall()` and `post_syscall()` can observe every guest syscall,
including the few Chimera services itself for correctness.
`do_syscall()` is only invoked for delegated syscalls; the default
implementation forwards to the host kernel via `host_syscall(call)`,
which returns a `SyscallResult` — `Ok(value)` on success,
`Error(errno)` on failure. `call.set_result(r)` writes `r` back to the
guest in the host's return-ABI (Linux: `-errno` in `rax`; Darwin:
`errno` plus the carry flag). A handler is free to do anything in
between — log, deny, rewrite arguments, fabricate a return value, or
skip the kernel entirely.

If no handler is installed, the runtime uses a `Passthrough` default
equivalent to the snippet above.

## Examples

* [sandbox](examples/sandbox/README.md) — an allowlist-based
  system-call sandbox: each syscall is matched against a set of regex
  patterns, matches forward to the host kernel, anything else returns
  `-EPERM` to the guest.
* [strace](examples/strace/README.md) — reimplements `strace(1)` on
  top of `SystemCalls`; logs every guest syscall in `strace(1)` format
  from `post_syscall()`, including Chimera-owned syscalls.

Run an example with:

```
cargo run --example sandbox -- --allow '^write$' /bin/echo hi
cargo run --example strace -- /bin/echo hi
```

## Design

See [ARCHITECTURE.md](../ARCHITECTURE.md).

## License

MIT.
