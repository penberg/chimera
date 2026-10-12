# strace

A reimplementation of `strace(1)` built on Chimera's `SystemCalls` trait.

The handler runs in Chimera's own address space alongside the guest, so it
can read guest pointers (e.g. the file path passed to `openat`) with a plain
dereference. Each syscall is logged to standard error in the same `name(args)
= ret` format `strace(1)` prints.

A guest speaks its host kernel's system-call interface, and Linux and Darwin
do not share one — numbers, argument conventions, and struct layouts all
differ — so there is one tracer per host: `strace_linux.rs` decodes the
x86-64 Linux calls with the rich argument formatters shown below, and
`strace_darwin.rs` names the arm64 BSD calls and Mach traps, printing the
first three argument registers in hex and decoding the `getattrlist`
family's attribute bitmaps. `main.rs` picks the one for the host it is built
on.

## Build and run

    cargo run --example strace -- <program> [args...]

For the conformance test:

    cargo run --example strace -- testing/conformance/entry/a.out hello

Sample output (abridged):

    brk(0x0)                                 = 0x55e36bf2c000
    openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC, 0o0) = 3
    fstat(3, 0x7ffe1c8f7a30)                 = 0
    mmap(0x0, 98731, PROT_READ, MAP_PRIVATE, 3, 0) = 0x7f1d2a5cd000
    openat(AT_FDCWD, "/lib64/libc.so.6", O_RDONLY|O_CLOEXEC, 0o0) = 3
    ...
    arch_prctl(0x1002, 0x7f1d2a829740)       = 0
    set_tid_address(0x7f1d2a829a10)          = 2039123
    exit_group(42)                           = ?

## How it works

The example implements the `chimera::SystemCalls` trait:

```rust
impl SystemCalls for Strace {
    fn post_syscall(&mut self, call: &SystemCall) {
        let name = syscall_name(call.number);
        let line = format!("{}({})", name, format_args(name, call));
        match call.result() {
            Some(result) => eprintln!("{:<40} = {}", line, format_ret(result)),
            None => eprintln!("{:<40} = ?", line),
        }
    }
}
```

`pre_syscall()` and `post_syscall()` run for every guest syscall, even
when Chimera services it internally. `do_syscall()` is only used for
delegated syscalls and defaults to forwarding them with
`host_syscall(call)`.

The Chimera runtime intercepts `arch_prctl(ARCH_SET_FS, ...)` inside
`syscall` so the guest's TLS setup doesn't disturb the runtime's own
FS-base register, handles the `mmap` family itself so guest mapping
bookkeeping stays authoritative, and terminates `exit`/`exit_group`
itself instead of forwarding them to the host kernel. Logging in
`post_syscall()` keeps those syscalls visible without making them
interceptable.
