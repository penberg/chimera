# sandbox

An allowlist-based system-call sandbox: every guest syscall is matched
against a set of regex patterns; matches forward to the host kernel,
anything else returns `-EPERM` to the guest. This is the simplest
non-trivial use of Chimera's `SystemCalls` trait — a starting point for
embedders that want to restrict what a workload can ask the kernel to
do.

## Build and run

    cargo run --example sandbox -- \
        --allow '<regex>' [--allow ...] <program> [args...]

`--allow` is repeatable and the patterns are unanchored (use `^...$`
for an exact match). The first denial of each distinct syscall is
logged to stderr; further denials of the same call are suppressed so
the output stays readable while a policy is being built up.

A workable starter policy for a dynamically linked binary looks
something like:

    cargo run --example sandbox -- \
        --allow '^(read|write|close|fstat|lseek|mmap|mprotect|munmap|brk|arch_prctl|set_tid_address|set_robust_list|rseq|prlimit64|getrandom|exit_group|openat|newfstatat|pread64)$' \
        /bin/echo hi

Less is more revealing — start with `--allow '^write$'` and let the
denial log tell you what the program asks for next.

## How it works

```rust
struct Allowlist { allowed: Vec<Regex>, /* ... */ }

impl SystemCalls for Allowlist {
    fn do_syscall(&mut self, call: &mut SystemCall) {
        let name = syscall_name(call.number);
        if self.allowed.iter().any(|r| r.is_match(name)) {
            call.set_result(host_syscall(call));
        } else {
            call.set_result(SyscallResult::Error(libc::EPERM));
        }
    }
}
```

`host_syscall(call)` issues the syscall on the host kernel and returns a
`SyscallResult` — `Ok(value)` on success, `Error(errno)` on failure.
`call.set_result(r)` writes `r` back to the guest in the host's
return-ABI, so `Error(EPERM)` here makes the denial look exactly like
a seccomp-enforced one.

Only delegated syscalls reach `do_syscall()`: Chimera-owned syscalls
like `exit_group` and the virtualized `arch_prctl` cases are handled by
the runtime before `post_syscall()` observers run.

Unknown syscall numbers serialize as `syscall_<n>` so a user-supplied
regex can never let one through by accident.
