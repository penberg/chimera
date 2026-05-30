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
    fn handle(&mut self, call: &mut SystemCall) {
        let name = syscall_name(call.number);
        if self.allowed.iter().any(|r| r.is_match(name)) {
            let ret = syscall(call);
            call.set_return(ret);
        } else {
            call.set_return(-(libc::EPERM as i64));
        }
    }
}
```

`syscall(call)` issues the syscall on the host kernel;
`call.set_return(v)` writes `v` into the guest's `rax` on resume.
Negative values in `[-4095, -1]` are interpreted by guest libc as
errno-encoded errors — `-EPERM` here makes the call look exactly like a
seccomp-enforced denial.

Unknown syscall numbers serialize as `syscall_<n>` so a user-supplied
regex can never let one through by accident.
