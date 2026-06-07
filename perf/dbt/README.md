# Steady-state DBT microbenchmarks

Each `perf-<name>.c` is a self-contained, self-timed microbenchmark that
isolates one guest execution pattern Chimera's dynamic binary translation has
to handle. `run.sh` builds each one and runs it native and under Chimera,
printing ns/op and the chimera/native ratio.

```
./run.sh
CHIMERA=../target/release/chimera ./run.sh
```

## Why self-timed

The hot loop is bracketed with `CLOCK_MONOTONIC` (see `perf.h`), so process
startup and the one-time translation of every block on the path to the loop are
excluded from the measurement. What each benchmark reports is the steady-state
per-operation overhead of executing already-translated code — the cost that
remains once the code cache is warm. Cold-start and translation latency are a
separate concern, measured by a separate tool.

## Workloads

| workload        | isolates |
|-----------------|----------|
| `loop`          | direct back-edge linking — a tight ALU loop, no exits |
| `direct`        | conditional direct-branch linking (guest mispredicts paid both ways) |
| `indirect`      | inline indirect-branch lookup, few targets |
| `indirect-mega` | indirect lookup overflowing the inline cache into the hash fallback |
| `ret`           | monomorphic return — one call site, inline lookup always hits |
| `callsites`     | one callee reached from many sites, so its return target varies |
| `syscall`       | code-cache exit + trampoline + XSAVE context switch (the sandbox boundary) |
| `signal`        | guest signal virtualization (frame build + handler round-trip) |
| `mem`           | control: dependent loads run native, so this should read ~1.0x |

`syscall` and `signal` use smaller default iteration counts than the in-cache
workloads because each op is far more expensive; pass an explicit count as the
first argument to override (e.g. `./build/syscall 50000000`).
