# Filesystem overhead

`fs_bench.c` measures what running a filesystem-heavy workload under Chimera's
userspace VFS costs over native. `run.sh` builds it and runs each operation
native and under `chimera run`, printing ns/op and the chimera/native ratio.

```
./run.sh
CHIMERA=../../target/release/chimera ./run.sh [iters] [path]
```

Use a release build; a debug build inflates the overhead.

## Modes

`fs_bench` decomposes where the time goes:

| mode   | isolates |
|--------|----------|
| `stat` | the path resolver — the per-component walk that stats every component of the path, so the cost scales with path depth |
| `open` | resolver plus fd-table install |
| `read` | the per-fd dispatch alone (`File::pread`, fd lookup, offset) — no path resolution |

`read` stays close to native, so the fd path is cheap; `stat`/`open` carry the
resolver cost. Pass a deeper or shallower `path` to see the resolver scale with
depth.
