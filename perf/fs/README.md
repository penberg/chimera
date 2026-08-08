# Filesystem overhead

`fs_bench.c` measures what running a filesystem-heavy workload under Chimera's
userspace VFS costs over native. `run.sh` builds it and runs each operation
four ways against the same file — native, under `chimera run`, and native
through two `chimera mount` FUSE views of an empty change-set over the live
host, one uncached (the default) and one with `--cache 60`. The Chimera
routes serve the same merged view; they differ in how the operations reach
the VFS (DBT syscall interception in-process, or the kernel's FUSE protocol)
and in what the kernel may cache.

```
./run.sh
CHIMERA=../../target/release/chimera ./run.sh [iters-per-round] [path] [rounds]
```

Each cell is measured over `rounds` (default 10) self-timed rounds of
`iters-per-round` operations within one process — one warm-up, many
independent samples — and reported as mean±sd ns/op with ratios of means
over native. The raw per-round samples land in `build/samples.csv` as
`route,op,round,ns_per_op`, which is what a plot with error bars should
consume; the table is only the summary.

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

Read the `fuse` column knowing the default mount advertises zero-TTL entries
and attributes — coherence with the live host, bought with round trips.
Every `stat`/`open` re-looks-up (and permission-checks) each in-mount path
component, so those figures are several protocol round trips each and scale
with depth. `read` serves its data from the kernel page cache, but the zero
attribute TTL still forces a GETATTR revalidation per call, so its figure is
one round trip, not a pure cache hit. The `fuse-c60` column is the other end
of the trade: with `--cache 60` the kernel answers repeated resolutions from
its own caches, so what remains is dentry/page-cache machinery, at the price
of staleness (bounded by the TTL) against changes made outside the mount.
