// A cheap syscall is expensive relative to translated ALU, so a far smaller N.
#define PERF_DEFAULT_N 10000000ull
#include "perf.h"

#include <unistd.h>
#include <sys/syscall.h>

// One cheap syscall per iteration. Unlike every other workload, this one leaves
// the code cache on each op: exit translated code -> trampoline -> full guest
// context save (XSAVE) -> embedder syscall handler -> restore -> re-enter. For
// a sandbox this boundary-crossing cost is the headline overhead — none of the
// in-cache workloads touch it. SYS_getpid is invoked directly (not via the
// glibc wrapper, which has historically cached the result) to guarantee a real
// syscall every iteration.
static uint64_t perf_run(uint64_t n, uint64_t seed) {
    uint64_t acc = seed;
    for (uint64_t i = 0; i < n; i++)
        acc += (uint64_t)syscall(SYS_getpid);
    return acc;
}

PERF_MAIN("syscall")
