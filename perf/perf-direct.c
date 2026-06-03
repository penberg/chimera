#include "perf.h"

// Data-dependent conditional branches over a pseudo-random stream. Both edges
// of each conditional are direct branches that linking keeps in the cache, so
// this isolates conditional-branch linking; the mispredictions are the guest's
// own cost and are paid identically native and translated.
static uint64_t perf_run(uint64_t n, uint64_t seed) {
    uint64_t acc = 0, x = seed | 1;
    for (uint64_t i = 0; i < n; i++) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        if (x & 1)
            acc += x & 0xff;
        else if (x & 2)
            acc -= (x >> 8) & 0xff;
        else
            acc ^= (x >> 16) & 0xff;
    }
    return acc;
}

PERF_MAIN("direct")
