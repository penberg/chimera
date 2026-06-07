#include "perf.h"

// One direct call + return per iteration to a single leaf function. The call
// edge is linked, and every `ret` resolves to the same return site — so both
// the CPU's return-stack and Chimera's inline indirect-branch lookup hit every
// time. This is the cheap, monomorphic return case; contrast with `callsites`,
// where the same callee is reached from many sites and its return target
// varies, defeating the inline lookup.
static __attribute__((noinline)) uint64_t leaf(uint64_t a, uint64_t i) {
    return a * 1000003ull + i;
}

static uint64_t perf_run(uint64_t n, uint64_t seed) {
    uint64_t acc = seed;
    for (uint64_t i = 0; i < n; i++)
        acc = leaf(acc, i);
    return acc;
}

PERF_MAIN("ret")
