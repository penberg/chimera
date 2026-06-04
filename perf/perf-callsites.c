#include "perf.h"

// Polymorphic return: a single leaf `helper` is reached through eight distinct
// caller functions, cycled per iteration. Because helper is called from eight
// sites, its `ret` resolves to a different return address each time, so
// Chimera's inline return lookup misses and falls back to the hash probe on
// most iterations. (Deep recursion would NOT exercise this — a recursive call's
// `ret` almost always returns to the one recursive-call site, staying
// monomorphic like `ret`. It takes many *distinct* call sites to vary the
// return target.) The callers are noinline so the compiler cannot merge them
// back into a single site.
static __attribute__((noinline)) uint64_t helper(uint64_t a, uint64_t i) {
    return a * 1000003ull + i;
}

#define CALLER(k)                                                              \
    static __attribute__((noinline)) uint64_t c_##k(uint64_t a, uint64_t i) {  \
        return helper(a, i) ^ (0x9e3779b97f4a7c15ull * ((k) + 1));             \
    }
CALLER(0) CALLER(1) CALLER(2) CALLER(3)
CALLER(4) CALLER(5) CALLER(6) CALLER(7)

static uint64_t perf_run(uint64_t n, uint64_t seed) {
    uint64_t acc = seed;
    for (uint64_t i = 0; i < n; i++) {
        switch (i & 7) {
        case 0:  acc = c_0(acc, i); break;
        case 1:  acc = c_1(acc, i); break;
        case 2:  acc = c_2(acc, i); break;
        case 3:  acc = c_3(acc, i); break;
        case 4:  acc = c_4(acc, i); break;
        case 5:  acc = c_5(acc, i); break;
        case 6:  acc = c_6(acc, i); break;
        default: acc = c_7(acc, i); break;
        }
    }
    return acc;
}

PERF_MAIN("callsites")
