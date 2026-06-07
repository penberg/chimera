#include "perf.h"

// Tight integer loop: a serial ALU dependency chain plus the loop back-edge.
// The back-edge is a direct conditional branch that linking keeps resolved in
// the code cache, so steady state is pure translated ALU with no exits.
static uint64_t perf_run(uint64_t n, uint64_t seed) {
    uint64_t acc = seed;
    for (uint64_t i = 0; i < n; i++) {
        acc = acc * 6364136223846793005ull + 1442695040888963407ull;
        acc ^= acc >> 29;
        acc += i;
    }
    return acc;
}

PERF_MAIN("loop")
