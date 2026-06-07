#include "perf.h"

// Dispatch through a small function-pointer table: an indirect call every
// iteration, target varying over just 4 entries so it stays within Chimera's
// inline indirect-branch prediction. Contrast with indirect-mega, which
// overflows that prediction into the hash-table fallback.
static __attribute__((noinline)) uint64_t op_add(uint64_t a, uint64_t b) { return a + b; }
static __attribute__((noinline)) uint64_t op_xor(uint64_t a, uint64_t b) { return a ^ (b + 1); }
static __attribute__((noinline)) uint64_t op_mul(uint64_t a, uint64_t b) { return a * 3 + b; }
static __attribute__((noinline)) uint64_t op_rot(uint64_t a, uint64_t b) { return (a << 1) | ((b ^ a) & 1); }
typedef uint64_t (*op_fn)(uint64_t, uint64_t);

static uint64_t perf_run(uint64_t n, uint64_t seed) {
    static const op_fn ops[4] = {op_add, op_xor, op_mul, op_rot};
    uint64_t acc = seed;
    for (uint64_t i = 0; i < n; i++) {
        op_fn f = ops[(acc ^ i) & 3];
        acc = f(acc, i);
    }
    return acc;
}

PERF_MAIN("indirect")
