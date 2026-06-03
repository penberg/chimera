#include "perf.h"

// Megamorphic indirect dispatch: a 64-entry function-pointer table whose target
// varies on essentially every iteration, overflowing Chimera's inline
// indirect-branch prediction (which only captures the first few targets) into
// the hash-table fallback. Contrast with the 4-target `indirect`. MAMBO's data
// shows inlining beyond a handful of targets yields diminishing returns, so the
// fallback cost is what dominates here.

#define OP_LIST                                                          \
    X(0)  X(1)  X(2)  X(3)  X(4)  X(5)  X(6)  X(7)                        \
    X(8)  X(9)  X(10) X(11) X(12) X(13) X(14) X(15)                       \
    X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23)                       \
    X(24) X(25) X(26) X(27) X(28) X(29) X(30) X(31)                       \
    X(32) X(33) X(34) X(35) X(36) X(37) X(38) X(39)                       \
    X(40) X(41) X(42) X(43) X(44) X(45) X(46) X(47)                       \
    X(48) X(49) X(50) X(51) X(52) X(53) X(54) X(55)                       \
    X(56) X(57) X(58) X(59) X(60) X(61) X(62) X(63)

#define X(k)                                                             \
    static __attribute__((noinline)) uint64_t op_##k(uint64_t a, uint64_t b) { \
        return a * (2ull * (k) + 1ull) + b + (k);                        \
    }
OP_LIST
#undef X

typedef uint64_t (*op_fn)(uint64_t, uint64_t);
static const op_fn ops[64] = {
#define X(k) op_##k,
    OP_LIST
#undef X
};

static uint64_t perf_run(uint64_t n, uint64_t seed) {
    uint64_t acc = seed;
    for (uint64_t i = 0; i < n; i++)
        acc = ops[(acc ^ i) & 63](acc, i);
    return acc;
}

PERF_MAIN("indirect-mega")
