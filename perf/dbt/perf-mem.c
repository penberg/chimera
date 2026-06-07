#include "perf.h"

// Control workload: a pointer-chasing dependent-load chain. In a same-ISA DBT,
// guest loads and stores execute as native loads and stores — Chimera adds no
// instrumentation to them — so this should run at ~1.0x. Its job is to confirm
// the runtime adds no hidden per-memory-op cost and to catch any regression
// that would. The chain is latency-bound (each load feeds the next address),
// so the figure is real memory latency, paid identically both ways.
#define MEM_CYCLE 4096 // 16 KiB of u32s; fits L1/L2, so this is L1-latency bound
static uint32_t cycle[MEM_CYCLE];

static uint64_t perf_run(uint64_t n, uint64_t seed) {
    // A single cycle through all slots: step by a stride coprime to the size.
    const uint32_t stride = 1469;
    for (uint32_t k = 0; k < MEM_CYCLE; k++)
        cycle[k] = (uint32_t)((k + stride) & (MEM_CYCLE - 1));

    uint32_t p = (uint32_t)(seed & (MEM_CYCLE - 1));
    uint64_t acc = 0;
    for (uint64_t i = 0; i < n; i++) {
        p = cycle[p];
        acc += p;
    }
    return acc;
}

PERF_MAIN("mem")
