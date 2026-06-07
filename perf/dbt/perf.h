// Shared self-timing harness for Chimera's steady-state DBT microbenchmarks.
//
// Each perf-<name>.c defines a single `perf_run(n, seed)` workload and ends
// with PERF_MAIN("<name>"). The hot loop is self-timed with CLOCK_MONOTONIC so
// process startup and one-time block translation are excluded from the figure
// — what remains is the steady-state per-operation overhead Chimera adds over
// native execution. (This is why the suite does not use hyperfine: wall-clock
// of the whole process would fold startup and cold translation back in, which
// is a different metric. See README.md.)
//
//   cc -O2 -o loop perf-loop.c
//   ./loop [iterations]
//   chimera run -- ./loop [iterations]

#ifndef PERF_H
#define PERF_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline uint64_t perf_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// Each workload file defines this; it must do `n` units of work and return a
// checksum so the optimizer cannot fold the loop away.
static uint64_t perf_run(uint64_t n, uint64_t seed);

// Iteration count. Cheap workloads keep the high default; syscall- and
// signal-bound ones lower it by #defining PERF_DEFAULT_N before including this.
#ifndef PERF_DEFAULT_N
#define PERF_DEFAULT_N 100000000ull
#endif

#define PERF_MAIN(name)                                                        \
    int main(int argc, char **argv) {                                          \
        uint64_t n = argc > 1 ? strtoull(argv[1], NULL, 0) : PERF_DEFAULT_N;   \
        /* Derive the seed from argc so the optimizer cannot fold the loop. */ \
        uint64_t seed = (uint64_t)argc * 2654435761u + 1;                      \
        uint64_t t0 = perf_now_ns();                                           \
        uint64_t result = perf_run(n, seed);                                   \
        uint64_t t1 = perf_now_ns();                                           \
        printf("%-14s %8.3f ns/op  (n=%llu, checksum=%llu)\n", name,           \
               (double)(t1 - t0) / (double)n, (unsigned long long)n,           \
               (unsigned long long)result);                                    \
        return 0;                                                              \
    }

#endif // PERF_H
