// Signal delivery is the most expensive op in the suite, so the lowest N.
#define PERF_DEFAULT_N 1000000ull
#include "perf.h"

#include <signal.h>

// One guest signal delivery per iteration. raise(SIGUSR1) traps into Chimera's
// guest-signal virtualization, which must build a guest signal frame, enter the
// handler as translated code, and unwind back — exercising the recently added
// virtualized-signal path. This is far costlier than a plain syscall (it is a
// syscall plus a full handler round-trip), hence the lower N.
static volatile sig_atomic_t hits;

static void on_sigusr1(int sig) {
    (void)sig;
    hits++;
}

static uint64_t perf_run(uint64_t n, uint64_t seed) {
    (void)seed;
    struct sigaction sa = {0};
    sa.sa_handler = on_sigusr1;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGUSR1, &sa, NULL);

    for (uint64_t i = 0; i < n; i++)
        raise(SIGUSR1);

    return (uint64_t)hits; // checksum == number of deliveries observed
}

PERF_MAIN("signal")
