// RUN: %cc %s -o %t && %runner %t
//
// Standard signals do not queue: multiple instances raised while blocked
// collapse into a single pending instance. Block SIGUSR1, raise it three times,
// then unblock: the handler must run exactly once. Each handler blocks all
// signals (SA_mask) so a delivery cannot nest. Exits 0 only if the handler ran
// exactly once.

#include <signal.h>

static volatile sig_atomic_t count;

static void handler(int sig) {
    (void) sig;
    count++;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handler;
    sigfillset(&sa.sa_mask);
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 1;

    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    if (sigprocmask(SIG_BLOCK, &set, 0) != 0) return 2;

    for (int i = 0; i < 3; i++) {
        if (raise(SIGUSR1) != 0) return 3;
    }

    if (sigprocmask(SIG_UNBLOCK, &set, 0) != 0) return 4;

    if (count != 1) return 5; // coalesced into a single delivery
    return 0;
}
