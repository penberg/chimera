// RUN: %cc %s -o %t && %runner %t
//
// Real-time signals queue: unlike standard signals, multiple instances raised
// while blocked are not coalesced. Block SIGRTMIN, raise it three times, then
// unblock: the handler must run exactly three times. Exits 0 only if all three
// queued instances are delivered.

#include <signal.h>

static volatile sig_atomic_t count;

static void handler(int sig) {
    (void) sig;
    count++;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handler;
    sigfillset(&sa.sa_mask); // serialize handler runs
    if (sigaction(SIGRTMIN, &sa, 0) != 0) return 1;

    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGRTMIN);
    if (sigprocmask(SIG_BLOCK, &set, 0) != 0) return 2;

    for (int i = 0; i < 3; i++) {
        if (raise(SIGRTMIN) != 0) return 3;
    }

    if (sigprocmask(SIG_UNBLOCK, &set, 0) != 0) return 4;

    if (count != 3) return 5; // all three queued instances delivered
    return 0;
}
