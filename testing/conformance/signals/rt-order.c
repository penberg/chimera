// RUN: %cc %s -o %t && %runner %t
//
// When several different real-time signals are pending, they are delivered
// starting with the lowest-numbered. Block SIGRTMIN and SIGRTMIN+1, raise the
// higher one first, then unblock: the handler must still see SIGRTMIN before
// SIGRTMIN+1. Each handler blocks all signals (SA_mask) so the two run strictly
// one after another, making the dequeue order observable. Exits 0 only if both
// ran in ascending order.

#include <signal.h>

static volatile sig_atomic_t order[2];
static volatile sig_atomic_t n;

static void handler(int sig) {
    if (n < 2) order[n++] = sig;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handler;
    sigfillset(&sa.sa_mask); // serialize: no handler nests inside another
    if (sigaction(SIGRTMIN, &sa, 0) != 0) return 1;
    if (sigaction(SIGRTMIN + 1, &sa, 0) != 0) return 2;

    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGRTMIN);
    sigaddset(&set, SIGRTMIN + 1);
    if (sigprocmask(SIG_BLOCK, &set, 0) != 0) return 3;

    // Raise the higher-numbered signal first: ordering is by number, not
    // arrival.
    if (raise(SIGRTMIN + 1) != 0) return 4;
    if (raise(SIGRTMIN) != 0) return 5;

    if (sigprocmask(SIG_UNBLOCK, &set, 0) != 0) return 6;

    if (n != 2) return 7;
    if (order[0] != SIGRTMIN) return 8;
    if (order[1] != SIGRTMIN + 1) return 9;
    return 0;
}
