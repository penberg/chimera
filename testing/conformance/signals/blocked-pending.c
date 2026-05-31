// RUN: %cc %s -o %t && %runner %t
//
// Signal masking: a blocked signal stays pending (its handler does not run)
// until it is unblocked, at which point it is delivered. Exits 0 only if the
// handler stayed silent while blocked and fired on unblock.

#include <signal.h>

static volatile sig_atomic_t ran;

static void handler(int sig) {
    (void) sig;
    ran = 1;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handler;
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 1;

    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    if (sigprocmask(SIG_BLOCK, &set, 0) != 0) return 2;

    if (raise(SIGUSR1) != 0) return 3;
    if (ran) return 4; // still blocked: must not have been delivered

    if (sigprocmask(SIG_UNBLOCK, &set, 0) != 0) return 5;
    if (!ran) return 6; // unblocking delivers the pending signal
    return 0;
}
