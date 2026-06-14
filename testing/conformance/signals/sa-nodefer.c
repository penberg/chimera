// RUN: %cc %s -o %t && %runner %t
//
// SA_NODEFER leaves the delivered signal unblocked during its own handler
// (normally it is blocked to prevent re-entry). The handler inspects its own
// mask. Exits 0 only if the signal is NOT blocked inside its handler.

#include <signal.h>

static volatile sig_atomic_t self_blocked = -1;

static void handler(int sig) {
    sigset_t m;
    sigemptyset(&m);
    sigprocmask(SIG_BLOCK, 0, &m);
    self_blocked = sigismember(&m, sig);
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handler;
    sa.sa_flags = SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 1;

    if (raise(SIGUSR1) != 0) return 2;

    if (self_blocked != 0) return 3; // SA_NODEFER: not blocked in its handler
    return 0;
}
