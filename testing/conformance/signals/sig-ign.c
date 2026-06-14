// RUN: %cc %s -o %t && %runner %t
//
// A signal set to SIG_IGN is discarded: raising it neither runs a previously
// armed handler nor triggers the default action (SIGUSR1 would otherwise
// terminate the process), and sigpending() must never report it. Exits 0 only
// if the process survives the raises with the handler untouched and SIGUSR1
// absent from the pending set throughout.

#include <signal.h>

static volatile sig_atomic_t ran;

static void handler(int sig) {
    (void) sig;
    ran = 1;
}

int main(void) {
    struct sigaction sa = {0};
    sigset_t set;
    sigset_t pend;

    // Arm a handler first, then override the disposition with SIG_IGN.
    sa.sa_handler = handler;
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 1;
    sa.sa_handler = SIG_IGN;
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 2;

    // Ensure the test is not satisfied by inheriting SIGUSR1 blocked.
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    if (sigprocmask(SIG_UNBLOCK, &set, 0) != 0) return 3;

    sigemptyset(&pend);
    if (sigpending(&pend) != 0) return 4;
    if (sigismember(&pend, SIGUSR1)) return 5;

    if (raise(SIGUSR1) != 0) return 6;
    if (raise(SIGUSR1) != 0) return 7;

    sigemptyset(&pend);
    if (sigpending(&pend) != 0) return 8;
    if (sigismember(&pend, SIGUSR1)) return 9; // ignored: must be discarded

    if (ran) return 10; // ignored: the handler must never have run
    return 0;
}
