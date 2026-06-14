// RUN: %cc %s -o %t && %runner %t
//
// A signal generated while blocked is marked pending until delivered.
// sigpending() must report it. Block SIGUSR1, raise it (it cannot be delivered
// while blocked), and confirm sigpending() shows it; then unblock and confirm
// it is no longer pending (it was delivered). Exits 0 only if sigpending()
// tracks the blocked signal correctly.

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
    if (ran) return 4; // blocked: must not have been delivered yet

    sigset_t pend;
    sigemptyset(&pend);
    if (sigpending(&pend) != 0) return 5;
    if (!sigismember(&pend, SIGUSR1)) return 6; // must be reported pending
    if (sigismember(&pend, SIGUSR2)) return 7;  // unrelated signal must not be

    if (sigprocmask(SIG_UNBLOCK, &set, 0) != 0) return 8;
    if (!ran) return 9; // unblocking delivers it

    sigemptyset(&pend);
    if (sigpending(&pend) != 0) return 10;
    if (sigismember(&pend, SIGUSR1)) return 11; // no longer pending
    return 0;
}
