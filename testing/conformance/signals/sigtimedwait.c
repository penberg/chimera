// RUN: %cc %s -o %t && %runner %t
//
// sigtimedwait() synchronously accepts a signal from a set without running its
// handler. Block SIGUSR1, raise it (now pending), and dequeue it with
// sigtimedwait(): it must return SIGUSR1, fill siginfo, leave the handler
// un-run, and clear it from the pending set. A second wait on an empty set with
// a short timeout must report EAGAIN. Exits 0 only if both hold.

#include <errno.h>
#include <signal.h>
#include <time.h>

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

    if (raise(SIGUSR1) != 0) return 3; // pending (blocked)

    siginfo_t info;
    info.si_signo = 0;
    struct timespec ts = {2, 0};
    int r = sigtimedwait(&set, &info, &ts);
    if (r != SIGUSR1) return 4;
    if (info.si_signo != SIGUSR1) return 5;
    if (ran) return 6; // the handler must NOT have run

    // The accepted signal is consumed: no longer pending.
    sigset_t pend;
    sigemptyset(&pend);
    if (sigpending(&pend) != 0) return 7;
    if (sigismember(&pend, SIGUSR1)) return 8;

    // Nothing pending in {SIGUSR2}: a short wait times out with EAGAIN.
    sigset_t set2;
    sigemptyset(&set2);
    sigaddset(&set2, SIGUSR2);
    struct timespec ts2 = {0, 50 * 1000 * 1000}; // 50 ms
    errno = 0;
    r = sigtimedwait(&set2, 0, &ts2);
    if (r != -1 || errno != EAGAIN) return 9;

    return 0;
}
