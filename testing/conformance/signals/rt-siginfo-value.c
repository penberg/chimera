// RUN: %cc %s -o %t && %runner %t
// UNSUPPORTED: darwin -- Darwin has no POSIX real-time signals.
//
// sigqueue() attaches a value to a real-time signal, delivered to an SA_SIGINFO
// handler as si_value with si_code == SI_QUEUE. Because real-time signals queue,
// each queued instance carries its own value. Block SIGRTMIN, queue three
// distinct values, then unblock: the handler must observe all three (here, their
// sum and count). Exits 0 only if every queued value arrived intact.

#include <signal.h>
#include <unistd.h>

static volatile sig_atomic_t sum;
static volatile sig_atomic_t count;
static volatile sig_atomic_t all_queue;

static void handler(int sig, siginfo_t *info, void *uc) {
    (void) sig;
    (void) uc;
    if (info->si_code != SI_QUEUE) all_queue = 1; // sticky: any non-SI_QUEUE
    sum += info->si_value.sival_int;
    count++;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO;
    sigfillset(&sa.sa_mask); // serialize handler runs
    if (sigaction(SIGRTMIN, &sa, 0) != 0) return 1;

    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGRTMIN);
    if (sigprocmask(SIG_BLOCK, &set, 0) != 0) return 2;

    union sigval v;
    int vals[3] = {10, 20, 30};
    for (int i = 0; i < 3; i++) {
        v.sival_int = vals[i];
        if (sigqueue(getpid(), SIGRTMIN, v) != 0) return 3;
    }

    if (sigprocmask(SIG_UNBLOCK, &set, 0) != 0) return 4;

    if (count != 3) return 5;     // all three queued instances delivered
    if (sum != 60) return 6;      // 10 + 20 + 30, each value intact
    if (all_queue) return 7;      // every one carried si_code == SI_QUEUE
    return 0;
}
