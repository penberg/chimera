// RUN: %cc %s -o %t && %runner %t
//
// The signals named in sa_mask are blocked for the duration of the handler (in
// addition to the signal being delivered, which is blocked unless SA_NODEFER).
// The handler inspects its own mask. Exits 0 only if both the delivered signal
// and the sa_mask signal are blocked inside the handler.

#include <signal.h>

static volatile sig_atomic_t usr1_blocked = -1;
static volatile sig_atomic_t usr2_blocked = -1;

static void handler(int sig) {
    (void) sig;
    sigset_t m;
    sigemptyset(&m);
    sigprocmask(SIG_BLOCK, 0, &m);
    usr1_blocked = sigismember(&m, SIGUSR1);
    usr2_blocked = sigismember(&m, SIGUSR2);
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handler;
    sigemptyset(&sa.sa_mask);
    sigaddset(&sa.sa_mask, SIGUSR2); // block SIGUSR2 during the handler
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 1;

    if (raise(SIGUSR1) != 0) return 2;

    if (usr1_blocked != 1) return 3; // the delivered signal is blocked too
    if (usr2_blocked != 1) return 4; // the sa_mask signal is blocked
    return 0;
}
