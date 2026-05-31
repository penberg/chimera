// RUN: %cc %s -o %t && %runner %t
//
// rt_sigaction plus a self-directed signal: the handler runs in the guest and
// execution resumes after it returns (the rt_sigreturn path). Exits 0 only if
// the handler actually ran. Previously the host delivered the signal straight
// into the guest handler natively, which faulted under W^X.

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

    if (raise(SIGUSR1) != 0) return 2;

    if (!ran) return 3; // handler must have run before raise() returned
    return 0;
}
