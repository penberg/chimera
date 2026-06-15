// RUN: %cc %s -o %t && %runner %t
//
// SIGKILL and SIGSTOP can never be caught: rt_sigaction must reject an attempt
// to install a handler for either with EINVAL, while a normal signal is
// accepted. Exits 0 only if both uncatchable signals are refused and SIGUSR1 is
// allowed.

#include <errno.h>
#include <signal.h>

static void handler(int sig) {
    (void) sig;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handler;

    errno = 0;
    if (sigaction(SIGKILL, &sa, 0) != -1 || errno != EINVAL) return 1;

    errno = 0;
    if (sigaction(SIGSTOP, &sa, 0) != -1 || errno != EINVAL) return 2;

    // A catchable signal is still accepted.
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 3;

    return 0;
}
