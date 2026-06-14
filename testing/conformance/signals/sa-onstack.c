// RUN: %cc %s -o %t && %runner %t
//
// SA_ONSTACK runs the handler on the alternate signal stack established by
// sigaltstack(2). The handler checks that a local variable lives within the
// alternate stack's bounds. Exits 0 only if the handler ran on that stack.

#include <signal.h>
#include <stdint.h>

static char altstack[SIGSTKSZ];
static volatile sig_atomic_t on_alt = -1;

static void handler(int sig) {
    (void) sig;
    int local;
    uintptr_t a = (uintptr_t) &local;
    uintptr_t lo = (uintptr_t) altstack;
    uintptr_t hi = lo + sizeof altstack;
    on_alt = (a >= lo && a < hi);
}

int main(void) {
    stack_t ss = {0};
    ss.ss_sp = altstack;
    ss.ss_size = sizeof altstack;
    ss.ss_flags = 0;
    if (sigaltstack(&ss, 0) != 0) return 1;

    struct sigaction sa = {0};
    sa.sa_handler = handler;
    sa.sa_flags = SA_ONSTACK;
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 2;

    if (raise(SIGUSR1) != 0) return 3;

    if (on_alt != 1) return 4; // handler ran on the alternate stack
    return 0;
}
