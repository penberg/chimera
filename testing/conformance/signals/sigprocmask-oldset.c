// RUN: %cc %s -o %t && %runner %t
//
// sigprocmask reports the previous mask through oldset, and SIG_SETMASK,
// SIG_BLOCK, and SIG_UNBLOCK compose as expected. Exits 0 only if every step
// observes the right mask.

#include <signal.h>

int main(void) {
    sigset_t usr1, usr2, prev, cur;
    sigemptyset(&usr1);
    sigaddset(&usr1, SIGUSR1);
    sigemptyset(&usr2);
    sigaddset(&usr2, SIGUSR2);

    // SETMASK to {SIGUSR1}.
    if (sigprocmask(SIG_SETMASK, &usr1, 0) != 0) return 1;

    // BLOCK {SIGUSR2}; the prior mask must be exactly {SIGUSR1}.
    sigemptyset(&prev);
    if (sigprocmask(SIG_BLOCK, &usr2, &prev) != 0) return 2;
    if (!sigismember(&prev, SIGUSR1)) return 3;
    if (sigismember(&prev, SIGUSR2)) return 4;

    // The mask is now {SIGUSR1, SIGUSR2}.
    sigemptyset(&cur);
    if (sigprocmask(SIG_BLOCK, 0, &cur) != 0) return 5;
    if (!sigismember(&cur, SIGUSR1) || !sigismember(&cur, SIGUSR2)) return 6;

    // UNBLOCK {SIGUSR1}; the mask becomes {SIGUSR2}.
    if (sigprocmask(SIG_UNBLOCK, &usr1, 0) != 0) return 7;
    sigemptyset(&cur);
    if (sigprocmask(SIG_BLOCK, 0, &cur) != 0) return 8;
    if (sigismember(&cur, SIGUSR1)) return 9;
    if (!sigismember(&cur, SIGUSR2)) return 10;

    return 0;
}
