// RUN: %cc %s -o %t && %runner %t
//
// Some signals have a default action of Ign: with no handler installed, raising
// SIGCHLD, SIGURG, or SIGWINCH must neither terminate the process nor remain
// pending. Exits 0 only if the process survives every raise with none of the
// signals present in sigpending().

#include <signal.h>

int main(void) {
    sigset_t set;
    sigset_t pend;

    // No disposition is installed, so each signal keeps its default action.
    // Ensure inherited blocked masks do not hide queued-vs-discarded bugs.
    sigemptyset(&set);
    sigaddset(&set, SIGCHLD);
    sigaddset(&set, SIGURG);
    sigaddset(&set, SIGWINCH);
    if (sigprocmask(SIG_UNBLOCK, &set, 0) != 0) return 1;

    sigemptyset(&pend);
    if (sigpending(&pend) != 0) return 2;
    if (sigismember(&pend, SIGCHLD)) return 3;
    if (sigismember(&pend, SIGURG)) return 4;
    if (sigismember(&pend, SIGWINCH)) return 5;

    if (raise(SIGCHLD) != 0) return 6;
    if (raise(SIGURG) != 0) return 7;
    if (raise(SIGWINCH) != 0) return 8;

    sigemptyset(&pend);
    if (sigpending(&pend) != 0) return 9;
    if (sigismember(&pend, SIGCHLD)) return 10;
    if (sigismember(&pend, SIGURG)) return 11;
    if (sigismember(&pend, SIGWINCH)) return 12;

    // Reaching here means none of them terminated the process or stayed pending.
    return 0;
}
