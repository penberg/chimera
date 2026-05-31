// RUN: %cc %s -o %t && %runner %t
//
// The headline case: a forked child exiting raises SIGCHLD in the parent, whose
// handler runs in the guest, and the parent reaps the child. This is the path
// that previously segfaulted (the host delivered SIGCHLD straight into the guest
// handler, natively, on a now-non-executable page). Exits 0 only if the handler
// ran and the child was reaped.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static volatile sig_atomic_t got_chld;

static void handler(int sig) {
    (void) sig;
    got_chld = 1;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handler;
    if (sigaction(SIGCHLD, &sa, 0) != 0) return 1;

    pid_t pid = fork();
    if (pid < 0) return 2;
    if (pid == 0) _exit(0);

    // Reap the child, retrying if SIGCHLD interrupts the wait (no SA_RESTART).
    int status = 0;
    pid_t w;
    do {
        w = waitpid(pid, &status, 0);
    } while (w < 0);

    if (w != pid) return 3;
    if (!got_chld) return 4; // the handler must have run
    return 0;
}
