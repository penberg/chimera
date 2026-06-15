// RUN: %cc %s -o %t && %runner %t
//
// A child created by fork(2) starts with an empty pending signal set, even if
// the parent had a signal pending. The parent blocks SIGUSR1 and raises it (now
// pending), then forks: the child must see nothing pending, and unblocking in
// the child must not run the handler. Exits 0 only if the child started clean.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

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

    if (raise(SIGUSR1) != 0) return 3; // pending in the parent

    pid_t pid = fork();
    if (pid < 0) return 4;
    if (pid == 0) {
        // The child must inherit no pending signals.
        sigset_t pend;
        sigemptyset(&pend);
        if (sigpending(&pend) != 0) _exit(10);
        if (sigismember(&pend, SIGUSR1)) _exit(11);
        // Unblocking must not deliver anything (nothing was pending).
        if (sigprocmask(SIG_UNBLOCK, &set, 0) != 0) _exit(12);
        if (ran) _exit(13);
        _exit(0);
    }

    int status = 0;
    pid_t w;
    do {
        w = waitpid(pid, &status, 0);
    } while (w < 0);

    if (w != pid) return 5;
    if (!WIFEXITED(status)) return 6;
    if (WEXITSTATUS(status) != 0) return 7;
    return 0;
}
