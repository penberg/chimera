// RUN: %cc %s -o %t && %runner %t
//
// A signal caught while blocked stays pending; reverting its disposition to
// SIG_DFL does not discard it (only SIG_IGN does). When it is later unblocked,
// the kernel's default action must run — for SIGTERM, terminate the process.
// This exercises the delivery path where the disposition is SIG_DFL at the
// moment of delivery, which must perform the default action rather than drop
// the signal. A forked child drives the sequence; the parent confirms it died
// by SIGTERM. Exits 0 only if the child was terminated by SIGTERM.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static void handler(int sig) {
    (void) sig;
}

int main(void) {
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        // Install a handler so the signal is caught (and made pending) rather
        // than taking its default action immediately.
        struct sigaction sa = {0};
        sa.sa_handler = handler;
        if (sigaction(SIGTERM, &sa, 0) != 0) _exit(10);

        sigset_t set;
        sigemptyset(&set);
        sigaddset(&set, SIGTERM);
        if (sigprocmask(SIG_BLOCK, &set, 0) != 0) _exit(11);

        if (raise(SIGTERM) != 0) _exit(12); // caught, now pending (blocked)

        // Revert to default while the signal is pending; SIG_DFL must not
        // discard the pending instance.
        struct sigaction df = {0};
        df.sa_handler = SIG_DFL;
        if (sigaction(SIGTERM, &df, 0) != 0) _exit(13);

        // Unblocking delivers the pending SIGTERM, whose default action
        // terminates the process.
        if (sigprocmask(SIG_UNBLOCK, &set, 0) != 0) _exit(14);

        _exit(123); // must not reach here
    }

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 2;
    if (!WIFSIGNALED(status)) return 3;
    if (WTERMSIG(status) != SIGTERM) return 4;
    return 0;
}
