// RUN: %cc %s -o %t && %runner %t
//
// SA_RESETHAND restores the default disposition after the handler runs once.
// A child installs an SA_RESETHAND handler for SIGTERM and raises it twice: the
// first raise runs the handler (which reports via a pipe), the reset reverts the
// disposition to SIG_DFL, and the second raise terminates the child by SIGTERM.
// Exits 0 only if the handler ran exactly once and the child died by SIGTERM.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static int wfd;

static void handler(int sig) {
    (void) sig;
    char b = 1;
    write(wfd, &b, 1); // async-signal-safe report that the handler ran
}

int main(void) {
    int fds[2];
    if (pipe(fds) != 0) return 1;

    struct sigaction sa = {0};
    sa.sa_handler = handler;
    sa.sa_flags = SA_RESETHAND;
    if (sigaction(SIGTERM, &sa, 0) != 0) return 2;

    pid_t pid = fork();
    if (pid < 0) return 3;
    if (pid == 0) {
        close(fds[0]);
        wfd = fds[1];
        raise(SIGTERM); // handler runs once, disposition resets to default
        raise(SIGTERM); // default action now terminates the child
        _exit(123);     // must not reach here
    }

    close(fds[1]);
    char buf[8];
    if (read(fds[0], buf, sizeof buf) != 1) return 4; // handler ran exactly once

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 5;
    if (!WIFSIGNALED(status)) return 6;
    if (WTERMSIG(status) != SIGTERM) return 7;
    return 0;
}
