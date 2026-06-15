// RUN: %cc %s -o %t && %runner %t
//
// Without SA_RESTART, a slow syscall interrupted by a signal fails with EINTR
// rather than restarting. The parent blocks in read() on a pipe; a helper child
// interrupts it with SIGUSR1 (whose handler lacks SA_RESTART). Exits 0 only if
// the read returned -1/EINTR and the handler ran.

#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static volatile sig_atomic_t ran;

static void handler(int sig) {
    (void) sig;
    ran = 1;
}

int main(void) {
    int fds[2];
    if (pipe(fds) != 0) return 1;

    struct sigaction sa = {0};
    sa.sa_handler = handler;
    sa.sa_flags = 0; // no SA_RESTART
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 2;

    pid_t pid = fork();
    if (pid < 0) return 3;
    if (pid == 0) {
        // Helper: wait for the parent to block in read(), then interrupt it.
        pid_t parent = getppid();
        usleep(200000);
        kill(parent, SIGUSR1);
        _exit(0);
    }

    char buf = 0;
    errno = 0;
    ssize_t n = read(fds[0], &buf, 1);
    if (n != -1 || errno != EINTR) return 4; // must fail with EINTR
    if (!ran) return 5;                       // the handler must have run

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 6;
    return 0;
}
