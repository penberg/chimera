// RUN: %cc %s -o %t && %runner %t
//
// A handler installed with SA_RESTART restarts a slow syscall interrupted by
// the signal rather than failing it with EINTR. The parent blocks in read() on
// a pipe; a helper child interrupts it with SIGUSR1 and then writes the byte
// the restarted read consumes. Exits 0 only if the handler ran (the read was
// interrupted) and the read still returned the byte (it was restarted).

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
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 2;

    pid_t pid = fork();
    if (pid < 0) return 3;
    if (pid == 0) {
        // Helper: wait for the parent to block in read(), interrupt it with
        // SIGUSR1, then provide the byte the restarted read will consume.
        pid_t parent = getppid();
        usleep(200000);
        kill(parent, SIGUSR1);
        usleep(100000);
        char b = 'x';
        if (write(fds[1], &b, 1) != 1) _exit(1);
        _exit(0);
    }

    char buf = 0;
    if (read(fds[0], &buf, 1) != 1) return 4; // restarted read returns the byte
    if (buf != 'x') return 5;
    if (!ran) return 6; // the handler ran, so the read really was interrupted

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 7;
    return 0;
}
