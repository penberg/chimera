// RUN: %cc %s -o %t && %runner %t
//
// pause() suspends the process until a signal whose disposition is to run a
// handler is delivered, then returns -1/EINTR. A helper child interrupts the
// paused parent with SIGUSR1. Exits 0 only if pause() returned -1/EINTR after
// the handler ran.

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
    struct sigaction sa = {0};
    sa.sa_handler = handler;
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 1;

    pid_t pid = fork();
    if (pid < 0) return 2;
    if (pid == 0) {
        pid_t parent = getppid();
        usleep(200000);
        kill(parent, SIGUSR1);
        _exit(0);
    }

    errno = 0;
    if (pause() != -1 || errno != EINTR) return 3;
    if (!ran) return 4; // the handler ran before pause() returned

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 5;
    return 0;
}
