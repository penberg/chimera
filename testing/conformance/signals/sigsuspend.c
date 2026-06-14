// RUN: %cc %s -o %t && %runner %t
//
// sigsuspend() atomically installs a temporary mask, waits until a caught
// signal runs its handler, then returns -1/EINTR with the previous mask
// restored. SIGUSR1 is blocked; sigsuspend() with an empty mask unblocks it for
// the duration of the wait so a helper child's SIGUSR1 is delivered. Exits 0
// only if the handler ran during the suspend, sigsuspend returned EINTR, and the
// original mask was restored afterwards.

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

    sigset_t block;
    sigemptyset(&block);
    sigaddset(&block, SIGUSR1);
    if (sigprocmask(SIG_BLOCK, &block, 0) != 0) return 2;

    pid_t pid = fork();
    if (pid < 0) return 3;
    if (pid == 0) {
        pid_t parent = getppid();
        usleep(200000);
        kill(parent, SIGUSR1);
        _exit(0);
    }

    // Suspend with an empty mask: SIGUSR1 is unblocked only for the wait.
    sigset_t empty;
    sigemptyset(&empty);
    errno = 0;
    if (sigsuspend(&empty) != -1 || errno != EINTR) return 4;
    if (!ran) return 5; // the handler ran during the suspend

    // The pre-suspend mask must be restored (SIGUSR1 blocked again).
    sigset_t cur;
    sigemptyset(&cur);
    if (sigprocmask(SIG_BLOCK, 0, &cur) != 0) return 6;
    if (!sigismember(&cur, SIGUSR1)) return 7;

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 8;
    return 0;
}
