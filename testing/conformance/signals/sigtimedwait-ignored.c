// RUN: %cc %s -o %t && %runner %t
// UNSUPPORTED: darwin -- Darwin has no sigtimedwait syscall.
//
// A signal that would be ignored on delivery does not interrupt
// sigtimedwait(): the kernel discards it and keeps waiting. Wait on {SIGUSR1}
// while a child sends SIGWINCH (default action Ign) and then a SIG_IGN'd
// SIGUSR2; the wait must ride through both and report EAGAIN on timeout, not
// EINTR. Exits 0 only if it does.

#include <errno.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = SIG_IGN;
    if (sigaction(SIGUSR2, &sa, 0) != 0) return 1;

    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    if (sigprocmask(SIG_BLOCK, &set, 0) != 0) return 2;

    pid_t parent = getpid();
    pid_t child = fork();
    if (child < 0) return 3;
    if (child == 0) {
        struct timespec delay = {0, 50 * 1000 * 1000}; // 50 ms: mid-wait
        nanosleep(&delay, 0);
        kill(parent, SIGWINCH);
        kill(parent, SIGUSR2);
        _exit(0);
    }

    struct timespec ts = {0, 200 * 1000 * 1000}; // 200 ms
    errno = 0;
    int r = sigtimedwait(&set, 0, &ts);
    int saved = errno;

    int status;
    if (waitpid(child, &status, 0) != child) return 4;
    if (r != -1 || saved != EAGAIN) return 5; // EINTR here means a spurious wake

    return 0;
}
