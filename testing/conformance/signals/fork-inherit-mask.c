// RUN: %cc %s -o %t && %runner %t
//
// A child created by fork(2) inherits a copy of the parent's signal mask. The
// parent blocks SIGUSR1 and SIGUSR2, then forks; the child checks its own mask
// and reports through its exit status. Exits 0 only if the child saw both
// signals blocked and nothing spurious.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
    sigset_t block;
    sigemptyset(&block);
    sigaddset(&block, SIGUSR1);
    sigaddset(&block, SIGUSR2);
    if (sigprocmask(SIG_BLOCK, &block, 0) != 0) return 1;

    pid_t pid = fork();
    if (pid < 0) return 2;
    if (pid == 0) {
        sigset_t cur;
        sigemptyset(&cur);
        if (sigprocmask(SIG_BLOCK, 0, &cur) != 0) _exit(10);
        if (!sigismember(&cur, SIGUSR1)) _exit(11);
        if (!sigismember(&cur, SIGUSR2)) _exit(12);
        if (sigismember(&cur, SIGTERM)) _exit(13); // not blocked: must not appear
        _exit(0);
    }

    int status = 0;
    pid_t w;
    do {
        w = waitpid(pid, &status, 0);
    } while (w < 0); // retry if a signal interrupts the wait

    if (w != pid) return 3;
    if (!WIFEXITED(status)) return 4;
    if (WEXITSTATUS(status) != 0) return 5; // child saw the inherited mask
    return 0;
}
