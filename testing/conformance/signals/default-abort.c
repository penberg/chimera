// RUN: %cc %s -o %t && %runner %t
//
// The default action of SIGABRT is Core: abort() must terminate the process by
// SIGABRT. A forked child calls abort() and the parent confirms the wait status
// reports death by signal SIGABRT. Exits 0 only if the child died via
// WIFSIGNALED/WTERMSIG == SIGABRT.

#include <signal.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        abort();
        _exit(42); // abort() must terminate before this runs
    }

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 2;
    if (!WIFSIGNALED(status)) return 3;
    if (WTERMSIG(status) != SIGABRT) return 4;
    return 0;
}
