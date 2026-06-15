// RUN: %cc %s -o %t && %runner %t
//
// The default action of SIGTERM is Term: with no handler installed, raising it
// must terminate the process by that signal. A forked child raises SIGTERM and
// the parent confirms the wait status reports death by signal SIGTERM. Exits 0
// only if the child died via WIFSIGNALED/WTERMSIG == SIGTERM.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        raise(SIGTERM);
        _exit(42); // the default action must terminate before this runs
    }

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 2;
    if (!WIFSIGNALED(status)) return 3;
    if (WTERMSIG(status) != SIGTERM) return 4;
    return 0;
}
