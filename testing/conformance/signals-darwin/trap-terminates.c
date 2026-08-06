// RUN: %cc %s -o %t && %runner %t
//
// A guest BRK must terminate the process by SIGTRAP, faithfully. BRK cannot
// run from the code cache — the host would take the exception against
// Chimera's own execution state — so the translator ends the block and the
// run loop raises the signal instead. A forked child takes the trap so the
// parent can read its wait status.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        __builtin_trap(); // brk on arm64
        _exit(42);        // must not be reached: the trap terminates first
    }

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 2;
    if (!WIFSIGNALED(status)) return 3; // exited normally -> trap was swallowed
    if (WTERMSIG(status) != SIGTRAP) return 4; // wrong signal
    return 0;
}
