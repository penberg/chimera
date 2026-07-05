// RUN: %cc %s -o %t && %runner %t
//
// The kernel consults the caller's real blocked mask when it generates a
// job-control signal: tcsetpgrp() from a background process group must proceed
// without stopping the caller when SIGTTOU is blocked. Bash's forked children
// rely on exactly this — block SIGTTOU, take the terminal, exec — so a blocked
// mask that never reaches the kernel leaves every spawned command stopped.
// Exits 0 only if the background child's tcsetpgrp() succeeded and the child
// was never stopped.

#define _XOPEN_SOURCE 700

#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
    int master = posix_openpt(O_RDWR | O_NOCTTY);
    if (master < 0) return 1;
    if (grantpt(master) != 0 || unlockpt(master) != 0) return 2;

    pid_t leader = fork();
    if (leader < 0) return 3;
    if (leader == 0) {
        // Session leader: the first tty opened becomes the controlling
        // terminal, and the leader's process group its foreground group.
        if (setsid() < 0) _exit(10);
        int tty = open(ptsname(master), O_RDWR);
        if (tty < 0) _exit(11);

        pid_t child = fork();
        if (child < 0) _exit(12);
        if (child == 0) {
            // Background process group member: with SIGTTOU blocked, taking
            // the terminal must succeed rather than stop this process.
            if (setpgid(0, 0) != 0) _exit(20);
            sigset_t set;
            sigemptyset(&set);
            sigaddset(&set, SIGTTOU);
            if (sigprocmask(SIG_BLOCK, &set, 0) != 0) _exit(21);
            if (tcsetpgrp(tty, getpgrp()) != 0) _exit(22);
            _exit(0);
        }

        int status;
        if (waitpid(child, &status, WUNTRACED) != child) _exit(13);
        if (WIFSTOPPED(status)) {
            // Stopped by SIGTTOU: the blocked mask never reached the kernel.
            kill(child, SIGKILL);
            waitpid(child, 0, 0);
            _exit(14);
        }
        _exit(WIFEXITED(status) ? WEXITSTATUS(status) : 15);
    }

    int status;
    if (waitpid(leader, &status, 0) != leader) return 4;
    if (!WIFEXITED(status)) return 5;
    return WEXITSTATUS(status);
}
