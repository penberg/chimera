// RUN: %cc %s -o %t && timeout 5 %runner %t
//
// A genuine guest fault (a write to an unmapped address) must terminate the
// process by SIGSEGV, faithfully. Chimera owns the host SIGSEGV slot for its
// self-modifying-code trap; this checks that a fault which is NOT an SMC write
// is classified as genuine, reported (the crash dump), and then re-raised so the
// kernel terminates the guest with WIFSIGNALED/WTERMSIG == SIGSEGV — rather than
// being swallowed or surfaced as the wrong signal. A forked child takes the
// fault so the parent can read its wait status.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        // Write through a null pointer: an unmapped-address fault (SEGV_MAPERR).
        *(volatile int *) 0 = 1;
        _exit(42); // must not be reached: the fault terminates first
    }

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 2;
    if (!WIFSIGNALED(status)) return 3; // exited normally -> fault was swallowed
    if (WTERMSIG(status) != SIGSEGV) return 4; // wrong signal
    return 0;
}
