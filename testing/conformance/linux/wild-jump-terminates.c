// RUN: %cc %s -o %t && timeout 5 %runner %t
//
// A wild jump to unmapped memory must terminate the process by SIGSEGV,
// faithfully. The fault here is on the instruction FETCH, not a data access:
// the translator is handed an arbitrary guest PC and must not dereference it
// itself -- doing so crashed Chimera in its own decoder when a real guest
// (node, via a corrupted function pointer) jumped to a near-null address.
// The translator reads the decode window through a guarded copy and the run
// loop raises the SIGSEGV the guest would have taken natively on the fetch.
// A forked child takes the fault so the parent can read its wait status.

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        // Jump through a near-null function pointer: an unmapped-fetch fault.
        void (*f)(void) = (void (*)(void)) 0x24;
        f();
        _exit(42); // must not be reached: the fault terminates first
    }

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 2;
    if (!WIFSIGNALED(status)) return 3; // exited normally -> fault was swallowed
    if (WTERMSIG(status) != SIGSEGV) return 4; // wrong signal
    return 0;
}
