// RUN: %cc %s -o %t && %runner %t
//
// clone(CLONE_VM) without CLONE_THREAD asks for a *separate process* that
// shares the caller's memory: it has its own PID and is reaped with
// waitpid(). That is not a thread, and a runtime must not quietly turn it
// into one. Natively the call succeeds with exactly those semantics —
// asserted here. Under Chimera it cannot: forwarding would run the child
// natively outside the sandbox, and an in-process host thread would corrupt
// the semantics (parent's PID, unwaitable). Chimera therefore refuses the
// shape with EPERM — also asserted here, as the documented, deliberate
// divergence. Either way the caller sees something coherent, never a child
// that half-works.

#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

static volatile int shared;
static char stack[64 * 1024];

static int child(void *arg) {
    (void) arg;
    shared = 42; // visible to the parent iff CLONE_VM really shared memory
    return 7;
}

int main(void) {
    int pid = clone(child, stack + sizeof(stack), CLONE_VM | SIGCHLD, NULL);

    if (pid == -1) {
        // The sandboxed answer: a clear refusal, nothing created.
        if (errno != EPERM) {
            fprintf(stderr, "clone(CLONE_VM|SIGCHLD): errno %d, want EPERM\n", errno);
            return 1;
        }
        return 0;
    }

    // The native answer: a real child process with its own PID, waitable,
    // sharing our memory.
    if (pid == getpid()) {
        fprintf(stderr, "child got the parent's PID\n");
        return 2;
    }
    int st;
    if (waitpid(pid, &st, 0) != pid) {
        fprintf(stderr, "waitpid(%d) failed\n", pid);
        return 3;
    }
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 7) {
        fprintf(stderr, "child status %d, want exit 7\n", st);
        return 4;
    }
    if (shared != 42) {
        fprintf(stderr, "CLONE_VM did not share memory: shared=%d\n", shared);
        return 5;
    }
    return 0;
}
