// RUN: %cc %s -o %t && %runner %t
//
// A wild jump to unmapped memory raises SIGSEGV on the instruction fetch, and
// a guest that catches it recovers: the handler sees the target as the
// interrupted pc and siglongjmps out. A runtime that runs the guest natively
// takes that fault in its own host handler, which must turn it into the
// guest's SIGSEGV rather than a fatal fault.

#define _GNU_SOURCE

#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <ucontext.h>

#define TARGET 0x24

static sigjmp_buf env;
static volatile uintptr_t rip;

static void on_segv(int sig, siginfo_t *info, void *ctx) {
    (void) sig;
    (void) info;
#if defined(__x86_64__)
    rip = ((ucontext_t *) ctx)->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
    rip = ((ucontext_t *) ctx)->uc_mcontext.pc;
#endif
    siglongjmp(env, 1);
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_sigaction = on_segv;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGSEGV, &sa, 0) != 0) return 1;

    if (sigsetjmp(env, 1) == 0) {
        void (*f)(void) = (void (*)(void)) TARGET;
        f();
        return 2; // the jump cannot return
    }
    if (rip != TARGET) return 3;
    return 0;
}
