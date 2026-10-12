// RUN: %cc %s -o %t && %runner %t
//
// A wild jump raises SIGSEGV on the instruction fetch, and a guest that
// catches it recovers: the handler sees the target as both the fault address
// and the interrupted pc, with SEGV_MAPERR when nothing is mapped there and
// SEGV_ACCERR when a mapping forbids the fetch, and siglongjmps out. A
// runtime that runs the guest natively takes that fault in its own host
// handler, which must turn it into the guest's SIGSEGV rather than a fatal
// fault.

#define _GNU_SOURCE

#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <sys/mman.h>
#include <ucontext.h>

static sigjmp_buf env;
static volatile uintptr_t addr, rip;
static volatile int code;

static void on_segv(int sig, siginfo_t *info, void *ctx) {
    (void) sig;
    addr = (uintptr_t) info->si_addr;
    code = info->si_code;
#if defined(__x86_64__)
    rip = ((ucontext_t *) ctx)->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
    rip = ((ucontext_t *) ctx)->uc_mcontext.pc;
#endif
    siglongjmp(env, 1);
}

// Jump to `target` and return the SIGSEGV's si_code, or -1 if the handler
// saw the wrong addresses.
static int jump(uintptr_t target) {
    if (sigsetjmp(env, 1) == 0) {
        void (*f)(void) = (void (*)(void)) target;
        f();
        return -2; // the jump cannot return
    }
    if (rip != target || addr != target) return -1;
    return code;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_sigaction = on_segv;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGSEGV, &sa, 0) != 0) return 1;

    if (jump(0x24) != SEGV_MAPERR) return 2;

    void *page = mmap(0, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) return 3;
    if (jump((uintptr_t) page) != SEGV_ACCERR) return 4;
    return 0;
}
