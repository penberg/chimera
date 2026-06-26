// RUN: %cc %s -o %t && %runner %t
//
// An asynchronous signal must reach the guest handler even when the guest is
// looping purely through linked direct-call edges — direct recursion with no
// back-branch and no `ret`. Such a cycle stays entirely in the code cache until
// the stack overflows, so without a safepoint poll on the direct-call back-edge
// the pending signal is never delivered. Each call does only straight-line work
// (no loop, so no competing back-edge poll can deliver the signal instead),
// which also slows the recursion enough that the signal lands long before the
// stack overflows. A helper child raises the signal; if delivery regresses, the
// child's watchdog kill bounds the failure instead of hanging the suite.

#include <signal.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>

// A handful of slow `idiv`s per call: enough to keep the recursion well short of
// overflow for tens of ms (so the signal lands first) while keeping the
// straight-line block far under the translator's basic-block size limit.
#define V8(x) x /= dvr; x /= dvr; x /= dvr; x /= dvr; x /= dvr; x /= dvr; x /= dvr; x /= dvr;
#define V64(x) V8(x) V8(x) V8(x) V8(x) V8(x) V8(x) V8(x) V8(x)

static volatile uint64_t sink;
static volatile uint64_t dvr = 1000003;
static volatile sig_atomic_t helper_pid;

static void on_sig(int sig) {
    (void) sig;
    // Stop the watchdog helper before exiting (kill is async-signal-safe), so it
    // can't outlive us and later SIGKILL a recycled PID.
    if (helper_pid > 0) {
        kill((pid_t) helper_pid, SIGKILL);
    }
    _exit(0); // delivery into the guest handler == success
}

static void recurse(void) {
    uint64_t x = sink + 0x9e3779b97f4a7c15ull;
    V64(x) // straight-line work: slows the call rate, no back-edge of its own
    sink = x;
    recurse(); // direct self-call: the only loop-closing edge (compiled at -O0,
               // so it is a real `call`, not a tail-call `jmp`)
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = on_sig;
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 1;

    // Block SIGUSR1 until helper_pid is published. Otherwise, if the parent is
    // descheduled past the helper's initial sleep, the signal could arrive before
    // the assignment below, the handler would take the success path without
    // stopping the helper, and the watchdog would survive to SIGKILL a recycled
    // PID. A signal that arrives while blocked stays pending and is delivered the
    // moment we unblock — after helper_pid is set.
    sigset_t block, old;
    sigemptyset(&block);
    sigaddset(&block, SIGUSR1);
    if (sigprocmask(SIG_BLOCK, &block, &old) != 0) return 4;

    pid_t parent = getpid();
    pid_t helper = fork();
    if (helper < 0) return 2;
    if (helper == 0) {
        usleep(50000); // let the parent get deep into the recursion
        kill(parent, SIGUSR1);
        usleep(3000000); // watchdog: if delivery regressed, force a bounded exit
        kill(parent, SIGKILL);
        _exit(0);
    }
    helper_pid = helper; // published before SIGUSR1 can be delivered (see above)
    sigprocmask(SIG_SETMASK, &old, 0); // now safe to deliver

    recurse(); // never returns: the handler exits, or we overflow without the fix
    return 3;  // unreachable
}
