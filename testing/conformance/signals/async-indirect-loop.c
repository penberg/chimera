// RUN: %cc %s -o %t && %runner %t
//
// The indirect-branch variant of async-tight-loop: a computed-goto dispatch
// loop that closes via an indirect jump kept resident in the inline IB cache,
// never returning to the run loop. A signal landing inside the shared IB-lookup
// routine (mid-hash, mid-probe, on the hit jump) must be recovered from that
// routine's borrowed slots, not just from a block body. A watchdog child
// bounds a regression to a non-zero exit instead of a hang.

#include <signal.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

static volatile sig_atomic_t got;

static void on_alarm(int sig) {
    (void) sig;
    got = 1;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = on_alarm;
    if (sigaction(SIGALRM, &sa, 0) != 0) return 1;

    pid_t parent = getpid();
    pid_t wd = fork();
    if (wd < 0) return 2;
    if (wd == 0) {
        sleep(5);
        kill(parent, SIGKILL);
        _exit(0);
    }

    struct itimerval it = {0};
    it.it_value.tv_usec = 50000; // 50ms
    if (setitimer(ITIMER_REAL, &it, 0) != 0) return 3;

    // Computed-goto loop: each iteration jumps indirectly through a table indexed
    // by the volatile flag, so the back-edge is a genuine indirect branch (no
    // direct conditional back-edge). After warm-up the target hits the inline IB
    // cache and the loop stays in the code cache.
    void *targets[2];
    targets[0] = &&spin;
    targets[1] = &&done;
    volatile unsigned long x = 0;
spin:
    x = x * 2654435761u + 1;
    goto *targets[got != 0];
done:
    kill(wd, SIGKILL);
    waitpid(wd, 0, 0);
    return 0;
}
