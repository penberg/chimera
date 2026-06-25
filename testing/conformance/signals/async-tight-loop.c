// RUN: %cc %s -o %t && %runner %t
//
// An asynchronous signal must reach the guest handler even while the guest is in
// a tight, syscall-free compute loop fully linked in the code cache. Such a loop
// closes via a conditional back-edge that never returns to the run loop, so
// before the safepoint poll a pending SIGALRM sat undelivered and this spun
// forever. A watchdog child hard-kills us if that regresses, so the failure is a
// bounded non-zero exit rather than a hung suite.

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

    // Watchdog: kill us after 5s if the signal never gets through, so a
    // regression terminates (failing) instead of hanging forever.
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

    // Tight, syscall-free reduction loop the translator fully links; the volatile
    // flag is reloaded each iteration and the loop closes via a conditional
    // back-edge whose safepoint poll must observe the pending signal.
    volatile unsigned long x = 0;
    while (!got) {
        x = x * 2654435761u + 1;
    }

    kill(wd, SIGKILL);
    waitpid(wd, 0, 0);
    return 0;
}
