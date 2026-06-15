// RUN: %cc %s -o %t && %runner %t
//
// An SA_SIGINFO handler receives a populated siginfo_t. For a signal sent with
// kill(2), si_code is SI_USER and si_pid/si_uid identify the sender. Sending to
// self gives si_pid == getpid(). Exits 0 only if the handler saw SI_USER and the
// sender's pid/uid.

#include <signal.h>
#include <unistd.h>

static volatile sig_atomic_t got_code;
static volatile sig_atomic_t got_pid;
static volatile sig_atomic_t got_uid;
static volatile sig_atomic_t ran;

static void handler(int sig, siginfo_t *info, void *uc) {
    (void) sig;
    (void) uc;
    got_code = info->si_code;
    got_pid = info->si_pid;
    got_uid = info->si_uid;
    ran = 1;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 1;

    // kill(2) — unlike raise()/tgkill — carries si_code == SI_USER.
    if (kill(getpid(), SIGUSR1) != 0) return 2;

    if (!ran) return 3;
    if (got_code != SI_USER) return 4;
    if (got_pid != getpid()) return 5;
    if (got_uid != (sig_atomic_t) getuid()) return 6;
    return 0;
}
