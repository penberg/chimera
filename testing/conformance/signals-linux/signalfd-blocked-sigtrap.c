// RUN: %cc %s -o %t && %runner %t
//
// A blocked SIGTRAP stays pending in the kernel, where a signalfd reads it.
// SIGTRAP is also how an `int3` reports itself, so a runtime that claims the
// signal for its own breakpoint handling must still leave a blocked one for
// the guest's signalfd rather than consume it.

#include <signal.h>
#include <sys/signalfd.h>
#include <unistd.h>

int main(void) {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGTRAP);
    if (sigprocmask(SIG_BLOCK, &set, NULL) != 0) return 1;
    int fd = signalfd(-1, &set, SFD_NONBLOCK);
    if (fd < 0) return 2;
    if (raise(SIGTRAP) != 0) return 3;
    struct signalfd_siginfo si;
    if (read(fd, &si, sizeof(si)) != sizeof(si)) return 4;
    if (si.ssi_signo != SIGTRAP) return 5;
    return 0;
}
