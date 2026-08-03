// RUN: %cc %s -o %t && timeout 10 %runner %t
//
// Installing a seccomp filter must report success and leave the process able
// to keep making syscalls and to exec. glycin's loader sandbox installs its
// allowlist right before exec'ing the loader; a sandbox that fails the
// install (or dies under it) loses every GNOME image load. The filter here
// allows everything, so the expectation is identical native and sandboxed —
// what is being pinned is that the install itself succeeds and that syscalls
// and a subsequent exec still work.

#define _GNU_SOURCE

#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
    struct sock_filter allow_all[] = {
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog prog = {
        .len = 1,
        .filter = allow_all,
    };

    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) return 1;
    if (syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog) != 0) return 2;
    if (getpid() <= 0) return 3;

    // The filtered process must still be able to spawn and exec.
    pid_t pid = fork();
    if (pid < 0) return 4;
    if (pid == 0) {
        execl("/usr/bin/true", "true", (char *)NULL);
        _exit(5);
    }
    int status;
    if (waitpid(pid, &status, 0) != pid) return 6;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 7;
}
