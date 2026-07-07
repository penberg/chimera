// RUN: %cc %s -o %t && timeout 10 %runner %t
//
// clone3's CLONE_CLEAR_SIGHAND resets the child's signal dispositions the way
// execve does: caught handlers revert to SIG_DFL, SIG_IGN survives, and the
// parent's table is untouched. Chimera must emulate that on the guest's
// virtual table rather than forward the flag — on the host it would flush the
// runtime's own handlers out of the child's slots.

#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#define CLONE_ARGS_SIZE_VER0 64
#ifndef CLONE_CLEAR_SIGHAND
#define CLONE_CLEAR_SIGHAND 0x100000000ULL
#endif

struct clone_args_v0 {
    uint64_t flags;
    uint64_t pidfd;
    uint64_t child_tid;
    uint64_t parent_tid;
    uint64_t exit_signal;
    uint64_t stack;
    uint64_t stack_size;
    uint64_t tls;
};

static void handler(int sig) { (void) sig; }

int main(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handler;
    if (sigaction(SIGUSR1, &sa, NULL) != 0) return 1;
    sa.sa_handler = SIG_IGN;
    if (sigaction(SIGUSR2, &sa, NULL) != 0) return 2;

    struct clone_args_v0 args;
    memset(&args, 0, sizeof(args));
    args.flags = CLONE_CLEAR_SIGHAND;
    args.exit_signal = SIGCHLD;
    long pid = syscall(SYS_clone3, &args, (size_t) CLONE_ARGS_SIZE_VER0);
    if (pid < 0) return 3;
    if (pid == 0) {
        struct sigaction q;
        if (sigaction(SIGUSR1, NULL, &q) != 0) _exit(10);
        if (q.sa_handler != SIG_DFL) _exit(11); // caught handler must be flushed
        if (sigaction(SIGUSR2, NULL, &q) != 0) _exit(12);
        if (q.sa_handler != SIG_IGN) _exit(13); // SIG_IGN must survive
        _exit(0);
    }

    int status = 0;
    if (waitpid((pid_t) pid, &status, 0) != (pid_t) pid) return 4;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return WIFEXITED(status) ? WEXITSTATUS(status) : 5;

    // The parent's own dispositions are not shared with the child's flush.
    struct sigaction q;
    if (sigaction(SIGUSR1, NULL, &q) != 0) return 6;
    if (q.sa_handler != handler) return 7;
    return 0;
}
