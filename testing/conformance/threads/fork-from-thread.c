// RUN: %cc %s -pthread -o %t && %runner %t
//
// fork() from a multithreaded process. The child is born with exactly one
// thread — the caller, now the leader of a fresh single-threaded process —
// regardless of what that thread was in the parent. So a child forked by a
// worker must be able to exit with its own status, and to execve a new
// image, exactly like any single-threaded process; and none of the parent's
// thread-group bookkeeping (live siblings, pending stops) may leak into it.
//
// Re-exec'd with a marker argument, this binary is the child's new image: it
// asserts it came up as a fresh single-threaded leader (tid == pid).

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

static char *self_path;
static volatile int spin_on = 1;

static void *busy(void *arg) {
    (void) arg;
    while (spin_on) usleep(1000);
    return 0;
}

static int wait_status(pid_t pid) {
    int st;
    if (pid < 0 || waitpid(pid, &st, 0) != pid) return -1;
    if (!WIFEXITED(st)) return -2;
    return WEXITSTATUS(st);
}

// Worker: fork a child that exits plainly; expect its status.
static void *fork_exit_worker(void *arg) {
    (void) arg;
    pid_t pid = fork();
    if (pid == 0) _exit(42);
    return (void *) (long) wait_status(pid);
}

// Worker: fork a child that execs the marker image; expect its status.
static void *fork_exec_worker(void *arg) {
    (void) arg;
    pid_t pid = fork();
    if (pid == 0) {
        char *argv[] = {self_path, "exec'd", NULL};
        char *envp[] = {NULL};
        execve(self_path, argv, envp);
        _exit(99);
    }
    return (void *) (long) wait_status(pid);
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "exec'd") == 0) {
        // No tid == pid check on Darwin: gettid does not exist, and
        // pthread_main_np() reads host-libpthread state that Chimera's
        // in-process exec deliberately carries over (the new image may run
        // on whichever host thread drove the exec). Dissolution is asserted
        // by this image running at all: a surviving old thread would have
        // crashed against the torn-down image or tripped the watchdog.
        #if !defined(__APPLE__)
        if (syscall(SYS_gettid) != getpid()) {
            fprintf(stderr, "exec'd image: gettid %ld != getpid %d\n",
                    syscall(SYS_gettid), getpid());
            return 1;
        }
        #endif
        return 9;
    }
    self_path = argv[0];

    // A busy sibling keeps the parent genuinely multithreaded across every
    // fork, so the children really do inherit a multi-entry live set.
    pthread_t spin;
    if (pthread_create(&spin, NULL, busy, NULL) != 0) return 2;

    pthread_t t;
    void *ret;
    if (pthread_create(&t, NULL, fork_exit_worker, NULL) != 0) return 3;
    pthread_join(t, &ret);
    if ((long) ret != 42) {
        fprintf(stderr, "child of worker fork, plain exit: status %ld, want 42\n", (long) ret);
        return 4;
    }

    if (pthread_create(&t, NULL, fork_exec_worker, NULL) != 0) return 5;
    pthread_join(t, &ret);
    if ((long) ret != 9) {
        fprintf(stderr, "child of worker fork, execve: status %ld, want 9\n", (long) ret);
        return 6;
    }

    // fork from the main thread with the worker still live: same rules.
    pid_t pid = fork();
    if (pid == 0) _exit(5);
    if (wait_status(pid) != 5) {
        fprintf(stderr, "child of main fork: bad status, want 5\n");
        return 7;
    }

    spin_on = 0;
    pthread_join(spin, NULL);
    return 0;
}
