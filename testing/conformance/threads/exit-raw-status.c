// RUN: %cc %s -pthread -o %t && %runner %t
// UNSUPPORTED: darwin -- exercises Linux exit_group / raw-exit process semantics.
//
// A raw SYS_exit must terminate the process with the right status. Absent an
// exit_group, the kernel reports the status of the *last* thread to exit as the
// process's wait(2) status — whichever thread that is. Each case runs in a
// forked child so the parent can assert the wait status.
//
// The single-threaded and "main exits last" cases have a determinate answer
// (the latter via pthread_join). The "worker outlives main" cases do NOT: which
// of two threads exiting independently records its status last is a scheduling
// race, native included — and Chimera makes it unobservable, since the initial
// thread's host thread lingers until the whole group drains, so a sibling
// cannot synchronize on main's exit (a futex on main's TID would deadlock
// against the wait). So those cases accept *either* valid last-exiter status;
// a pipe biases the worker to be the last one, exercising the superseding path
// in the common case, while a bogus status (e.g. 0 from the pre-fix bug that
// reported the group as exited before any thread's status was adopted) still
// fails.

#include <pthread.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

// One byte written by the initial thread immediately before its SYS_exit; the
// worker blocks reading it, so it usually exits after main without depending on
// absolute timing.
static int order[2];

static void wait_until_main_exits(void) {
    char b;
    while (read(order[0], &b, 1) != 1) {
    }
}

static void *outlive_raw_exit_9(void *arg) {
    (void) arg;
    wait_until_main_exits();
    syscall(SYS_exit, 9);
    return 0;
}

static void *outlive_return(void *arg) {
    (void) arg;
    wait_until_main_exits();
    return 0; // glibc ends the thread with SYS_exit(0)
}

static void *nap_short(void *arg) {
    (void) arg;
    usleep(10 * 1000);
    return 0;
}

static int wait_status(pid_t pid) {
    int st;
    if (pid < 0 || waitpid(pid, &st, 0) != pid) return -1;
    if (!WIFEXITED(st)) return -2;
    return WEXITSTATUS(st);
}

static int check(const char *what, pid_t pid, int want) {
    int st = wait_status(pid);
    if (st != want) {
        fprintf(stderr, "%s: status %d, want %d\n", what, st, want);
        return 1;
    }
    return 0;
}

// Either thread may legitimately be the last to exit; accept both statuses.
static int check_either(const char *what, pid_t pid, int a, int b) {
    int st = wait_status(pid);
    if (st != a && st != b) {
        fprintf(stderr, "%s: status %d, want %d or %d\n", what, st, a, b);
        return 1;
    }
    return 0;
}

int main(void) {
    // Single-threaded: SYS_exit(42) is the process status.
    pid_t pid = fork();
    if (pid == 0) syscall(SYS_exit, 42);
    if (check("single-threaded SYS_exit(42)", pid, 42)) return 1;

    // The initial thread raw-exits(7); a worker outlives it and raw-exits(9).
    // The last exiter's status wins — normally the worker's 9, but the initial
    // thread's 7 if it is scheduled to exit last.
    pid = fork();
    if (pid == 0) {
        if (pipe(order) != 0) syscall(SYS_exit, 98);
        pthread_t t;
        if (pthread_create(&t, NULL, outlive_raw_exit_9, NULL) != 0) syscall(SYS_exit, 99);
        char b = 1;
        if (write(order[1], &b, 1) != 1) syscall(SYS_exit, 97);
        syscall(SYS_exit, 7);
    }
    if (check_either("worker outlives main, raw SYS_exit(9)", pid, 9, 7)) return 2;

    // Same, but the surviving worker just returns: glibc ends it with
    // SYS_exit(0), so the status is 0 (worker last) or 7 (initial thread last).
    pid = fork();
    if (pid == 0) {
        if (pipe(order) != 0) syscall(SYS_exit, 98);
        pthread_t t;
        if (pthread_create(&t, NULL, outlive_return, NULL) != 0) syscall(SYS_exit, 99);
        char b = 1;
        if (write(order[1], &b, 1) != 1) syscall(SYS_exit, 97);
        syscall(SYS_exit, 7);
    }
    if (check_either("worker outlives main, plain return", pid, 0, 7)) return 3;

    // Every worker is already gone when the initial thread raw-exits: it is
    // the last thread, so its status is the process's. The join makes the
    // ordering deterministic.
    pid = fork();
    if (pid == 0) {
        pthread_t t;
        if (pthread_create(&t, NULL, nap_short, NULL) != 0) syscall(SYS_exit, 99);
        pthread_join(t, NULL);
        syscall(SYS_exit, 5);
    }
    if (check("main exits last", pid, 5)) return 4;

    return 0;
}
