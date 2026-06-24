// RUN: %cc %s -pthread -o %t && %runner %t
//
// A raw SYS_exit must terminate the process with the right status. Absent an
// exit_group, the kernel reports the status of the *last* thread to exit as
// the process's wait(2) status — whichever thread that is: a single-threaded
// SYS_exit(n) reports n, an initial thread that raw-exits while a worker still
// runs is outlived and superseded by the worker's status, and an initial
// thread that raw-exits after every worker is done reports its own. Each case
// runs in a forked child so the parent can assert the wait status.

#include <pthread.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

static void *nap_return(void *arg) {
    (void) arg;
    usleep(150 * 1000);
    return 0;
}

static void *nap_raw_exit_9(void *arg) {
    (void) arg;
    usleep(150 * 1000);
    syscall(SYS_exit, 9);
    return 0;
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

int main(void) {
    // Single-threaded: SYS_exit(42) is the process status.
    pid_t pid = fork();
    if (pid == 0) syscall(SYS_exit, 42);
    if (check("single-threaded SYS_exit(42)", pid, 42)) return 1;

    // The initial thread raw-exits first; a worker outlives it and raw-exits
    // with 9. The last exiter's status wins.
    pid = fork();
    if (pid == 0) {
        pthread_t t;
        if (pthread_create(&t, NULL, nap_raw_exit_9, NULL) != 0) syscall(SYS_exit, 99);
        syscall(SYS_exit, 7);
    }
    if (check("worker outlives main, raw SYS_exit(9)", pid, 9)) return 2;

    // Same, but the surviving worker just returns: glibc ends it with
    // SYS_exit(0), so the process reports 0 — not the initial thread's 7.
    pid = fork();
    if (pid == 0) {
        pthread_t t;
        if (pthread_create(&t, NULL, nap_return, NULL) != 0) syscall(SYS_exit, 99);
        syscall(SYS_exit, 7);
    }
    if (check("worker outlives main, plain return", pid, 0)) return 3;

    // Every worker is already gone when the initial thread raw-exits: it is
    // the last thread, so its status is the process's.
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
