// RUN: %cc %s -pthread -o %t && %runner %t
// UNSUPPORTED: darwin -- gettid/tgkill are Linux-specific.
//
// gettid is per-thread; getpid is per-process. Each guest thread runs on its own
// host thread, so gettid (forwarded to the host) returns a distinct kernel TID,
// while getpid returns the one shared process id every thread sees. The main
// thread is special: its tid equals the pid. Exits 0 only if every relation
// holds — distinct tids, one shared pid, and main's tid == pid.

#define _GNU_SOURCE
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

#define WORKERS 4

static pid_t tids[WORKERS];
static pid_t pids[WORKERS];

static long gettid_(void) { return syscall(SYS_gettid); }

static void *worker(void *arg) {
    long i = (long) arg;
    tids[i] = (pid_t) gettid_();
    pids[i] = getpid();
    return 0;
}

int main(void) {
    pid_t pid = getpid();
    if ((pid_t) gettid_() != pid) return 1; // main thread: tid == pid

    pthread_t t[WORKERS];
    for (long i = 0; i < WORKERS; i++)
        if (pthread_create(&t[i], NULL, worker, (void *) i) != 0) return 2;
    for (int i = 0; i < WORKERS; i++) pthread_join(t[i], NULL);

    for (int i = 0; i < WORKERS; i++) {
        if (pids[i] != pid) return 3;       // getpid is shared
        if (tids[i] == pid) return 4;       // a worker's tid is not the pid
        for (int j = i + 1; j < WORKERS; j++)
            if (tids[i] == tids[j]) return 5; // tids are distinct from each other
    }
    return 0;
}
