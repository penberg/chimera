// RUN: %cc %s -pthread -o %t && %runner %t
//
// exit_group must end the whole process even when other threads — including the
// main thread — are parked indefinitely in a host syscall. Here main blocks in
// pthread_join on a worker that never returns, so it sits forever in a futex
// wait; a second worker then calls _exit (exit_group). The kernel's exit_group
// tears down every task regardless of what it is doing, so the process must
// still exit with status 0. Chimera has to interrupt the thread blocked in the
// kernel so it observes the process-exit request rather than waiting for the
// blocking syscall to return on its own (which here it never would).

#include <pthread.h>
#include <time.h>
#include <unistd.h>

static void *blocker(void *arg) {
    (void) arg;
    for (;;) pause(); // never returns on its own
    return 0;
}

static void *killer(void *arg) {
    (void) arg;
    struct timespec ts = {0, 10 * 1000 * 1000}; // 10ms: let main reach the join
    nanosleep(&ts, 0);
    _exit(0); // exit_group(0): must terminate the whole process
    return 0;
}

int main(void) {
    pthread_t a, b;
    if (pthread_create(&a, NULL, blocker, NULL) != 0) return 1;
    if (pthread_create(&b, NULL, killer, NULL) != 0) return 1;
    pthread_join(a, NULL); // blocks forever unless exit_group tears us down
    return 2;              // reaching here means exit_group did not end us
}
