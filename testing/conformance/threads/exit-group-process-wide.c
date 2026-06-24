// RUN: %cc %s -pthread -o %t && %runner %t
//
// exit_group terminates the whole thread group, not just the calling thread.
// A worker thread calling _exit (which is exit_group) must end the entire
// process — including the main thread — with the worker's status code. The main
// thread here never returns on its own: it loops forever issuing a short sleep,
// so the only way this program terminates with status 0 is if the worker's
// exit_group tears the whole process down. If exit_group were treated as a
// thread-local exit (ending only the worker), main would loop until the suite's
// per-test timeout and the test would fail.
//
// The main thread is deliberately woken often (a 1ms sleep) rather than blocked
// indefinitely, so it observes the process-exit request promptly at a syscall
// boundary.

#include <pthread.h>
#include <time.h>
#include <unistd.h>

static void *worker(void *arg) {
    (void) arg;
    struct timespec ts = {0, 5 * 1000 * 1000}; // let main reach its sleep loop
    nanosleep(&ts, 0);
    _exit(0); // glibc _exit -> exit_group(0): must end the whole process
    return 0;
}

int main(void) {
    pthread_t t;
    if (pthread_create(&t, NULL, worker, NULL) != 0) return 1;
    for (;;) {
        struct timespec ts = {0, 1000 * 1000}; // 1ms; returns to the run loop
        nanosleep(&ts, 0);
    }
    return 2; // unreachable; reaching here would mean exit_group did not fire
}
