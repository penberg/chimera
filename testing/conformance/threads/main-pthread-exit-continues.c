// RUN: %cc %s -pthread -o %t && %runner %t; test $? -eq 7
// UNSUPPORTED: darwin -- exercises Linux exit_group / raw-exit process semantics.
//
// pthread_exit from the main thread does not terminate the process: POSIX keeps
// it alive until the last thread exits. Here main spawns a worker and then calls
// pthread_exit, ending only the main thread. The worker outlives it and, as the
// last thread, calls _exit(7), which is what the process must exit with. If
// main's exit had torn the process down early (the wrong, single-thread
// behaviour), the worker's _exit(7) would never run and the status would not be
// 7 — so the RUN line asserts the exit code is exactly 7.

#include <pthread.h>
#include <time.h>
#include <unistd.h>

static void *worker(void *arg) {
    (void) arg;
    struct timespec ts = {0, 40 * 1000 * 1000}; // outlive main's pthread_exit
    nanosleep(&ts, 0);
    _exit(7); // last thread: exit_group(7) sets the process status
    return 0;
}

int main(void) {
    pthread_t t;
    if (pthread_create(&t, NULL, worker, NULL) != 0) return 1;
    pthread_exit((void *) 0); // end only main; the process must live on
    return 2;                 // unreachable
}
