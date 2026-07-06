// RUN: %cc %s -pthread -o %t && %runner %t
//
// pthread_self / pthread_equal identity. Each thread has its own thread
// descriptor (reached through its private TLS), so pthread_self is stable
// within a thread, equal to the id pthread_create handed back, and distinct
// across threads. This leans on the per-thread %fs Chimera sets up for each
// guest thread. Exits 0 only if every identity relation holds.

#include <pthread.h>
#include <stdint.h>

static pthread_t main_id;
static pthread_t worker_id;

static void *worker(void *arg) {
    (void) arg;
    worker_id = pthread_self();
    // Stable within the thread: self equals self.
    return (void *) (intptr_t) (pthread_equal(worker_id, pthread_self()) ? 0 : 1);
}

int main(void) {
    main_id = pthread_self();
    if (!pthread_equal(main_id, pthread_self())) return 1;

    pthread_t t;
    if (pthread_create(&t, NULL, worker, NULL) != 0) return 2;
    void *wret = (void *) 1;
    if (pthread_join(t, &wret) != 0) return 3;
    if (wret != 0) return 4; // worker's own self-equality check failed

    if (pthread_equal(main_id, worker_id)) return 5;  // distinct across threads
    if (!pthread_equal(t, worker_id)) return 6;       // create id == the thread's self
    return 0;
}
