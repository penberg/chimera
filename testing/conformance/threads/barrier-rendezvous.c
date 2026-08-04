// RUN: %cc %s -pthread -o %t && %runner %t
// UNSUPPORTED: darwin -- uses a pthread primitive or shape absent on macOS (pthread_barrier, reader/writer rwlock, __thread, pthread_once, or SysV/unnamed semaphores).
//
// pthread_barrier_wait must hold every thread until all of them arrive, then
// release them together, returning PTHREAD_BARRIER_SERIAL_THREAD to exactly one.
// Eight threads bump an "arrived" counter, wait at a barrier sized for 8, and
// after crossing check that all 8 had arrived — no thread may cross early. If
// the barrier released anyone before the full party assembled, that thread would
// see fewer than 8; if it never released, the test would hang. Exits 0 only if
// every thread saw the full count and exactly one got the serial sentinel.

#include <pthread.h>
#include <stdint.h>

#define N 8

static pthread_barrier_t barrier;
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static int arrived;
static int saw_all;

static void *worker(void *arg) {
    (void) arg;
    pthread_mutex_lock(&mtx);
    arrived++;
    pthread_mutex_unlock(&mtx);

    int rc = pthread_barrier_wait(&barrier);

    pthread_mutex_lock(&mtx);
    if (arrived == N) saw_all++; // every arrival happened before this crossing
    pthread_mutex_unlock(&mtx);

    return (void *) (intptr_t) (rc == PTHREAD_BARRIER_SERIAL_THREAD ? 1 : 0);
}

int main(void) {
    if (pthread_barrier_init(&barrier, NULL, N) != 0) return 1;

    pthread_t t[N];
    for (int i = 0; i < N; i++) {
        if (pthread_create(&t[i], NULL, worker, NULL) != 0) return 2;
    }
    int serial_count = 0;
    for (int i = 0; i < N; i++) {
        void *r;
        if (pthread_join(t[i], &r) != 0) return 3;
        serial_count += (int) (intptr_t) r;
    }
    pthread_barrier_destroy(&barrier);

    if (saw_all != N) return 4;        // someone crossed before all arrived
    if (serial_count != 1) return 5;   // exactly one serial thread
    return 0;
}
