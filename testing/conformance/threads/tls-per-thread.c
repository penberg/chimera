// RUN: %cc %s -pthread -o %t && %runner %t
// UNSUPPORTED: darwin -- uses a pthread primitive or shape absent on macOS (pthread_barrier, reader/writer rwlock, __thread, pthread_once, or SysV/unnamed semaphores).
//
// A __thread variable is private to each thread: writes from one thread are
// invisible to others. Eight threads each store a distinct value into the same
// __thread int, rendezvous at a barrier so every store has happened before any
// read-back, then read it; each must still see its own value. If the storage
// were shared (e.g. the per-thread %fs were not really per-thread under Chimera),
// the last writer would win and every thread would read the same value. The main
// thread, which never wrote, must still see 0. Exits 0 only if all are distinct
// and correct.

#include <pthread.h>
#include <stdint.h>

#define N 8

static __thread int tls_val;
static int results[N];
static pthread_barrier_t bar;

static void *worker(void *arg) {
    int id = (int) (intptr_t) arg;
    tls_val = id * 1000 + 7;
    pthread_barrier_wait(&bar); // every thread has written before any reads back
    results[id] = tls_val;      // must still be this thread's own value
    return 0;
}

int main(void) {
    if (pthread_barrier_init(&bar, NULL, N) != 0) return 1;
    pthread_t t[N];
    for (int i = 0; i < N; i++) {
        if (pthread_create(&t[i], NULL, worker, (void *) (intptr_t) i) != 0) return 2;
    }
    for (int i = 0; i < N; i++) pthread_join(t[i], NULL);
    pthread_barrier_destroy(&bar);

    for (int i = 0; i < N; i++) {
        if (results[i] != i * 1000 + 7) return 3; // bled across threads
    }
    if (tls_val != 0) return 4; // main's own copy, never written
    return 0;
}
