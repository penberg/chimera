// RUN: %cc %s -pthread -o %t && %runner %t
//
// pthread_once must run its init routine exactly once no matter how many threads
// race to call it, with the losers blocking until the winner finishes. Sixteen
// threads rendezvous at a barrier so they all hit pthread_once at once, then call
// it; the init function counts its own invocations. The count must be 1 and every
// thread must return (proving the losers waited for, and saw, the completed
// init). pthread_once rides an atomic state word plus a futex wait, so this leans
// on both under Chimera. Exits 0 only if init ran once and all threads returned.

#include <pthread.h>

#define THREADS 16

static pthread_once_t once = PTHREAD_ONCE_INIT;
static pthread_barrier_t start;
static int init_count;
static int observers;

static void init_fn(void) {
    __atomic_fetch_add(&init_count, 1, __ATOMIC_SEQ_CST);
}

static void *worker(void *arg) {
    (void) arg;
    pthread_barrier_wait(&start); // all threads call pthread_once together
    pthread_once(&once, init_fn);
    __atomic_fetch_add(&observers, 1, __ATOMIC_SEQ_CST);
    return 0;
}

int main(void) {
    if (pthread_barrier_init(&start, NULL, THREADS) != 0) return 1;
    pthread_t t[THREADS];
    for (int i = 0; i < THREADS; i++) {
        if (pthread_create(&t[i], NULL, worker, NULL) != 0) return 2;
    }
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
    pthread_barrier_destroy(&start);

    if (init_count != 1) return 3;      // init ran exactly once
    if (observers != THREADS) return 4; // every thread returned from pthread_once
    return 0;
}
