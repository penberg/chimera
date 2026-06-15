// RUN: %cc %s -pthread -o %t && %runner %t
//
// Lock-prefixed atomics must stay atomic through translation. Eight threads
// each do 200k __atomic_fetch_add on one shared counter — no mutex, no futex,
// just a contended `lock xadd`. The only thing making the total come out exact
// is the LOCK prefix on that instruction; if the translator dropped or split it,
// concurrent read-modify-writes would race and lose increments, leaving the
// total short. Same-ISA translation must copy the prefix verbatim. Exits 0 only
// if the count is exact.

#include <pthread.h>

#define THREADS 8
#define ITERS 200000

static long counter;

static void *worker(void *arg) {
    (void) arg;
    for (int i = 0; i < ITERS; i++) {
        __atomic_fetch_add(&counter, 1, __ATOMIC_SEQ_CST);
    }
    return 0;
}

int main(void) {
    pthread_t t[THREADS];
    for (int i = 0; i < THREADS; i++) {
        if (pthread_create(&t[i], NULL, worker, NULL) != 0) return 1;
    }
    for (int i = 0; i < THREADS; i++) {
        if (pthread_join(t[i], NULL) != 0) return 2;
    }
    return counter == (long) THREADS * ITERS ? 0 : 3;
}
