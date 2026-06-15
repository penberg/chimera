// RUN: %cc %s -pthread -o %t && %runner %t
//
// A contended mutex must serialize access to a shared counter exactly. Eight
// threads each increment a shared counter 100k times under one mutex; the total
// must be THREADS*ITERS with no lost updates. This stresses several things at
// once under Chimera: the mutex's uncontended fast path (a translated atomic
// compare-exchange — the LOCK prefix must survive translation) and its
// contended slow path (futex sleep/wake, forwarded to the host), all while
// every thread translates and runs out of the one shared code cache. A dropped
// LOCK prefix, a lost wakeup, or a code-cache race would show up as a wrong
// total or a hang.

#include <pthread.h>

#define THREADS 8
#define ITERS 100000

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static long counter;

static void *worker(void *arg) {
    (void) arg;
    for (int i = 0; i < ITERS; i++) {
        pthread_mutex_lock(&lock);
        counter++;
        pthread_mutex_unlock(&lock);
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
