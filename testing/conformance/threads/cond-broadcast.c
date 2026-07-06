// RUN: %cc %s -pthread -o %t && %runner %t
//
// pthread_cond_broadcast must wake every thread waiting on the condition, not
// just one. Eight waiters block in pthread_cond_wait; the main thread waits for
// all of them to be blocked, sets the predicate, and issues a single broadcast.
// All eight must then proceed. If broadcast woke only one (e.g. it forwarded as
// a wake-1 instead of wake-all), the rest would never be joined and the test
// would hang. Exits 0 only if all waiters woke.

#include <pthread.h>
#include <sched.h>

#define WAITERS 8

static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
static int go;    // predicate the broadcast publishes
static int ready; // how many waiters have reached the wait
static int woke;  // how many waiters observed the predicate and proceeded

static void *waiter(void *arg) {
    (void) arg;
    pthread_mutex_lock(&mtx);
    ready++;
    while (!go) pthread_cond_wait(&cv, &mtx);
    woke++;
    pthread_mutex_unlock(&mtx);
    return 0;
}

int main(void) {
    pthread_t t[WAITERS];
    for (int i = 0; i < WAITERS; i++) {
        if (pthread_create(&t[i], NULL, waiter, NULL) != 0) return 1;
    }

    // Wait until every waiter is parked in cond_wait, so the broadcast really
    // has to wake all of them at once.
    pthread_mutex_lock(&mtx);
    while (ready < WAITERS) {
        pthread_mutex_unlock(&mtx);
        sched_yield();
        pthread_mutex_lock(&mtx);
    }
    go = 1;
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mtx);

    for (int i = 0; i < WAITERS; i++) {
        if (pthread_join(t[i], NULL) != 0) return 2;
    }
    return woke == WAITERS ? 0 : 3;
}
