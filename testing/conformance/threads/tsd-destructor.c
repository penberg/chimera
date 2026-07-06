// RUN: %cc %s -pthread -o %t && %runner %t
//
// A pthread_key_create destructor runs when a thread that set a value through
// that key exits. glibc walks the thread's thread-specific data on the exit path
// and calls the destructor with the stored value — ordinary guest code on the
// thread's teardown, so it must run under Chimera too. The worker thread sets a
// value and returns; by the time the joiner observes it, the destructor must
// have run exactly once with the value the worker stored. Exits 0 only if so.

#include <pthread.h>
#include <stdatomic.h>

static pthread_key_t key;
static atomic_int destroyed;
static void *seen;

static void destructor(void *value) {
    seen = value;
    atomic_fetch_add(&destroyed, 1);
}

static void *worker(void *arg) {
    pthread_setspecific(key, arg);
    return 0;
}

int main(void) {
    if (pthread_key_create(&key, destructor) != 0) return 1;

    int marker = 0;
    pthread_t t;
    if (pthread_create(&t, NULL, worker, &marker) != 0) return 2;
    if (pthread_join(t, NULL) != 0) return 3;

    if (atomic_load(&destroyed) != 1) return 4; // ran exactly once
    if (seen != &marker) return 5;              // with the value the worker set
    return 0;
}
