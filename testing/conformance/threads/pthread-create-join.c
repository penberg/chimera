// RUN: %cc %s -pthread -o %t && %runner %t
//
// A real glibc pthread: create a thread, have it return a value, join it, and
// check the value came back. Modern glibc implements pthread_create with clone3
// and pthread_join with a futex on the joinee's TID word (cleared and woken on
// exit). This exercises the whole thread lifecycle end to end — clone3 spawn,
// CLONE_SETTLS TLS so glibc's thread setup runs, and the clear-tid/futex-wake
// join — through the public API rather than raw clone. Exits 0 only if the join
// returned the worker's value.

#include <pthread.h>
#include <stdint.h>

static void *worker(void *arg) {
    return (void *) ((intptr_t) arg + 1);
}

int main(void) {
    pthread_t t;
    if (pthread_create(&t, NULL, worker, (void *) 41) != 0) {
        return 1;
    }
    void *ret = NULL;
    if (pthread_join(t, &ret) != 0) {
        return 2;
    }
    return (intptr_t) ret == 42 ? 0 : 3;
}
