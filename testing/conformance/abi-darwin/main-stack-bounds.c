// RUN: %cc %s -pthread -o %t && %runner %t
//
// libpthread's stack-bounds queries must describe the stack a thread actually
// runs on. Under Chimera the guest's main thread runs on a runtime-allocated
// stack while the host pthread records the host main stack, so the dispatch
// loop answers pthread_get_stackaddr_np/pthread_get_stacksize_np for the main
// thread itself; JavaScriptCore's stack sanitizer release-asserts if the
// current stack pointer falls outside the reported bounds. Worker threads
// exercise the untouched path: their host pthreads are created directly on
// the guest stack. Exits 0 only if both threads' stack pointers lie inside
// their reported bounds.

#include <pthread.h>
#include <stdint.h>

static int sp_within_bounds(void) {
    pthread_t self = pthread_self();
    uintptr_t base = (uintptr_t) pthread_get_stackaddr_np(self);
    size_t size = pthread_get_stacksize_np(self);
    volatile char probe;
    uintptr_t sp = (uintptr_t) &probe;
    return sp < base && sp >= base - size;
}

static void *worker(void *arg) {
    *(int *) arg = sp_within_bounds();
    return 0;
}

int main(void) {
    if (!sp_within_bounds()) return 1;

    int worker_ok = 0;
    pthread_t t;
    if (pthread_create(&t, 0, worker, &worker_ok)) return 2;
    if (pthread_join(t, 0)) return 3;
    if (!worker_ok) return 4;
    return 0;
}
