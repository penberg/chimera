// RUN: %cc -x c++ %s -pthread -o %t && %runner %t
//
// C++ thread_local with a dynamic initializer: the constructor must run once
// per thread, at that thread's first touch, and each thread must get its own
// copy. Modern clang lowers this to inline guard-based lazy initialization in
// the wrapper function (no linker/loader involvement), so this exercises the
// whole TLS access path — descriptor thunks and per-thread block
// materialization included — under each thread. Deliberately avoids any C++
// runtime dependency: no destructor (no __cxa_thread_atexit), no exceptions,
// no library calls in the constructor.

#include <pthread.h>

static int ctor_calls;

struct Slot {
    int value;
    Slot() {
        __atomic_add_fetch(&ctor_calls, 1, __ATOMIC_SEQ_CST);
        value = 42;
    }
};

static thread_local Slot slot;

static void *worker(void *) {
    if (slot.value != 42) return (void *) 1; // a fresh, constructed copy
    slot.value = 7;
    if (slot.value != 7) return (void *) 2;
    return 0;
}

int main() {
    if (slot.value != 42) return 1; // constructed at main's first touch
    slot.value = 5;

    pthread_t t;
    void *r = (void *) 9;
    if (pthread_create(&t, 0, worker, 0) != 0) return 2;
    if (pthread_join(t, &r) != 0 || r != 0) return 3;

    if (slot.value != 5) return 4; // the worker wrote its own copy
    if (__atomic_load_n(&ctor_calls, __ATOMIC_SEQ_CST) != 2) return 5; // once per thread
    return 0;
}
