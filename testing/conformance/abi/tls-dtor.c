// RUN: %cc -x c++ %s -pthread -o %t && %runner %t
//
// A thread_local with a non-trivial destructor. The compiler registers the
// destructor with the loader (`_tlv_atexit` on Darwin, `__cxa_thread_atexit`
// elsewhere), which runs it when the thread ends. A translator must keep that
// registration for itself: the loader is shared with the runtime and would
// call the guest's destructor natively, off the translator — which faults on
// a guest page that is never executable, and would run guest code outside the
// sandbox if it were.
//
// Two worker threads each construct and destroy their own copy, then main
// checks both ran. Deliberately free of any other C++ runtime dependency:
// no exceptions, no library calls in the constructor or destructor.

#include <pthread.h>

static int ctors;
static int dtors;

struct Slot {
    int value;
    Slot() {
        __atomic_add_fetch(&ctors, 1, __ATOMIC_SEQ_CST);
        value = 42;
    }
    ~Slot() {
        // Runs at thread exit, on the thread that owns this copy.
        __atomic_add_fetch(&dtors, 1, __ATOMIC_SEQ_CST);
    }
};

static thread_local Slot slot;

static void *worker(void *) {
    if (slot.value != 42) return (void *) 1; // constructed on first touch
    slot.value = 7;                          // this thread's own copy
    return 0;
}

int main() {
    pthread_t t[2];
    for (int i = 0; i < 2; i++)
        if (pthread_create(&t[i], 0, worker, 0) != 0) return 2;
    for (int i = 0; i < 2; i++) {
        void *r = (void *) 9;
        if (pthread_join(t[i], &r) != 0 || r != 0) return 3;
    }

    // Both workers have been joined, so both destructors have run.
    if (__atomic_load_n(&ctors, __ATOMIC_SEQ_CST) != 2) return 4;
    if (__atomic_load_n(&dtors, __ATOMIC_SEQ_CST) != 2) return 5;
    return 0;
}
