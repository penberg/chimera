// RUN: %cc %s -pthread -o %t && %runner %t
// UNSUPPORTED: darwin -- uses a pthread primitive or shape absent on macOS (pthread_barrier, reader/writer rwlock, __thread, pthread_once, or SysV/unnamed semaphores).
//
// Stress a rwlock's writer path across the thread-startup window.
//
// A freshly spawned thread must observe its own TID (glibc's TCB `pd->tid`) from
// its first instruction, exactly as the kernel guarantees by writing the set-TID
// word at clone time; Chimera spawns the host thread itself and replicates that
// write before the child runs any guest code. The TID is a rwlock writer's
// identity: `pthread_rwlock` records `__cur_writer` on wrlock and picks the
// writer- vs reader-release unlock path by comparing it against `pd->tid`. A
// writer that ever ran with a stale TID would mismatch the owner check on
// unlock, take the reader-release path, and underflow `__readers` to
// ~0xfffffff8 — wedging the lock.
//
// This hammers that startup window: many short rounds, each spawning several
// writers on one shared rwlock and joining them. After every round the lock must
// be back to a clean, zero-reader state and still fully functional; a failure
// shows up as a hung join (caught by the suite's per-test timeout) or a corrupt
// `__readers` word. Oversubscription (more writers than cores) widens the
// window, so this leans on a busy/loaded runner — the 2-core CI host
// oversubscribes automatically.

#include <pthread.h>
#include <stdio.h>

#define ROUNDS 400
#define WRITERS 4
#define ITERS 5000

static pthread_rwlock_t lk = PTHREAD_RWLOCK_INITIALIZER;
static long counter;

static void *writer(void *arg) {
    (void) arg;
    for (int i = 0; i < ITERS; i++) {
        pthread_rwlock_wrlock(&lk);
        counter++;
        pthread_rwlock_unlock(&lk);
    }
    return 0;
}

int main(void) {
    for (int r = 0; r < ROUNDS; r++) {
        pthread_t t[WRITERS];
        for (int i = 0; i < WRITERS; i++) {
            if (pthread_create(&t[i], NULL, writer, NULL) != 0) {
                fprintf(stderr, "pthread_create failed (round %d)\n", r);
                return 1;
            }
        }
        for (int i = 0; i < WRITERS; i++) pthread_join(t[i], NULL);
        // No thread holds the lock now and there were never any readers, so the
        // rwlock's reader-count field (`__readers`, offset 0, bits 3 and up) must
        // be empty; only the low phase/lock bits are ever set at rest. The
        // underflow leaves a huge bogus value (~0xfffffff8), caught here.
        unsigned readers = *(volatile unsigned *) &lk;
        if (readers > 0x10000) {
            fprintf(stderr, "rwlock corrupt after round %d: __readers=0x%08x\n", r, readers);
            return 2;
        }
    }
    if (counter != (long) ROUNDS * WRITERS * ITERS) {
        fprintf(stderr, "lost updates: counter=%ld\n", counter);
        return 3;
    }
    return 0;
}
