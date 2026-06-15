// RUN: %cc %s -pthread -o %t && %runner %t
//
// A read/write lock must let readers share but keep writers exclusive. Two
// invariants, checked deterministically:
//
//  - Reader sharing: every reader takes a read lock and rendezvouses at a
//    barrier while still holding it. A lock that serialized readers (acted like
//    a mutex) could never get all of them inside at once, so the barrier would
//    deadlock. (Its own rwlock, untouched by writers, so no writer-preference
//    can stall it.)
//  - Exclusion: writers increment a plain counter under the write lock — its
//    final value is exact only if writers never overlap — and set a "writing"
//    flag while inside; a reader that ever observes the flag set while holding
//    a read lock proves a writer was concurrent. Neither may happen.
//
// Exits 0 only if both invariants hold.

#include <pthread.h>

#define READERS 6
#define WRITERS 4
#define ITERS 30000

static pthread_rwlock_t share = PTHREAD_RWLOCK_INITIALIZER;
static pthread_barrier_t sbar;

static pthread_rwlock_t excl = PTHREAD_RWLOCK_INITIALIZER;
static long counter;
static volatile int writing;
static int reader_saw_write;

static void *reader(void *arg) {
    (void) arg;
    // Sharing: all READERS hold a read lock together (or this deadlocks).
    pthread_rwlock_rdlock(&share);
    pthread_barrier_wait(&sbar);
    pthread_rwlock_unlock(&share);

    // Exclusion: no writer may be inside while we hold a read lock.
    for (int i = 0; i < ITERS; i++) {
        pthread_rwlock_rdlock(&excl);
        if (writing) __atomic_store_n(&reader_saw_write, 1, __ATOMIC_RELAXED);
        pthread_rwlock_unlock(&excl);
    }
    return 0;
}

static void *writer(void *arg) {
    (void) arg;
    for (int i = 0; i < ITERS; i++) {
        pthread_rwlock_wrlock(&excl);
        writing = 1;
        counter++; // non-atomic: exact only under exclusive write locking
        writing = 0;
        pthread_rwlock_unlock(&excl);
    }
    return 0;
}

int main(void) {
    if (pthread_barrier_init(&sbar, NULL, READERS) != 0) return 1;

    pthread_t r[READERS], w[WRITERS];
    for (int i = 0; i < READERS; i++) {
        if (pthread_create(&r[i], NULL, reader, NULL) != 0) return 2;
    }
    for (int i = 0; i < WRITERS; i++) {
        if (pthread_create(&w[i], NULL, writer, NULL) != 0) return 3;
    }
    for (int i = 0; i < READERS; i++) pthread_join(r[i], NULL);
    for (int i = 0; i < WRITERS; i++) pthread_join(w[i], NULL);
    pthread_barrier_destroy(&sbar);

    if (counter != (long) WRITERS * ITERS) return 4; // writer-writer exclusion
    if (reader_saw_write) return 5;                   // reader-writer exclusion
    return 0;
}
