// RUN: %cc %s -pthread -o %t && %runner %t
//
// Mutex semantics beyond the plain lock/unlock: a recursive mutex can be locked
// again by the thread that already holds it (and needs a matching unlock per
// lock), and pthread_mutex_trylock reports EBUSY on a mutex that is already
// held rather than blocking. These ride on glibc's mutex state machine
// (owner/recursion-count tracking and atomic compare-exchange), translated and
// run under Chimera. Exits 0 only if every step returns what POSIX requires.

#include <errno.h>
#include <pthread.h>

int main(void) {
    // Recursive mutex: the holding thread may lock it again.
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr) != 0) return 1;
    if (pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE) != 0) return 2;
    pthread_mutex_t rec;
    if (pthread_mutex_init(&rec, &attr) != 0) return 3;
    if (pthread_mutex_lock(&rec) != 0) return 4;
    if (pthread_mutex_lock(&rec) != 0) return 5;   // re-lock by the owner succeeds
    if (pthread_mutex_unlock(&rec) != 0) return 6; // one unlock per lock
    if (pthread_mutex_unlock(&rec) != 0) return 7;
    if (pthread_mutex_trylock(&rec) != 0) return 8; // fully released: trylock now wins
    if (pthread_mutex_unlock(&rec) != 0) return 9;
    pthread_mutex_destroy(&rec);
    pthread_mutexattr_destroy(&attr);

    // trylock: succeeds when free, EBUSY when already held, succeeds once freed.
    pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
    if (pthread_mutex_trylock(&m) != 0) return 10;
    if (pthread_mutex_trylock(&m) != EBUSY) return 11;
    if (pthread_mutex_unlock(&m) != 0) return 12;
    if (pthread_mutex_trylock(&m) != 0) return 13;
    if (pthread_mutex_unlock(&m) != 0) return 14;
    return 0;
}
