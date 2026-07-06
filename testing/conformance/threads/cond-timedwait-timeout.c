// RUN: %cc %s -pthread -o %t && %runner %t
//
// pthread_cond_timedwait must return ETIMEDOUT when the deadline passes with no
// signal — and it must actually wait until then, not return early. This drives
// a futex wait with an absolute timeout (forwarded to the host under Chimera).
// The test waits 100ms on a condvar nobody signals and checks both the return
// code and that at least ~80ms elapsed, so a broken timeout that returned
// immediately would fail rather than pass. Exits 0 only if both hold.

#include <errno.h>
#include <pthread.h>
#include <time.h>

int main(void) {
    pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cv = PTHREAD_COND_INITIALIZER; // default clock: CLOCK_REALTIME

    // Absolute deadline 100ms out, on the clock the condvar uses.
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_nsec += 100 * 1000 * 1000;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000L;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    pthread_mutex_lock(&m);
    int rc = pthread_cond_timedwait(&cv, &m, &deadline);
    pthread_mutex_unlock(&m);

    clock_gettime(CLOCK_MONOTONIC, &end);

    if (rc != ETIMEDOUT) return 1;

    long elapsed_ms =
        (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    if (elapsed_ms < 80) return 2; // must have actually blocked, not returned early
    return 0;
}
