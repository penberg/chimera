// RUN: %cc %s -pthread -o %t && %runner %t
//
// errno is per-thread. Two threads concurrently make *different* failing
// syscalls in a tight loop — one gets EBADF, the other EINVAL — and each checks
// that errno holds its own value every time. If errno were shared (not the
// per-thread __thread variable glibc makes it), the two would clobber each
// other's and a check would fail. This leans on the per-thread %fs Chimera
// installs and on errno being set correctly after each forwarded syscall. Exits
// 0 only if neither thread ever saw the other's errno.

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

#define ITERS 100000

static void *worker(void *arg) {
    int expected = (int) (intptr_t) arg;
    for (int i = 0; i < ITERS; i++) {
        errno = 0;
        if (expected == EBADF) {
            close(-1); // bad fd -> EBADF
        } else {
            struct timespec bad = {-1, -1};
            nanosleep(&bad, NULL); // out-of-range -> EINVAL
        }
        if (errno != expected) return (void *) 1; // saw another thread's errno
    }
    return 0;
}

int main(void) {
    pthread_t a, b;
    if (pthread_create(&a, NULL, worker, (void *) (intptr_t) EBADF) != 0) return 1;
    if (pthread_create(&b, NULL, worker, (void *) (intptr_t) EINVAL) != 0) return 2;
    void *ra, *rb;
    pthread_join(a, &ra);
    pthread_join(b, &rb);
    return (ra == 0 && rb == 0) ? 0 : 3;
}
