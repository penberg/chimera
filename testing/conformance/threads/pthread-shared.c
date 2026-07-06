// RUN: %cc %s -pthread -o %t && %runner %t
//
// Threads share the creator's memory (CLONE_VM) and open file descriptors
// (CLONE_FILES). Under Chimera the guest threads are host threads of one
// process, so both the address space and the host fd table are shared. The
// worker mutates a global the main thread set and reads through a pipe fd the
// main thread opened; the parent checks both took effect. Exits 0 only if so.

#include <pthread.h>
#include <stdint.h>
#include <unistd.h>

static int shared_global;
static int pipe_rd;
static volatile char got;

static void *worker(void *arg) {
    (void) arg;
    shared_global += 100;             // sees and mutates the main thread's global
    char c = 0;
    ssize_t n = read(pipe_rd, &c, 1); // reads through the fd the main thread opened
    got = c;
    return (void *) (intptr_t) (n == 1 ? 0 : 1);
}

int main(void) {
    shared_global = 5;

    int p[2];
    if (pipe(p) != 0) return 1;
    pipe_rd = p[0];
    if (write(p[1], "Z", 1) != 1) return 2;

    pthread_t t;
    if (pthread_create(&t, NULL, worker, NULL) != 0) return 3;
    void *wret = (void *) 1;
    if (pthread_join(t, &wret) != 0) return 4;

    if (wret != 0) return 5;            // worker's read through the shared fd failed
    if (shared_global != 105) return 6; // CLONE_VM: the global was not shared
    if (got != 'Z') return 7;           // CLONE_FILES: wrong fd or data
    return 0;
}
