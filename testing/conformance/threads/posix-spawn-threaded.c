// RUN: %cc %s -pthread -o %t && timeout 10 %runner %t
//
// posix_spawn from a multithreaded process. Chimera emulates the spawn's
// CLONE_VM child as a fork of its own (multithreaded) host process, so it must
// hold its internal locks across the duplication -- otherwise a lock a busy
// sibling thread held at fork time is orphaned-locked in the single-threaded
// child, and the child deadlocks the moment its execve path (translation,
// mapping, signal reset) reaches for it. Without that, this hangs.
//
// Worker threads keep translating fresh code (recursive calls) while the main
// thread spawns children in a loop; every spawn must succeed and every child
// must exit with the expected code.

#include <pthread.h>
#include <spawn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

#define MARKER "child"
#define CHILD_CODE 33
#define WORKERS 4
#define SPAWNS 20

static atomic_int stop = 0;

// A deep-ish recursion the workers keep running, so the translator is busy
// (and may hold the address-space lock) while the main thread forks to spawn.
static uint64_t burn(uint64_t n) {
    if (n == 0) return 0;
    return n + burn(n - 1);
}

static void *worker(void *arg) {
    (void) arg;
    volatile uint64_t acc = 0;
    while (!atomic_load(&stop)) acc += burn(200);
    (void) acc;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], MARKER) == 0) return CHILD_CODE;

    pthread_t th[WORKERS];
    for (int i = 0; i < WORKERS; i++)
        if (pthread_create(&th[i], 0, worker, 0) != 0) return 1;

    char *child_argv[] = {argv[0], (char *) MARKER, NULL};
    for (int i = 0; i < SPAWNS; i++) {
        pid_t pid;
        if (posix_spawn(&pid, argv[0], NULL, NULL, child_argv, environ) != 0) return 2;
        int status = 0;
        if (waitpid(pid, &status, 0) != pid) return 3;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != CHILD_CODE) return 4;
    }

    atomic_store(&stop, 1);
    for (int i = 0; i < WORKERS; i++) pthread_join(th[i], 0);
    return 0;
}
