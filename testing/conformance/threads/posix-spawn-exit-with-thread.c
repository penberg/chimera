// RUN: %cc %s -pthread -o %t && %runner %t
//
// glibc's posix_spawn is clone3(CLONE_VM | CLONE_VFORK | CLONE_CLEAR_SIGHAND).
// Chimera emulates the spawn as a host fork, and CLONE_CLEAR_SIGHAND must not
// reach that fork: the kernel would flush the runtime's own host handlers out
// of the child's slots, and the child would then die with Chimera's reserved
// realtime signal the first time exit_group interrupts a sibling parked in a
// blocking syscall. So the spawned child here parks a thread of its own in a
// blocking read and exits from main — the parent must see the child's exit
// code, not a death by signal.

#include <pthread.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

#define MARKER "child"
#define CHILD_CODE 42

static int ready[2], park[2];

static void *parked(void *arg) {
    (void) arg;
    char b = 'r';
    if (write(ready[1], &b, 1) != 1) _exit(101);
    (void) read(park[0], &b, 1); // blocks until exit_group tears the thread down
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], MARKER) == 0) {
        if (pipe(ready) != 0 || pipe(park) != 0) return 102;
        pthread_t th;
        if (pthread_create(&th, 0, parked, 0) != 0) return 103;
        char b;
        if (read(ready[0], &b, 1) != 1) return 104;
        usleep(50 * 1000); // let the thread reach the blocking read
        exit(CHILD_CODE);
    }

    char *child_argv[] = {argv[0], (char *) MARKER, NULL};
    pid_t pid;
    if (posix_spawn(&pid, argv[0], NULL, NULL, child_argv, environ) != 0) return 1;
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 2;
    if (!WIFEXITED(status)) return 3;
    if (WEXITSTATUS(status) != CHILD_CODE) return 4;
    return 0;
}
