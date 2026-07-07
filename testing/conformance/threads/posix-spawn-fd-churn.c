// RUN: %cc %s -pthread -o %t && timeout 20 %runner %t
//
// posix_spawn while sibling threads churn the descriptor table. A handler
// that virtualizes descriptors pairs its table with kernel state (reserved
// numbers, backing fds), and the spawn's fork copies both; a copy landing
// between the two halves of an update leaves the child a table entry whose
// number the kernel has already freed. The number gets reused — by the exec
// loader's own image fd, among others — and the entry's close-on-exec
// processing at the child's execve then closes the impostor, killing the
// exec. So the fork must hold the handler's locks (the pthread_atfork
// discipline the runtime applies to its own), and every kernel+table update
// pair must be atomic under them. Each churn thread works on its own
// descriptors, so the test is correct under POSIX; only the fork snapshot is
// under attack.

#include <fcntl.h>
#include <pthread.h>
#include <spawn.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

#define MARKER "child"
#define CHILD_CODE 19
#define WORKERS 4
#define SPAWNS 25

static atomic_int stop = 0;
static const char *self_path;

static void *churn(void *arg) {
    (void) arg;
    while (!atomic_load(&stop)) {
        int fd = open(self_path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) continue;
        int d = dup(fd);
        int e = fcntl(fd, F_DUPFD_CLOEXEC, 20);
        if (d >= 0 && e >= 0) dup2(d, e);
        if (e >= 0) close(e);
        if (d >= 0) close(d);
        close(fd);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], MARKER) == 0) return CHILD_CODE;
    self_path = argv[0];

    pthread_t th[WORKERS];
    for (int i = 0; i < WORKERS; i++)
        if (pthread_create(&th[i], 0, churn, 0) != 0) return 1;

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
