// RUN: %cc %s -o %t && %runner %t
//
// pthread_atfork handlers must run around fork in POSIX order: prepare
// handlers in reverse registration order before the fork, parent and child
// handlers in registration order after it, each in the process that owns
// them. On Darwin, Chimera keeps guest registrations off libpthread's global
// list (the runtime's own posix_spawn fork must not call guest pointers
// natively) and runs them translated around the fork trap — this checks that
// diversion preserves the ordering a native fork delivers.

#include <pthread.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static char order[8];
static int n;

static void add(char c) {
    if (n < 7) order[n++] = c;
}
static void prepare1(void) { add('a'); }
static void prepare2(void) { add('b'); }
static void parent1(void) { add('c'); }
static void parent2(void) { add('d'); }
static void child1(void) { add('e'); }
static void child2(void) { add('f'); }

int main(void) {
    if (pthread_atfork(prepare1, parent1, child1)) return 1;
    if (pthread_atfork(prepare2, parent2, child2)) return 2;

    pid_t pid = fork();
    if (pid < 0) return 3;
    if (pid == 0)
        _exit(strcmp(order, "baef") == 0 ? 0 : 1);

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 4;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return 5;
    return strcmp(order, "bacd") == 0 ? 0 : 6;
}
