// RUN: %cc %s -pthread -o %t && %runner %t
//
// fork must not hang while sibling threads start and exit around it. A lock
// that a sibling holds at the instant of the fork stays locked forever in the
// child, which has no copy of that sibling to release it; a runtime that
// takes such a lock as a thread starts or exits, and again in a fork child,
// hangs the child (caught here by the suite's per-test timeout).

#include <pthread.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

static volatile int done;

static void *nothing(void *arg) {
    return arg;
}

static void *churn(void *arg) {
    while (!done) {
        pthread_t t;
        if (pthread_create(&t, NULL, nothing, NULL) != 0) abort();
        pthread_join(t, NULL);
    }
    return arg;
}

int main(void) {
    pthread_t churners[4];
    for (int i = 0; i < 4; i++)
        if (pthread_create(&churners[i], NULL, churn, NULL) != 0) return 1;
    for (int i = 0; i < 500; i++) {
        pid_t pid = fork();
        if (pid < 0) return 1;
        if (pid == 0) _exit(0);
        int status;
        if (waitpid(pid, &status, 0) != pid) return 1;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return 1;
    }
    done = 1;
    for (int i = 0; i < 4; i++) pthread_join(churners[i], NULL);
    return 0;
}
