// RUN: %cc %s -o %t && timeout 10 %runner %t
//
// A forked child's system calls must still reach the embedder's handler. The
// parent is intercepted by construction — it is the process Chimera started —
// but a child inherits its interception from whatever mechanism the backend
// uses, and a backend can lose it there silently: an escaped syscall does not
// fail, it succeeds against the host. Under syscall user dispatch that is
// exactly what happens if the child does not re-arm, because the kernel
// clears the dispatch configuration across fork; the child then runs entirely
// outside the sandbox while every call it makes appears to work.
//
// `/proc/self/exe` is the probe: the embedder virtualizes it to the guest
// program, so an intercepted child reads back its own path and an escaped one
// reads back the runtime's binary. Native execution reads back the same path
// the intercepted child does, so the test is a real assertion in every mode.

#define _GNU_SOURCE

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int read_exe(char *buf, size_t len) {
    ssize_t n = readlink("/proc/self/exe", buf, len - 1);
    if (n < 0) return -1;
    buf[n] = '\0';
    return 0;
}

int main(void) {
    char parent[PATH_MAX];
    if (read_exe(parent, sizeof parent) != 0) {
        perror("parent readlink");
        return 1;
    }

    int fds[2];
    if (pipe(fds) != 0) {
        perror("pipe");
        return 1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }
    if (pid == 0) {
        close(fds[0]);
        char child[PATH_MAX];
        if (read_exe(child, sizeof child) != 0) _exit(2);
        size_t len = strlen(child);
        _exit(write(fds[1], child, len) == (ssize_t) len ? 0 : 3);
    }

    close(fds[1]);
    char child[PATH_MAX] = {0};
    ssize_t n = read(fds[0], child, sizeof child - 1);
    if (n < 0) {
        perror("read");
        return 1;
    }
    child[n] = '\0';

    int status;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        return 1;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "child failed: status %#x\n", status);
        return 1;
    }
    if (strcmp(parent, child) != 0) {
        fprintf(stderr, "child escaped interception: parent %s, child %s\n", parent, child);
        return 1;
    }
    return 0;
}
