// RUN: %cc %s -o %t && %runner %t
//
// posix_spawn reports an exec failure synchronously as its return value, not
// only through the child's exit status. glibc issues clone(CLONE_VM|CLONE_VFORK)
// and stays blocked until the child reaches execve or reports an error; Chimera
// emulates that spawn with a fork, so it must reproduce the same synchronous
// failure — a missing program must make posix_spawn return ENOENT, and a real
// one must still spawn and run.

#include <errno.h>
#include <spawn.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

int main(void) {
    // A program that does not exist: posix_spawn must fail with ENOENT rather
    // than report success and leave the caller to discover a 127 exit later.
    pid_t pid = -1;
    char *missing[] = {(char *) "/nonexistent/definitely-not-here", NULL};
    int rc = posix_spawn(&pid, missing[0], NULL, NULL, missing, environ);
    if (rc != ENOENT) {
        fprintf(stderr, "missing binary: posix_spawn returned %d (%s), want ENOENT\n",
                rc, strerror(rc));
        return 1;
    }

    // A real program must spawn, run, and report its status normally.
    pid = -1;
#if defined(__APPLE__)
    // macOS has no /bin/true; /usr/bin/true is a real (arm64e, shared-cache
    // linked) system binary, spawned here as a guest.
    char *ok[] = {(char *) "/usr/bin/true", NULL};
#else
    char *ok[] = {(char *) "/bin/true", NULL};
#endif
    rc = posix_spawn(&pid, ok[0], NULL, NULL, ok, environ);
    if (rc != 0) {
        fprintf(stderr, "/bin/true: posix_spawn returned %d (%s)\n", rc, strerror(rc));
        return 2;
    }
    int status;
    if (waitpid(pid, &status, 0) != pid) {
        perror("waitpid");
        return 3;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "/bin/true: bad status 0x%x\n", status);
        return 4;
    }

    return 0;
}
