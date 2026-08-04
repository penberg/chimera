// RUN: %cc %s -o %t && %runner %t
//
// posix_spawnp resolves the program inside the spawned child: glibc's spawn
// child walks $PATH issuing one execve per candidate, so early ENOENT
// failures are routinely followed by an execve that succeeds. Chimera's spawn
// emulation reports the exec outcome to the blocked parent through a pipe,
// and that report must wait for the child to exec or exit — reporting the
// first failed candidate makes posix_spawnp return ENOENT for any program
// that is not in the first $PATH entry, while the child keeps walking, execs,
// and runs the program as an orphan whose side effects the caller never
// asked to keep.

#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

#define MARKER "child"
#define CHILD_CODE 7

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], MARKER) == 0) return CHILD_CODE;

    // The program's own directory last, behind entries that cannot hold it.
    char *slash = strrchr(argv[0], '/');
    if (slash == NULL) return 1;
    *slash = '\0';
    char path[4096];
    snprintf(path, sizeof(path), "/nonexistent-one:/nonexistent-two:%s", argv[0]);
    if (setenv("PATH", path, 1) != 0) return 2;
    char *base = slash + 1;
    *slash = '/';

    char *child_argv[] = {base, (char *) MARKER, NULL};
    pid_t pid;
    if (posix_spawnp(&pid, base, NULL, NULL, child_argv, environ) != 0) return 3;
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 4;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != CHILD_CODE) return 5;
    return 0;
}
