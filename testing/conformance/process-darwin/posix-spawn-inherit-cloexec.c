// RUN: %cc %s -o %t && %runner %t
//
// posix_spawn's inherit file action keeps a descriptor open in the child even
// when that descriptor carries FD_CLOEXEC: the kernel clears the flag as part
// of the action. libuv marks every stdio stream it adopts close-on-exec and
// names it inherited when it spawns, so a runtime that honours only the
// CLOEXEC_DEFAULT sweep and not the flag clear leaves every node child with
// its stdio closed. Spawn /bin/echo with stdout marked FD_CLOEXEC and named
// inherited; its output must arrive.

#include <fcntl.h>
#include <spawn.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

int main(void) {
    int p[2];
    if (pipe(p) != 0) return 1;
    // The write end is the child's stdout: dup it to 1, then mark 1
    // close-on-exec the way libuv leaves an adopted stream.
    int saved = dup(1);
    if (saved < 0 || dup2(p[1], 1) < 0) return 2;
    if (fcntl(1, F_SETFD, FD_CLOEXEC) != 0) return 3;
    close(p[1]);

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addinherit_np(&fa, 1);
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_CLOEXEC_DEFAULT);
    char *argv[] = {"/bin/echo", "inherited", NULL};
    pid_t pid;
    int rc = posix_spawn(&pid, "/bin/echo", &fa, &attr, argv, environ);
    dup2(saved, 1);
    if (rc != 0) return 4;

    char buf[32] = {0};
    ssize_t n = read(p[0], buf, sizeof buf - 1);
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 5;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return 6;
    return n > 0 && strcmp(buf, "inherited\n") == 0 ? 0 : 7;
}
