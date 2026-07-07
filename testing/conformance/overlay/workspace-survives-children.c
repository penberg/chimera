// RUN: %cc %s -o %t && rm -rf %t.probe && %runner %t %t.probe && rm -f %t.probe
//
// A guest child's exit must not end the session's workspace. Guest fork is a
// host fork, so a child's host process carries a copy of the CLI and returns
// through it when the child exits; the end-of-session disposition (the
// empty-delta removal, the kept notice, --rm) belongs to the session root
// alone. The regression this pins: an interactive bash forks short-lived rc
// children before anything writes, the first child's exit garbage-collected
// the still-empty workspace, and every later write in the session failed
// with ENOENT while reads kept working.

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc != 2) return 10;

    // A child that exits while the workspace's delta is still empty.
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) _exit(0);
    int status;
    if (waitpid(pid, &status, 0) != pid) return 2;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return 3;

    // The session must still be able to write.
    int fd = open(argv[1], O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) return 4;
    if (write(fd, "alive", 5) != 5) return 5;
    if (close(fd) != 0) return 6;

    char buf[8] = {0};
    fd = open(argv[1], O_RDONLY);
    if (fd < 0) return 7;
    if (read(fd, buf, sizeof(buf) - 1) != 5 || strcmp(buf, "alive") != 0)
        return 8;
    close(fd);
    return 0;
}
