// execveat names its target by file descriptor. With AT_EMPTY_PATH and an empty
// pathname (the fexecve case) it execs the open fd itself; the runtime resolves
// that through /proc/self/fd and translates the new image like any other target.
//
// RUN: %cc %s -o %t && %runner %t; test $? = 55
// UNSUPPORTED: darwin -- execveat is a Linux syscall; Darwin names an exec target by path only.

#define _GNU_SOURCE
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

extern char **environ;

int main(int argc, char **argv) {
    if (argc == 1) {
        int fd = open(argv[0], O_RDONLY | O_CLOEXEC);
        if (fd < 0)
            return 10;
        char *newargv[] = {argv[0], "viafd", 0};
        execveat(fd, "", newargv, environ, AT_EMPTY_PATH);
        return 11; // execveat does not return on success
    }
    if (argc == 2 && strcmp(argv[1], "viafd") == 0)
        return 55;
    return 12;
}
