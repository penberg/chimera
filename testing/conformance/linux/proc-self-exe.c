// RUN: %cc %s -o %t && %runner %t
//
// /proc/self/exe names the process's executed image. Under an emulated
// execve the kernel's own link points at the runtime binary, so a sandbox
// must virtualize it — a self-respawning program (Bun's execPath) would
// otherwise exec the runtime instead of itself. Exits 0 only if readlink
// reports this binary, the link opens and reads as an ELF, and stat agrees
// with the binary's identity.

#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char self[PATH_MAX], me[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n < 0) return 1;
    self[n] = 0;
    if (!realpath(argv[0], me)) return 2;
    if (strcmp(self, me) != 0) {
        fprintf(stderr, "exe=%s me=%s\n", self, me);
        return 3;
    }

    int fd = open("/proc/self/exe", O_RDONLY);
    if (fd < 0) return 4;
    char magic[4];
    if (read(fd, magic, 4) != 4 || memcmp(magic, "\177ELF", 4) != 0) return 5;
    close(fd);

    struct stat a, b;
    if (stat("/proc/self/exe", &a) != 0 || stat(me, &b) != 0) return 6;
    if (a.st_ino != b.st_ino || a.st_dev != b.st_dev) return 7;

    return 0;
}
