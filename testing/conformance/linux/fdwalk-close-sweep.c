// RUN: %cc %s -o %t && %runner %t
//
// Spawn hygiene the fdwalk way: bwrap lists /proc/self/fd and closes every
// descriptor above stderr it did not open before exec'ing its command. Under
// a sandbox, that listing is the host's — it names descriptors the runtime
// owns as well as the guest's — and the sweep must not be able to close the
// runtime out from under the process: the filesystem must keep answering
// afterwards. Exits 0 only if a file still opens and reads after the sweep.

#define _GNU_SOURCE

#include <dirent.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    DIR *d = opendir("/proc/self/fd");
    if (!d) return 1;

    // Collect before closing: the walk holds a descriptor of its own, and
    // closing entries out from under it would corrupt the walk.
    int fds[4096];
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL && n < 4096) {
        int fd = atoi(e->d_name);
        if (fd > 2)
            fds[n++] = fd;
    }
    closedir(d);
    for (int i = 0; i < n; i++)
        close(fds[i]);

    int fd = open("/etc/hostname", O_RDONLY);
    if (fd < 0) return 2;
    char c;
    if (read(fd, &c, 1) != 1) return 3;
    return 0;
}
