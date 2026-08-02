// RUN: %cc %s -o %t && rm -rf %t.ws %t.probe
// RUN: CHIMERA_FS=%t.ws %runner %t write %t.probe
// RUN: CHIMERA_FS=%t.ws %runner %t read %t.probe
// RUN: %t verify %t.ws %t.probe
//
// Attach: two sessions of one filesystem. Run 1 writes a file; run 2, a
// separate chimera invocation attached to the same filesystem, reads it back
// — the delta format is self-describing on disk, so reopening is the whole
// mechanism. Verify then proves the file lives in the filesystem and never
// touched the host (natively the env var is inert and the file is simply on
// disk, which verify accepts).

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "write") == 0) {
        int fd = open(argv[2], O_CREAT | O_EXCL | O_WRONLY, 0644);
        if (fd < 0) return 21;
        if (write(fd, "session-one", 11) != 11) return 22;
        return close(fd) == 0 ? 0 : 23;
    }
    if (argc == 3 && strcmp(argv[1], "read") == 0) {
        char buf[32] = {0};
        int fd = open(argv[2], O_RDONLY);
        if (fd < 0) return 31;
        if (read(fd, buf, sizeof(buf) - 1) != 11) return 32;
        return strcmp(buf, "session-one") == 0 ? 0 : 33;
    }
    if (argc == 4 && strcmp(argv[1], "verify") == 0) {
        char probe[4096];
        struct stat st;
        snprintf(probe, sizeof(probe), "%s/data%s", argv[2], argv[3]);
        if (stat(probe, &st) == 0) {
            // Overlay ran: the file is filesystem-only.
            return stat(argv[3], &st) == 0 || errno != ENOENT ? 41 : 0;
        }
        // Native runs: the writes landed directly on disk.
        return stat(argv[3], &st) == 0 ? 0 : 42;
    }
    return 10;
}
