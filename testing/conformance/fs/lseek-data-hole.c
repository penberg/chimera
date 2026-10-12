// RUN: %cc %s -o %t && %runner %t %t.data
//
// SEEK_DATA and SEEK_HOLE on a file opened through the overlay. The virtual
// descriptor models the whence values the fd table can answer itself (SET,
// CUR, END); the extent queries are the kernel's to answer, through the
// backing open file, and the offset it names must become the descriptor's
// position. A runtime that refuses them with EINVAL breaks every sparse-aware
// copy (`cp` on both hosts probes a file's extents this way).

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc != 2) return 10;
    int fd = open(argv[1], O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;
    char buf[4096];
    memset(buf, 'x', sizeof buf);
    if (write(fd, buf, sizeof buf) != (ssize_t) sizeof buf) return 2;
    // Written data is counted as data however the filesystem allocates it.
    if (fsync(fd) != 0) return 3;

    // The data starts at 0, and a dense file's first hole is the one at EOF.
    if (lseek(fd, 0, SEEK_DATA) != 0) return 4;
    off_t hole = lseek(fd, 0, SEEK_HOLE);
    if (hole != (off_t) sizeof buf) return 5;
    // The answer is the descriptor's new position: a plain write lands there.
    if (lseek(fd, 0, SEEK_CUR) != hole) return 6;
    if (write(fd, "tail", 4) != 4) return 7;
    char back[4];
    if (pread(fd, back, 4, hole) != 4 || memcmp(back, "tail", 4) != 0) return 8;

    // Past the end there is no data, and the position is left alone.
    errno = 0;
    if (lseek(fd, hole + 4, SEEK_DATA) != -1 || errno != ENXIO) return 9;
    if (lseek(fd, 0, SEEK_CUR) != hole + 4) return 11;
    return close(fd) == 0 ? 0 : 12;
}
