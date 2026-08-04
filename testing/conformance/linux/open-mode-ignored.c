// RUN: %cc %s -o %t && %runner %t
// UNSUPPORTED: darwin -- Linux-only host interfaces
//
// open(2)'s mode argument is meaningless unless the call creates a file
// (O_CREAT or O_TMPFILE): the parameter is variadic, so callers routinely
// pass a stale or habitual value on plain reads — Bun hands every open a
// 0666. The kernel ignores it; a sandbox that resolves opens through the
// stricter openat2 (which rejects a nonzero mode on a non-creating open
// with EINVAL) must ignore it too, or valid guest opens start failing.
// Exits 0 only if a read-only open with a junk mode succeeds and reads.

#define _GNU_SOURCE /* O_PATH */

#include <fcntl.h>
#include <unistd.h>

int main(void) {
    int fd = open("/etc/hostname", O_RDONLY, 0666);
    if (fd < 0) return 1;
    char c;
    if (read(fd, &c, 1) != 1) return 2;
    close(fd);

    fd = openat(AT_FDCWD, "/etc/hostname", O_RDONLY | O_NOCTTY, 0777);
    if (fd < 0) return 3;
    close(fd);

    // O_PATH gets the same leniency: open(2) silently masks companion flags
    // other than O_CLOEXEC/O_DIRECTORY/O_NOFOLLOW out of an O_PATH open
    // (openat2 would reject them with EINVAL), and callers rely on it —
    // Bun's existence probes open O_PATH|O_LARGEFILE|O_NONBLOCK.
    fd = open("/etc/hostname", O_PATH | O_NONBLOCK | 0100000 /* O_LARGEFILE bit */);
    if (fd < 0) return 4;
    close(fd);

    return 0;
}
