// RUN: %cc %s -o %t && %runner %t
// UNSUPPORTED: darwin -- Linux-only host interfaces
//
// The kernel resolves /dev/fd/N and /proc/self/fd/N as magic links to the
// open file itself, not through the filesystem — their readlink target
// (`pipe:[…]`) is not a path — so a userspace path walker cannot follow
// them. They are what bash's process substitution and the /dev/std*
// redirections name, so opening one must reopen the descriptor's object.
// Exits 0 only if a pipe read end reopens through /dev/fd and
// /proc/self/fd and delivers the pipe's bytes, and /dev/stdout accepts a
// write.

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int reopen_reads(const char *dir, int fd, const char *want) {
    char path[64];
    snprintf(path, sizeof(path), "%s/%d", dir, fd);
    int reopened = open(path, O_RDONLY);
    if (reopened < 0) return 0;
    char buf[16] = {0};
    ssize_t n = read(reopened, buf, strlen(want));
    close(reopened);
    return n == (ssize_t)strlen(want) && strcmp(buf, want) == 0;
}

int main(void) {
    int fds[2];
    if (pipe(fds) != 0) return 1;
    if (write(fds[1], "ab", 2) != 2 || write(fds[1], "cd", 2) != 2) return 2;
    close(fds[1]);

    // Each reopen is an independent handle on the same pipe; two bytes
    // remain for the second because the first consumed two.
    if (!reopen_reads("/dev/fd", fds[0], "ab")) return 3;
    if (!reopen_reads("/proc/self/fd", fds[0], "cd")) return 4;
    close(fds[0]);

    int out = open("/dev/stdout", O_WRONLY);
    if (out < 0) return 5;
    if (write(out, "", 0) != 0) return 6;
    close(out);

    return 0;
}
