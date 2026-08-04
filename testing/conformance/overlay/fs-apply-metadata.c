// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: test -z "%runner" || CHIMERA_FS=%t.fixture/ws %runner %t mutate %t.fixture
// RUN: test -z "%runner" || %t drive %t.fixture "%runner"
// UNSUPPORTED: darwin -- the workspace overlay is a Linux-only feature (xattrs, sysmacros.h, st_atim)
//
// Apply must reproduce the guest-visible metadata, not just the bytes. A
// filesystem holding only a chmod, a utimensat, or a user xattr change prints
// a successful M line, so the host must actually end up with the set-id
// bits, the timestamps, and the xattr — and a combined content-plus-metadata
// change must land all of it together. Chimera's own user.chimera.*
// bookkeeping must never reach the host. The drive step skips itself when no
// delta materialized (native and --unsafe runs, where the guest mutated the
// host directly).

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <unistd.h>

static int write_file(const char *path, const char *content) {
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) return -1;
    ssize_t n = (ssize_t)strlen(content);
    if (write(fd, content, n) != n) return -1;
    return close(fd);
}

static int read_file(const char *path, char *buf, size_t size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    ssize_t n = read(fd, buf, size - 1);
    close(fd);
    if (n < 0) return -1;
    buf[n] = 0;
    return 0;
}

static int lower_canon(const char *fixture, char *out) {
    char lower[PATH_MAX];
    snprintf(lower, sizeof(lower), "%s/lower", fixture);
    return realpath(lower, out) ? 0 : -1;
}

static int prep(const char *fixture) {
    char path[PATH_MAX];
    if (mkdir(fixture, 0755) != 0 && errno != EEXIST) return 11;
    snprintf(path, sizeof(path), "%s/lower", fixture);
    if (mkdir(path, 0755) != 0 && errno != EEXIST) return 12;
    const char *names[] = {"m", "t", "x", "c"};
    for (int i = 0; i < 4; i++) {
        snprintf(path, sizeof(path), "%s/lower/%s", fixture, names[i]);
        if (write_file(path, "host v1") != 0) return 13 + i;
    }
    return 0;
}

static int mutate(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX];
    if (lower_canon(fixture, lower) != 0) return 21;

    // Mode only, including a set-id bit.
    snprintf(path, sizeof(path), "%s/m", lower);
    if (chmod(path, 04751) != 0) return 22;

    // Timestamps only.
    struct timespec times[2] = {{11111, 0}, {22222, 0}};
    snprintf(path, sizeof(path), "%s/t", lower);
    if (utimensat(AT_FDCWD, path, times, 0) != 0) return 23;

    // A user xattr only.
    snprintf(path, sizeof(path), "%s/x", lower);
    if (setxattr(path, "user.test", "hello", 5, 0) != 0) return 24;

    // Content and metadata together.
    snprintf(path, sizeof(path), "%s/c", lower);
    if (write_file(path, "guest c") != 0) return 25;
    if (chmod(path, 0640) != 0) return 26;
    if (setxattr(path, "user.combo", "yes", 3, 0) != 0) return 27;
    struct timespec ctimes[2] = {{33333, 0}, {44444, 0}};
    if (utimensat(AT_FDCWD, path, ctimes, 0) != 0) return 28;
    return 0;
}

static int drive(const char *fixture, const char *runner) {
    char ws[PATH_MAX], probe[PATH_MAX], lower[PATH_MAX], path[PATH_MAX];
    char buf[256];
    snprintf(ws, sizeof(ws), "%s/ws", fixture);
    snprintf(probe, sizeof(probe), "%s/data", ws);
    struct stat st;
    if (stat(probe, &st) != 0) return 0; // native or --unsafe: no delta
    if (lower_canon(fixture, lower) != 0) return 31;

    char chim[PATH_MAX];
    if (sscanf(runner, "%s", chim) != 1) return 32;

    char cmd[PATH_MAX * 2];
    snprintf(cmd, sizeof(cmd), "%s fs apply %s >/dev/null 2>&1", chim, ws);
    if (system(cmd) != 0) return 33;

    // Mode change landed, set-id bit included.
    snprintf(path, sizeof(path), "%s/m", lower);
    if (stat(path, &st) != 0) return 34;
    if ((st.st_mode & 07777) != 04751) return 35;

    // Timestamps landed (stat before any content read touches atime).
    snprintf(path, sizeof(path), "%s/t", lower);
    if (stat(path, &st) != 0) return 36;
    if (st.st_atim.tv_sec != 11111 || st.st_mtim.tv_sec != 22222) return 37;

    // The xattr landed.
    snprintf(path, sizeof(path), "%s/x", lower);
    if (getxattr(path, "user.test", buf, sizeof(buf)) != 5 ||
        memcmp(buf, "hello", 5) != 0)
        return 38;

    // The combined change landed whole: bytes, mode, xattr, mtime — and no
    // chimera bookkeeping came along.
    snprintf(path, sizeof(path), "%s/c", lower);
    if (stat(path, &st) != 0) return 39;
    if ((st.st_mode & 07777) != 0640) return 40;
    if (st.st_mtim.tv_sec != 44444) return 41;
    if (getxattr(path, "user.combo", buf, sizeof(buf)) != 3 ||
        memcmp(buf, "yes", 3) != 0)
        return 42;
    if (getxattr(path, "user.chimera.origin", buf, sizeof(buf)) >= 0 ||
        errno != ENODATA)
        return 43;
    if (read_file(path, buf, sizeof(buf)) != 0 || strcmp(buf, "guest c") != 0)
        return 44;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (argc == 3 && strcmp(argv[1], "mutate") == 0) return mutate(argv[2]);
    if (argc == 4 && strcmp(argv[1], "drive") == 0)
        return drive(argv[2], argv[3]);
    return 10;
}
