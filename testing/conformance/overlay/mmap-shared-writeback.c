// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: CHIMERA_COW=%t.fixture/delta %runner %t check %t.fixture
// RUN: %t verify %t.fixture
//
// Writable MAP_SHARED under the overlay: the mapping requires a descriptor
// opened for writing, eager copy-up means such a descriptor already points
// at an upper file, so stores through the mapping land in the delta and read
// back through the file — while the host file keeps its original bytes. The
// verify step branches like write-preserves-host.c: delta present → host
// pristine; no delta (native check) → mutation landed directly.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
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

static int lower_target(const char *fixture, char *out, size_t size) {
    char lower[PATH_MAX], canon[PATH_MAX];
    snprintf(lower, sizeof(lower), "%s/lower", fixture);
    if (!realpath(lower, canon)) return -1;
    snprintf(out, size, "%s/mapped", canon);
    return 0;
}

static int prep(const char *fixture) {
    char path[PATH_MAX];
    if (mkdir(fixture, 0755) != 0 && errno != EEXIST) return 11;
    snprintf(path, sizeof(path), "%s/lower", fixture);
    if (mkdir(path, 0755) != 0 && errno != EEXIST) return 11;
    snprintf(path, sizeof(path), "%s/lower/mapped", fixture);
    if (write_file(path, "0123456789abcdef") != 0) return 12;
    return 0;
}

static int check(const char *fixture) {
    char path[PATH_MAX], buf[64];
    if (lower_target(fixture, path, sizeof(path)) != 0) return 31;

    int fd = open(path, O_RDWR);
    if (fd < 0) return 32;
    char *map = mmap(NULL, 16, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) return 33;
    memcpy(map, "MAPPED-WRITE!!!!", 16);
    if (msync(map, 16, MS_SYNC) != 0) return 34;
    if (munmap(map, 16) != 0) return 35;
    close(fd);

    // The store is visible through ordinary reads of the file.
    if (read_file(path, buf, sizeof(buf)) != 0) return 36;
    if (strcmp(buf, "MAPPED-WRITE!!!!") != 0) return 37;
    return 0;
}

static int verify(const char *fixture) {
    char path[PATH_MAX], probe[PATH_MAX], buf[64];
    if (lower_target(fixture, path, sizeof(path)) != 0) return 51;
    snprintf(probe, sizeof(probe), "%s/delta/data%s", fixture, path);

    struct stat st;
    int overlay_ran = stat(probe, &st) == 0;
    if (read_file(path, buf, sizeof(buf)) != 0) return 52;
    if (overlay_ran) {
        if (strcmp(buf, "0123456789abcdef") != 0) return 53;
        if (read_file(probe, buf, sizeof(buf)) != 0) return 54;
        if (strcmp(buf, "MAPPED-WRITE!!!!") != 0) return 55;
    } else {
        if (strcmp(buf, "MAPPED-WRITE!!!!") != 0) return 56;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) return 10;
    if (strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (strcmp(argv[1], "check") == 0) return check(argv[2]);
    if (strcmp(argv[1], "verify") == 0) return verify(argv[2]);
    return 10;
}
