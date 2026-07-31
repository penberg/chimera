// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: CHIMERA_WORKSPACE=%t.fixture/delta %runner %t check %t.fixture
// RUN: %t verify %t.fixture
//
// Attribute changes route through the overlay: chmod and utimensat of a
// lower file copy it up and apply in the delta, the guest observes the new
// bits and times, and the host file keeps its own. The verify step branches
// on whether a delta materialized, as the other overlay tests do.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int lower_target(const char *fixture, char *out, size_t size) {
    char lower[PATH_MAX], canon[PATH_MAX];
    snprintf(lower, sizeof(lower), "%s/lower", fixture);
    if (!realpath(lower, canon)) return -1;
    snprintf(out, size, "%s/f", canon);
    return 0;
}

static int prep(const char *fixture) {
    char path[PATH_MAX];
    if (mkdir(fixture, 0755) != 0 && errno != EEXIST) return 11;
    snprintf(path, sizeof(path), "%s/lower", fixture);
    if (mkdir(path, 0755) != 0 && errno != EEXIST) return 12;
    snprintf(path, sizeof(path), "%s/lower/f", fixture);
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0 || write(fd, "x", 1) != 1 || close(fd) != 0) return 13;
    if (chmod(path, 0644) != 0) return 14;
    return 0;
}

static int check(const char *fixture) {
    char path[PATH_MAX];
    struct stat st;
    if (lower_target(fixture, path, sizeof(path)) != 0) return 31;

    if (chmod(path, 0600) != 0) return 32;
    if (stat(path, &st) != 0 || (st.st_mode & 07777) != 0600) return 33;

    struct timespec ts[2] = {{111111, 0}, {222222, 0}};
    if (utimensat(AT_FDCWD, path, ts, 0) != 0) return 34;
    if (stat(path, &st) != 0 || st.st_mtim.tv_sec != 222222) return 35;
    return 0;
}

static int verify(const char *fixture) {
    char path[PATH_MAX], probe[PATH_MAX];
    struct stat st;
    if (lower_target(fixture, path, sizeof(path)) != 0) return 51;
    snprintf(probe, sizeof(probe), "%s/delta/data%s", fixture, path);

    if (stat(probe, &st) == 0) {
        // Overlay ran: the delta copy carries the change…
        if ((st.st_mode & 07777) != 0600) return 52;
        if (st.st_mtim.tv_sec != 222222) return 53;
        // …and the host file was never touched.
        if (stat(path, &st) != 0) return 54;
        if ((st.st_mode & 07777) != 0644) return 55;
        if (st.st_mtim.tv_sec == 222222) return 56;
    } else {
        // Native check run: the change landed directly.
        if (stat(path, &st) != 0) return 57;
        if ((st.st_mode & 07777) != 0600) return 58;
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
