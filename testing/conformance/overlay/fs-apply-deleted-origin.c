// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: test -z "%runner" || CHIMERA_FS=%t.fixture/ws %runner %t mutate %t.fixture
// RUN: test -z "%runner" || %t drive %t.fixture "%runner"
//
// Apply must treat every host-side change as a conflict, not only an edit.
// A copied-up file records the identity of the lower it shadowed; if the
// host deleted that file since, recreating it would silently overrule the
// host's decision — deletion is as much a host change as modification. The
// mirror image: a guest-created file has no origin, but if a host entry
// appeared at that name since the guest created it, landing the addition
// would overwrite it. The drive step sours the host both ways after the
// guest run, then requires apply to fail, leave the deleted path absent,
// and leave the squatter intact. Skips itself when no delta materialized
// (native and --unsafe runs).

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    snprintf(path, sizeof(path), "%s/lower/gone", fixture);
    if (write_file(path, "host v1") != 0) return 13;
    return 0;
}

static int mutate(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX];
    if (lower_canon(fixture, lower) != 0) return 21;
    snprintf(path, sizeof(path), "%s/gone", lower);
    if (write_file(path, "guest version") != 0) return 22;
    snprintf(path, sizeof(path), "%s/added", lower);
    if (write_file(path, "guest fresh") != 0) return 23;
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

    // The host moves on underneath the filesystem: the copied-up file is
    // deleted, and a squatter appears where the guest created a new file.
    snprintf(path, sizeof(path), "%s/gone", lower);
    if (unlink(path) != 0) return 33;
    snprintf(path, sizeof(path), "%s/added", lower);
    if (write_file(path, "host squatter") != 0) return 34;

    // Both entries are conflicts; apply must fail.
    char cmd[PATH_MAX * 2];
    snprintf(cmd, sizeof(cmd), "%s fs apply %s >/dev/null 2>&1", chim, ws);
    if (system(cmd) == 0) return 35;

    // The deletion stands: the filesystem must not resurrect the file.
    snprintf(path, sizeof(path), "%s/gone", lower);
    if (stat(path, &st) == 0 || errno != ENOENT) return 36;

    // The squatter stands: the addition must not overwrite it.
    snprintf(path, sizeof(path), "%s/added", lower);
    if (read_file(path, buf, sizeof(buf)) != 0) return 37;
    if (strcmp(buf, "host squatter") != 0) return 38;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (argc == 3 && strcmp(argv[1], "mutate") == 0) return mutate(argv[2]);
    if (argc == 4 && strcmp(argv[1], "drive") == 0)
        return drive(argv[2], argv[3]);
    return 10;
}
