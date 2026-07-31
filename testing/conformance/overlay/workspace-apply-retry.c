// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: test -z "%runner" || CHIMERA_WORKSPACE=%t.fixture/ws %runner %t mutate %t.fixture
// RUN: test -z "%runner" || %t drive %t.fixture "%runner"
//
// A partially successful apply must be safe to run again. The first apply
// lands the applicable entries and reports the genuine conflict; after the
// user resolves only that conflict (here by dropping the entry from the
// workspace), the rerun must succeed — the entries that already applied must
// count as applied, not resurface as new conflicts against their own
// post-apply host state. And the applied state must keep protecting the
// host: an entry edited or recreated on the host after its successful apply
// is a conflict on the next run, never silently overwritten or re-deleted.
// The drive step skips itself when no delta materialized (native and
// --unsafe runs).

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

static int append_file(const char *path, const char *content) {
    int fd = open(path, O_WRONLY | O_APPEND);
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
    snprintf(path, sizeof(path), "%s/lower/keep", fixture);
    if (write_file(path, "host v1") != 0) return 13;
    snprintf(path, sizeof(path), "%s/lower/edit", fixture);
    if (write_file(path, "host v1") != 0) return 14;
    snprintf(path, sizeof(path), "%s/lower/gone", fixture);
    if (write_file(path, "bye") != 0) return 15;
    return 0;
}

static int mutate(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX];
    if (lower_canon(fixture, lower) != 0) return 21;
    snprintf(path, sizeof(path), "%s/keep", lower);
    if (write_file(path, "guest keep") != 0) return 22;
    snprintf(path, sizeof(path), "%s/edit", lower);
    if (write_file(path, "guest edit") != 0) return 23;
    snprintf(path, sizeof(path), "%s/gone", lower);
    if (unlink(path) != 0) return 24;
    return 0;
}

static int apply(const char *chim, const char *ws) {
    char cmd[PATH_MAX * 2];
    snprintf(cmd, sizeof(cmd), "%s fs apply %s >/dev/null 2>&1", chim, ws);
    return system(cmd);
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

    // Sour one file so the first apply has a genuine conflict.
    snprintf(path, sizeof(path), "%s/edit", lower);
    if (append_file(path, " + host edit") != 0) return 33;

    // Apply #1: fails on the conflict, applies everything else.
    if (apply(chim, ws) == 0) return 34;
    snprintf(path, sizeof(path), "%s/keep", lower);
    if (read_file(path, buf, sizeof(buf)) != 0 ||
        strcmp(buf, "guest keep") != 0)
        return 35;
    snprintf(path, sizeof(path), "%s/gone", lower);
    if (stat(path, &st) == 0 || errno != ENOENT) return 36;

    // Resolve only the original conflict: drop it from the workspace.
    snprintf(path, sizeof(path), "%s/data%s/edit", ws, lower);
    if (unlink(path) != 0) return 37;

    // Apply #2: everything left already applied — idempotent success, and
    // the applied state stays applied.
    if (apply(chim, ws) != 0) return 38;
    snprintf(path, sizeof(path), "%s/keep", lower);
    if (read_file(path, buf, sizeof(buf)) != 0 ||
        strcmp(buf, "guest keep") != 0)
        return 39;
    snprintf(path, sizeof(path), "%s/gone", lower);
    if (stat(path, &st) == 0 || errno != ENOENT) return 40;

    // The host moves on after the successful apply: edits the applied file,
    // recreates the applied deletion.
    snprintf(path, sizeof(path), "%s/keep", lower);
    if (append_file(path, " + later host edit") != 0) return 41;
    snprintf(path, sizeof(path), "%s/gone", lower);
    if (write_file(path, "host reborn") != 0) return 42;

    // Apply #3: both are conflicts now; neither may be clobbered.
    if (apply(chim, ws) == 0) return 43;
    snprintf(path, sizeof(path), "%s/keep", lower);
    if (read_file(path, buf, sizeof(buf)) != 0 ||
        strcmp(buf, "guest keep + later host edit") != 0)
        return 44;
    snprintf(path, sizeof(path), "%s/gone", lower);
    if (read_file(path, buf, sizeof(buf)) != 0 ||
        strcmp(buf, "host reborn") != 0)
        return 45;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (argc == 3 && strcmp(argv[1], "mutate") == 0) return mutate(argv[2]);
    if (argc == 4 && strcmp(argv[1], "drive") == 0)
        return drive(argv[2], argv[3]);
    return 10;
}
