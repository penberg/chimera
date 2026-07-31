// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: test -z "%runner" || CHIMERA_WORKSPACE=%t.fixture/ws %runner %t mutate %t.fixture
// RUN: test -z "%runner" || %t drive %t.fixture "%runner"
//
// Cross-type replacement: removing the existing host entry must be selected
// from what the host actually has, never from the incoming upper type. A
// guest that turned a file into a directory, a directory into a symlink,
// and a directory into a FIFO leaves an opaque directory, a symlink, and a
// FIFO in the workspace; apply used to pick remove_dir_all or remove_file
// from those incoming types and then fail with ENOTDIR replacing the file
// and EISDIR replacing the directories. Apply must land all three: the
// directory over the file (with its child), the symlink over the directory
// (as a symlink, not through it), and the FIFO over the directory. The
// drive step skips itself when no delta materialized (native and --unsafe
// runs, where the guest already reshaped the host directly).

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
    snprintf(path, sizeof(path), "%s/lower/f2d", fixture);
    if (write_file(path, "flat") != 0) return 13;
    snprintf(path, sizeof(path), "%s/lower/d2l", fixture);
    if (mkdir(path, 0755) != 0) return 14;
    snprintf(path, sizeof(path), "%s/lower/d2l/child", fixture);
    if (write_file(path, "x") != 0) return 15;
    snprintf(path, sizeof(path), "%s/lower/d2f", fixture);
    if (mkdir(path, 0755) != 0) return 16;
    snprintf(path, sizeof(path), "%s/lower/d2f/child", fixture);
    if (write_file(path, "y") != 0) return 17;
    return 0;
}

static int mutate(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX], sub[PATH_MAX];
    if (lower_canon(fixture, lower) != 0) return 21;

    // File becomes a directory.
    snprintf(path, sizeof(path), "%s/f2d", lower);
    if (unlink(path) != 0) return 22;
    if (mkdir(path, 0755) != 0) return 23;
    snprintf(sub, sizeof(sub), "%s/f2d/inner", lower);
    if (write_file(sub, "deep") != 0) return 24;

    // Directory becomes a symlink.
    snprintf(sub, sizeof(sub), "%s/d2l/child", lower);
    if (unlink(sub) != 0) return 25;
    snprintf(path, sizeof(path), "%s/d2l", lower);
    if (rmdir(path) != 0) return 26;
    if (symlink("somewhere", path) != 0) return 27;

    // Directory becomes a FIFO.
    snprintf(sub, sizeof(sub), "%s/d2f/child", lower);
    if (unlink(sub) != 0) return 28;
    snprintf(path, sizeof(path), "%s/d2f", lower);
    if (rmdir(path) != 0) return 29;
    if (mkfifo(path, 0644) != 0) return 30;
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

    // The directory landed over the file, child and all.
    snprintf(path, sizeof(path), "%s/f2d", lower);
    if (lstat(path, &st) != 0 || !S_ISDIR(st.st_mode)) return 34;
    snprintf(path, sizeof(path), "%s/f2d/inner", lower);
    if (read_file(path, buf, sizeof(buf)) != 0 || strcmp(buf, "deep") != 0)
        return 35;

    // The symlink landed over the directory — as a symlink.
    snprintf(path, sizeof(path), "%s/d2l", lower);
    if (lstat(path, &st) != 0 || !S_ISLNK(st.st_mode)) return 36;
    ssize_t n = readlink(path, buf, sizeof(buf) - 1);
    if (n < 0) return 37;
    buf[n] = 0;
    if (strcmp(buf, "somewhere") != 0) return 38;

    // The FIFO landed over the directory.
    snprintf(path, sizeof(path), "%s/d2f", lower);
    if (lstat(path, &st) != 0 || !S_ISFIFO(st.st_mode)) return 39;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (argc == 3 && strcmp(argv[1], "mutate") == 0) return mutate(argv[2]);
    if (argc == 4 && strcmp(argv[1], "drive") == 0)
        return drive(argv[2], argv[3]);
    return 10;
}
