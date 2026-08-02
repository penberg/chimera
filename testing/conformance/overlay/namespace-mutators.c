// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: CHIMERA_FS=%t.fixture/delta %runner %t check %t.fixture
// RUN: %t verify %t.fixture
//
// Namespace mutation under the overlay: unlink hides a lower name
// (delete-then-readdir), recreating a deleted name starts fresh
// (delete-then-recreate — lower contents must not bleed through), rename of
// a file moves it in the merged view, and rename of a lower-only directory
// answers EXDEV, the signal that makes mv fall back to copy+delete — the
// fallback is exercised here the way mv does it. The check step asserts
// plain POSIX outcomes, true natively too; verify then branches on whether a
// delta materialized and asserts the host tree survived untouched.

#define _GNU_SOURCE

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int mkdirs(const char *path) {
    char buf[PATH_MAX];
    snprintf(buf, sizeof(buf), "%s", path);
    for (char *p = buf + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        if (mkdir(buf, 0755) != 0 && errno != EEXIST) return -1;
        *p = '/';
    }
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

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

static int lower_canon(const char *fixture, char *out, size_t size) {
    char lower[PATH_MAX];
    snprintf(lower, sizeof(lower), "%s/lower", fixture);
    return realpath(lower, out) ? (void)size, 0 : -1;
}

static int dir_has(const char *dir, const char *name) {
    DIR *d = opendir(dir);
    if (!d) return -1;
    struct dirent *e;
    int found = 0;
    while ((e = readdir(d)) != NULL)
        if (strcmp(e->d_name, name) == 0) found = 1;
    closedir(d);
    return found;
}

static int prep(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX];
    snprintf(lower, sizeof(lower), "%s/lower", fixture);
    if (mkdirs(lower) != 0) return 11;
    snprintf(path, sizeof(path), "%s/doomed", lower);
    if (write_file(path, "doomed lower bytes") != 0) return 12;
    snprintf(path, sizeof(path), "%s/mover", lower);
    if (write_file(path, "mover payload") != 0) return 13;
    snprintf(path, sizeof(path), "%s/dir", lower);
    if (mkdirs(path) != 0) return 14;
    snprintf(path, sizeof(path), "%s/dir/inner", lower);
    if (write_file(path, "inner bytes") != 0) return 15;
    return 0;
}

static int check(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX], dst[PATH_MAX], buf[128];
    if (lower_canon(fixture, lower, sizeof(lower)) != 0) return 30;

    // Delete, then readdir: the name is gone.
    snprintf(path, sizeof(path), "%s/doomed", lower);
    if (unlink(path) != 0) return 31;
    struct stat st;
    if (stat(path, &st) == 0 || errno != ENOENT) return 32;
    if (dir_has(lower, "doomed") != 0) return 33;

    // Recreate: fresh content, nothing bleeds through.
    if (write_file(path, "reborn") != 0) return 34;
    if (read_file(path, buf, sizeof(buf)) != 0) return 35;
    if (strcmp(buf, "reborn") != 0) return 36;

    // Rename a file: source gone, destination carries the payload.
    snprintf(path, sizeof(path), "%s/mover", lower);
    snprintf(dst, sizeof(dst), "%s/moved", lower);
    if (rename(path, dst) != 0) return 37;
    if (stat(path, &st) == 0 || errno != ENOENT) return 38;
    if (read_file(dst, buf, sizeof(buf)) != 0) return 39;
    if (strcmp(buf, "mover payload") != 0) return 40;

    // Rename a lower directory: EXDEV under the overlay, plain success
    // natively — so do what mv does, falling back to copy+delete on EXDEV.
    snprintf(path, sizeof(path), "%s/dir", lower);
    snprintf(dst, sizeof(dst), "%s/dir2", lower);
    if (rename(path, dst) != 0) {
        if (errno != EXDEV) return 41;
        char from[PATH_MAX], to[PATH_MAX];
        if (mkdir(dst, 0755) != 0) return 42;
        snprintf(from, sizeof(from), "%s/inner", path);
        snprintf(to, sizeof(to), "%s/inner", dst);
        if (read_file(from, buf, sizeof(buf)) != 0) return 43;
        if (write_file(to, buf) != 0) return 44;
        if (unlink(from) != 0) return 45;
        if (rmdir(path) != 0) return 46;
    }
    // Whichever route ran, the merged outcome is identical.
    if (stat(path, &st) == 0 || errno != ENOENT) return 47;
    snprintf(dst, sizeof(dst), "%s/dir2/inner", lower);
    if (read_file(dst, buf, sizeof(buf)) != 0) return 48;
    if (strcmp(buf, "inner bytes") != 0) return 49;
    return 0;
}

static int verify(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX], probe[PATH_MAX], buf[128];
    if (lower_canon(fixture, lower, sizeof(lower)) != 0) return 51;
    snprintf(probe, sizeof(probe), "%s/delta/data%s", fixture, lower);

    struct stat st;
    int overlay_ran = stat(probe, &st) == 0;
    if (!overlay_ran) {
        // Native check: the mutations really happened on the host.
        snprintf(path, sizeof(path), "%s/moved", lower);
        return stat(path, &st) == 0 ? 0 : 52;
    }

    // The host tree is exactly what prep built.
    snprintf(path, sizeof(path), "%s/doomed", lower);
    if (read_file(path, buf, sizeof(buf)) != 0) return 53;
    if (strcmp(buf, "doomed lower bytes") != 0) return 54;
    snprintf(path, sizeof(path), "%s/mover", lower);
    if (read_file(path, buf, sizeof(buf)) != 0) return 55;
    if (strcmp(buf, "mover payload") != 0) return 56;
    snprintf(path, sizeof(path), "%s/dir/inner", lower);
    if (read_file(path, buf, sizeof(buf)) != 0) return 57;
    if (strcmp(buf, "inner bytes") != 0) return 58;
    snprintf(path, sizeof(path), "%s/moved", lower);
    if (stat(path, &st) == 0 || errno != ENOENT) return 59;
    snprintf(path, sizeof(path), "%s/dir2", lower);
    if (stat(path, &st) == 0 || errno != ENOENT) return 60;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) return 10;
    if (strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (strcmp(argv[1], "check") == 0) return check(argv[2]);
    if (strcmp(argv[1], "verify") == 0) return verify(argv[2]);
    return 10;
}
