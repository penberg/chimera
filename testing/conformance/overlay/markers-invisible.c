// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: test -f %t.fixture/skip || CHIMERA_FS=%t.fixture/delta %runner %t check %t.fixture
// UNSUPPORTED: darwin -- the workspace overlay is a Linux-only feature (xattrs, sysmacros.h, st_atim)
//
// Overlay resolution: a whiteout deletes a lower name from the merged view
// (and never appears itself), an upper file shadows its lower counterpart,
// and an upper-only file joins the merge. The first RUN always executes
// natively and builds the fixture: a lower tree plus a delta whose data/
// mirrors the lower's canonical host path with a whiteout for `f`, an
// override for `g`, and a fresh `h` (writing `skip` instead when the
// filesystem has no user xattrs). The check step discovers its world at
// runtime — under the overlay `f` is deleted; natively (empty %runner, the
// env var inert) the same fixture shows the untouched lower.

#define _GNU_SOURCE

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/xattr.h>
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

static int prep(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX], data[PATH_MAX];

    snprintf(lower, sizeof(lower), "%s/lower", fixture);
    if (mkdirs(lower) != 0) return 11;
    snprintf(path, sizeof(path), "%s/f", lower);
    if (write_file(path, "lower-f") != 0) return 12;
    snprintf(path, sizeof(path), "%s/g", lower);
    if (write_file(path, "lower-g") != 0) return 13;

    // The delta's data/ tree mirrors canonical host paths — the overlay
    // resolves through the walker, which sees symlink-free absolute paths.
    char canon[PATH_MAX];
    if (!realpath(lower, canon)) return 14;
    snprintf(data, sizeof(data), "%s/delta/data%s", fixture, canon);
    if (mkdirs(data) != 0) return 15;

    // Record where the guest should look, so check needs no path games.
    snprintf(path, sizeof(path), "%s/abs", fixture);
    if (write_file(path, canon) != 0) return 16;

    // The whiteout for f: empty file + user.chimera.whiteout.
    snprintf(path, sizeof(path), "%s/f", data);
    if (write_file(path, "") != 0) return 17;
    if (setxattr(path, "user.chimera.whiteout", "1", 1, 0) != 0) {
        if (errno != ENOTSUP) return 18;
        // No user xattrs here: the overlay cannot run; tell RUN line 2.
        snprintf(path, sizeof(path), "%s/skip", fixture);
        return write_file(path, "") != 0 ? 19 : 0;
    }
    snprintf(path, sizeof(path), "%s/g", data);
    if (write_file(path, "upper-g") != 0) return 20;
    snprintf(path, sizeof(path), "%s/h", data);
    if (write_file(path, "upper-h") != 0) return 21;
    return 0;
}

static int check(const char *fixture) {
    char abs_path[PATH_MAX], lower[PATH_MAX], path[PATH_MAX], buf[64];

    snprintf(abs_path, sizeof(abs_path), "%s/abs", fixture);
    if (read_file(abs_path, lower, sizeof(lower)) != 0) return 31;

    struct stat st;
    snprintf(path, sizeof(path), "%s/f", lower);
    int overlay = stat(path, &st) != 0;
    if (overlay && errno != ENOENT) return 32;

    if (!overlay) {
        // Native world: the fixture's lower tree is untouched.
        if (read_file(path, buf, sizeof(buf)) != 0) return 33;
        if (strcmp(buf, "lower-f") != 0) return 34;
        snprintf(path, sizeof(path), "%s/g", lower);
        if (read_file(path, buf, sizeof(buf)) != 0) return 35;
        if (strcmp(buf, "lower-g") != 0) return 36;
        snprintf(path, sizeof(path), "%s/h", lower);
        if (stat(path, &st) == 0 || errno != ENOENT) return 37;
        return 0;
    }

    // Overlay world: f is whiteouted, g overridden, h added.
    snprintf(path, sizeof(path), "%s/f", lower);
    if (open(path, O_RDONLY) >= 0 || errno != ENOENT) return 38;
    snprintf(path, sizeof(path), "%s/g", lower);
    if (read_file(path, buf, sizeof(buf)) != 0) return 39;
    if (strcmp(buf, "upper-g") != 0) return 40;
    snprintf(path, sizeof(path), "%s/h", lower);
    if (read_file(path, buf, sizeof(buf)) != 0) return 41;
    if (strcmp(buf, "upper-h") != 0) return 42;

    // The listing agrees: f (and its marker) invisible, g once, h present.
    DIR *d = opendir(lower);
    if (!d) return 43;
    int f = 0, g = 0, h = 0, other = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0)
            continue;
        else if (strcmp(e->d_name, "f") == 0) f++;
        else if (strcmp(e->d_name, "g") == 0) g++;
        else if (strcmp(e->d_name, "h") == 0) h++;
        else other++;
    }
    closedir(d);
    if (f != 0 || g != 1 || h != 1 || other != 0) return 44;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) return 10;
    if (strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (strcmp(argv[1], "check") == 0) return check(argv[2]);
    return 10;
}
