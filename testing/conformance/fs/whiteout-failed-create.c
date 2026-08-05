// RUN: %cc %s -o %t && rm -rf %t.d && %t prep %t.d
// RUN: %runner %t check %t.d
//
// A failed syscall must leave the merged namespace exactly as it was. The
// deleted names here are hidden by whiteouts; a creation over one that
// fails partway — a symlink with an overlong target (ENAMETOOLONG), an
// unprivileged device node (EPERM) — must not eat the marker and resurrect
// the entry the guest deleted. After each failure the name must still read
// as deleted, by lookup and in the parent's listing, and a subsequent valid
// creation must take the name over normally — including a directory whose
// deleted lower children must never show through it. Natively the failed
// calls change nothing by definition.

#define _GNU_SOURCE

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

static int write_file(const char *path, const char *content) {
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) return -1;
    ssize_t n = (ssize_t)strlen(content);
    if (write(fd, content, n) != n) return -1;
    return close(fd);
}

static int prep(const char *d) {
    char p[4096];
    if (mkdir(d, 0755) != 0) return 11;
    snprintf(p, sizeof(p), "%s/linkname", d);
    if (write_file(p, "old") != 0) return 12;
    snprintf(p, sizeof(p), "%s/nodename", d);
    if (write_file(p, "old") != 0) return 13;
    snprintf(p, sizeof(p), "%s/dirname", d);
    if (mkdir(p, 0755) != 0) return 14;
    snprintf(p, sizeof(p), "%s/dirname/lower-child", d);
    if (write_file(p, "hidden") != 0) return 15;
    return 0;
}

static int listed(const char *dir, const char *name) {
    DIR *dp = opendir(dir);
    if (!dp) return -1;
    struct dirent *e;
    int found = 0;
    while ((e = readdir(dp)))
        if (strcmp(e->d_name, name) == 0) found = 1;
    closedir(dp);
    return found;
}

static int deleted(const char *d, const char *name) {
    char p[4096];
    struct stat st;
    snprintf(p, sizeof(p), "%s/%s", d, name);
    if (lstat(p, &st) == 0 || errno != ENOENT) return 1;
    if (listed(d, name) != 0) return 2;
    return 0;
}

static int check(const char *d) {
    char p[4096];
    struct stat st;

    // Delete the lower entries; every name below starts as a whiteout.
    snprintf(p, sizeof(p), "%s/linkname", d);
    if (unlink(p) != 0) return 21;
    snprintf(p, sizeof(p), "%s/nodename", d);
    if (unlink(p) != 0) return 22;
    snprintf(p, sizeof(p), "%s/dirname/lower-child", d);
    if (unlink(p) != 0) return 23;
    snprintf(p, sizeof(p), "%s/dirname", d);
    if (rmdir(p) != 0) return 24;

    // A symlink whose target is too long fails ENAMETOOLONG — and the name
    // must stay deleted, not fall back to the old lower file.
    char long_target[PATH_MAX + 16];
    memset(long_target, 'x', sizeof(long_target) - 1);
    long_target[sizeof(long_target) - 1] = 0;
    snprintf(p, sizeof(p), "%s/linkname", d);
    if (symlink(long_target, p) == 0 || errno != ENAMETOOLONG) return 31;
    if (deleted(d, "linkname") != 0) return 32;

    // An unprivileged block-device node fails EPERM — same requirement.
    snprintf(p, sizeof(p), "%s/nodename", d);
    if (mknod(p, S_IFBLK | 0644, makedev(1, 3)) == 0 || errno != EPERM)
        return 33;
    if (deleted(d, "nodename") != 0) return 34;

    // Valid creations then take the names over normally.
    snprintf(p, sizeof(p), "%s/linkname", d);
    if (symlink("fresh-target", p) != 0) return 41;
    char buf[64];
    ssize_t n = readlink(p, buf, sizeof(buf) - 1);
    if (n < 0) return 42;
    buf[n] = 0;
    if (strcmp(buf, "fresh-target") != 0) return 43;

    snprintf(p, sizeof(p), "%s/nodename", d);
    if (mknod(p, S_IFIFO | 0644, 0) != 0) return 44;
    if (lstat(p, &st) != 0 || !S_ISFIFO(st.st_mode)) return 45;

    // A directory over its whiteout arrives opaque: the deleted lower child
    // is never visible through it.
    snprintf(p, sizeof(p), "%s/dirname", d);
    if (mkdir(p, 0755) != 0) return 46;
    if (listed(p, "lower-child") != 0) return 47;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) return 10;
    if (strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (strcmp(argv[1], "check") == 0) return check(argv[2]);
    return 10;
}
