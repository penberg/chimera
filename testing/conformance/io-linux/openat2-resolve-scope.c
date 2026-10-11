// RUN: %cc %s -o %t && rm -rf %t.d && %runner %t %t.d
//
// openat2's RESOLVE_IN_ROOT and RESOLVE_BENEATH scope a lookup to its dirfd.
// IN_ROOT resolves as if the dirfd were the process root: an absolute path,
// an absolute symlink target, and `..` at the dirfd all stay inside it.
// BENEATH refuses each of those with EXDEV instead. bwrap opens every bind
// source as openat2(oldroot_fd, "/", RESOLVE_IN_ROOT); a resolver that reads
// that "/" as the process root binds the wrong tree.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/openat2.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

static int dfd;

static int open_scoped(const char *path, unsigned long long resolve) {
    struct open_how how = {
        .flags = O_PATH,
        .resolve = resolve,
    };
    return (int)syscall(SYS_openat2, dfd, path, &how, sizeof(how));
}

// `path` under IN_ROOT names the same inode as `expect` relative to dfd.
static int lands_on(const char *path, const char *expect) {
    int fd = open_scoped(path, RESOLVE_IN_ROOT);
    if (fd < 0) {
        fprintf(stderr, "IN_ROOT %s: %s\n", path, strerror(errno));
        return 0;
    }
    struct stat got, want;
    int ok = fstat(fd, &got) == 0 && fstatat(dfd, expect, &want, 0) == 0 &&
             got.st_dev == want.st_dev && got.st_ino == want.st_ino;
    close(fd);
    if (!ok) fprintf(stderr, "IN_ROOT %s: not %s\n", path, expect);
    return ok;
}

static int refused(const char *path, unsigned long long resolve, int err) {
    int fd = open_scoped(path, resolve);
    if (fd >= 0) {
        close(fd);
        fprintf(stderr, "resolve=%#llx %s: opened\n", resolve, path);
        return 0;
    }
    if (errno != err) {
        fprintf(stderr, "resolve=%#llx %s: %s\n", resolve, path, strerror(errno));
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    char path[4096];
    const char *d = argv[1];
    if (mkdir(d, 0755) != 0) return 2;
    snprintf(path, sizeof(path), "%s/sub", d);
    if (mkdir(path, 0755) != 0) return 3;
    snprintf(path, sizeof(path), "%s/sub/f", d);
    int f = open(path, O_CREAT | O_WRONLY, 0644);
    if (f < 0) return 4;
    close(f);
    snprintf(path, sizeof(path), "%s/abs", d);
    if (symlink("/sub", path) != 0) return 5;
    dfd = open(d, O_PATH | O_DIRECTORY);
    if (dfd < 0) return 6;

    int ok = 1;
    ok &= lands_on("/", ".");
    ok &= lands_on("/sub/f", "sub/f");
    ok &= lands_on("../../..", ".");
    ok &= lands_on("sub/../../sub/f", "sub/f");
    ok &= lands_on("abs/f", "sub/f");

    ok &= refused("/sub/f", RESOLVE_BENEATH, EXDEV);
    ok &= refused("..", RESOLVE_BENEATH, EXDEV);
    ok &= refused("sub/../..", RESOLVE_BENEATH, EXDEV);
    ok &= refused("abs/f", RESOLVE_BENEATH, EXDEV);
    int fd = open_scoped("sub/../sub/f", RESOLVE_BENEATH);
    if (fd < 0) {
        fprintf(stderr, "BENEATH sub/../sub/f: %s\n", strerror(errno));
        ok = 0;
    } else {
        close(fd);
    }

    ok &= refused("sub", RESOLVE_IN_ROOT | RESOLVE_BENEATH, EINVAL);
    ok &= refused("sub", 1ULL << 40, EINVAL);
    return ok ? 0 : 7;
}
