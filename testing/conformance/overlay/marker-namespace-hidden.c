// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: CHIMERA_FS=%t.fixture/delta %runner %t check %t.fixture
// RUN: %t verify %t.fixture
//
// The overlay's bookkeeping xattr namespace (user.chimera.*) is the guest's
// blind spot: a copied-up file carries user.chimera.origin, but getxattr
// answers ENODATA, listxattr omits it, and setxattr/removexattr refuse the
// namespace with EPERM — a guest can neither read nor forge markers, while
// its own user.* attributes keep working. The check step discovers its world
// by probing a marker-name setxattr: EPERM means the filter (overlay) is
// active; success means a native run, where the name is an ordinary xattr.

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
    return 0;
}

static int list_has_marker(const char *list, ssize_t len) {
    for (ssize_t i = 0; i < len;) {
        if (strncmp(list + i, "user.chimera.", 13) == 0) return 1;
        i += strlen(list + i) + 1;
    }
    return 0;
}

static int check(const char *fixture) {
    char path[PATH_MAX], buf[256];
    if (lower_target(fixture, path, sizeof(path)) != 0) return 31;

    // Force a copy-up so the upper file carries user.chimera.origin.
    int fd = open(path, O_RDWR);
    if (fd < 0) return 32;
    if (pwrite(fd, "y", 1, 0) != 1) return 33;

    // World probe: forging a marker either hits the filter or is a plain
    // user xattr.
    int r = setxattr(path, "user.chimera.forged", "1", 1, 0);
    if (r == 0) {
        // Native (no filter): clean up and pass.
        close(fd);
        return removexattr(path, "user.chimera.forged") == 0 ? 0 : 34;
    }
    if (errno == ENOTSUP) {
        close(fd);
        return 0; // no user xattrs here at all; nothing to hide
    }
    if (errno != EPERM) return 35;

    // The origin marker exists on the upper file, but reads as absent…
    if (getxattr(path, "user.chimera.origin", buf, sizeof(buf)) >= 0 ||
        errno != ENODATA)
        return 36;
    if (fgetxattr(fd, "user.chimera.origin", buf, sizeof(buf)) >= 0 ||
        errno != ENODATA)
        return 37;

    // …vanishes from listings…
    ssize_t n = listxattr(path, buf, sizeof(buf));
    if (n < 0 || list_has_marker(buf, n)) return 38;
    n = flistxattr(fd, buf, sizeof(buf));
    if (n < 0 || list_has_marker(buf, n)) return 39;

    // …and can be neither forged nor removed, by path or by fd.
    if (fsetxattr(fd, "user.chimera.whiteout", "1", 1, 0) == 0 ||
        errno != EPERM)
        return 40;
    if (removexattr(path, "user.chimera.origin") == 0 || errno != EPERM)
        return 41;
    if (fremovexattr(fd, "user.chimera.origin") == 0 || errno != EPERM)
        return 42;

    // The guest's own attributes are unaffected.
    if (setxattr(path, "user.mine", "v", 1, 0) != 0) return 43;
    if (getxattr(path, "user.mine", buf, sizeof(buf)) != 1 || buf[0] != 'v')
        return 44;
    n = listxattr(path, buf, sizeof(buf));
    if (n < 0) return 45;
    int mine = 0;
    for (ssize_t i = 0; i < n; i += (ssize_t)strlen(buf + i) + 1)
        if (strcmp(buf + i, "user.mine") == 0) mine = 1;
    if (!mine) return 46;
    if (removexattr(path, "user.mine") != 0) return 47;

    close(fd);
    return 0;
}

static int verify(const char *fixture) {
    char path[PATH_MAX], probe[PATH_MAX], buf[256];
    if (lower_target(fixture, path, sizeof(path)) != 0) return 51;
    snprintf(probe, sizeof(probe), "%s/delta/data%s", fixture, path);

    struct stat st;
    if (stat(probe, &st) != 0) return 0; // native check run: nothing to prove

    // The marker the guest could not see is really there on the upper file,
    // and the guest's forgeries are not.
    if (getxattr(probe, "user.chimera.origin", buf, sizeof(buf)) <= 0)
        return 52;
    if (getxattr(probe, "user.chimera.forged", buf, sizeof(buf)) >= 0 ||
        errno != ENODATA)
        return 53;
    if (getxattr(probe, "user.chimera.whiteout", buf, sizeof(buf)) >= 0 ||
        errno != ENODATA)
        return 54;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) return 10;
    if (strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (strcmp(argv[1], "check") == 0) return check(argv[2]);
    if (strcmp(argv[1], "verify") == 0) return verify(argv[2]);
    return 10;
}
