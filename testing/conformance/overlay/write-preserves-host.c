// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: CHIMERA_WORKSPACE=%t.fixture/delta %runner %t check %t.fixture
// RUN: %t verify %t.fixture
//
// The write path: a guest write-open of a lower file copies it up and the
// write lands in the delta; O_CREAT creates in the delta; O_APPEND
// interleaves through two descriptors the way one kernel file does. The
// check step asserts plain POSIX write semantics, true under the overlay and
// natively alike; the verify step then runs natively and branches on whether
// a delta materialized — if it did (the overlay ran), the lower file must be
// byte-identical to what prep wrote and the created file absent from the
// host; natively the mutations landed directly and must read back mutated.

#define _GNU_SOURCE

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

static void lower_path(const char *fixture, const char *name, char *out,
                       size_t size) {
    char lower[PATH_MAX], canon[PATH_MAX];
    snprintf(lower, sizeof(lower), "%s/lower", fixture);
    if (!realpath(lower, canon)) canon[0] = 0;
    snprintf(out, size, "%s/%s", canon, name);
}

static int prep(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX];
    snprintf(lower, sizeof(lower), "%s/lower", fixture);
    if (mkdirs(lower) != 0) return 11;
    snprintf(path, sizeof(path), "%s/target", lower);
    if (write_file(path, "pristine host bytes") != 0) return 12;
    snprintf(path, sizeof(path), "%s/log", lower);
    if (write_file(path, "") != 0) return 13;
    return 0;
}

static int check(const char *fixture) {
    char path[PATH_MAX], buf[128];

    // Overwrite a lower file through a write-open.
    lower_path(fixture, "target", path, sizeof(path));
    int fd = open(path, O_RDWR);
    if (fd < 0) return 31;
    if (pwrite(fd, "guest was here.....", 19, 0) != 19) return 32;
    close(fd);
    if (read_file(path, buf, sizeof(buf)) != 0) return 33;
    if (strcmp(buf, "guest was here.....") != 0) return 34;

    // Create a brand-new file next to it.
    lower_path(fixture, "created", path, sizeof(path));
    if (write_file(path, "made by the guest") != 0) return 35;
    if (read_file(path, buf, sizeof(buf)) != 0) return 36;
    if (strcmp(buf, "made by the guest") != 0) return 37;

    // O_APPEND through two descriptors of one file interleaves at EOF: six
    // alternating writes always total twelve bytes, whatever the order.
    lower_path(fixture, "log", path, sizeof(path));
    int a = open(path, O_WRONLY | O_APPEND);
    int b = open(path, O_WRONLY | O_APPEND);
    if (a < 0 || b < 0) return 38;
    for (int i = 0; i < 3; i++) {
        if (write(a, "aa", 2) != 2) return 39;
        if (write(b, "bb", 2) != 2) return 40;
    }
    close(a);
    close(b);
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size != 12) return 41;
    if (read_file(path, buf, sizeof(buf)) != 0) return 42;
    if (strcmp(buf, "aabbaabbaabb") != 0) return 43;
    return 0;
}

static int verify(const char *fixture) {
    char path[PATH_MAX], probe[PATH_MAX], buf[128];

    // Did a delta materialize? Its data/ mirrors the canonical lower path.
    char lower[PATH_MAX], canon[PATH_MAX];
    snprintf(lower, sizeof(lower), "%s/lower", fixture);
    if (!realpath(lower, canon)) return 51;
    snprintf(probe, sizeof(probe), "%s/delta/data%s/target", fixture, canon);
    struct stat st;
    int overlay_ran = stat(probe, &st) == 0;

    snprintf(path, sizeof(path), "%s/target", canon);
    if (read_file(path, buf, sizeof(buf)) != 0) return 52;
    if (overlay_ran) {
        // The host file is byte-identical to what prep wrote…
        if (strcmp(buf, "pristine host bytes") != 0) return 53;
        // …the guest's version lives in the delta…
        if (read_file(probe, buf, sizeof(buf)) != 0) return 54;
        if (strcmp(buf, "guest was here.....") != 0) return 55;
        // …and the created file never touched the host.
        snprintf(path, sizeof(path), "%s/created", canon);
        if (stat(path, &st) == 0 || errno != ENOENT) return 56;
        snprintf(path, sizeof(path), "%s/log", canon);
        if (stat(path, &st) != 0 || st.st_size != 0) return 57;
    } else {
        // Native check run: the writes landed directly.
        if (strcmp(buf, "guest was here.....") != 0) return 58;
        snprintf(path, sizeof(path), "%s/created", canon);
        if (stat(path, &st) != 0) return 59;
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
