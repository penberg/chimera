// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: CHIMERA_FS=%t.fixture/delta %runner %t check %t.fixture
// RUN: %t verify %t.fixture
//
// Fork coherence — the payoff of encoding all overlay state in the
// filesystem: Chimera emulates guest fork with a host fork, so one sandbox is
// many host processes, and any in-memory index would tear at the first fork.
// A parent's unlink (a whiteout) must be visible in an already-forked child,
// and a child's copy-up must be visible in the parent — with no mechanism
// beyond the shared delta tree. The lockstep runs over pipes; the assertions
// are plain POSIX truths natively as well. Verify then proves the host never
// changed and the delta carries the whiteout and the copy-up.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/xattr.h>
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
    snprintf(path, sizeof(path), "%s/lower/doomed", fixture);
    if (write_file(path, "bye") != 0) return 13;
    snprintf(path, sizeof(path), "%s/lower/shared", fixture);
    if (write_file(path, "lower") != 0) return 14;
    return 0;
}

static int check(const char *fixture) {
    char lower[PATH_MAX], doomed[PATH_MAX], shared[PATH_MAX], buf[64];
    if (lower_canon(fixture, lower) != 0) return 30;
    snprintf(doomed, sizeof(doomed), "%s/doomed", lower);
    snprintf(shared, sizeof(shared), "%s/shared", lower);

    // Watchdog: a wedged lockstep must fail loudly (SIGALRM's default
    // action), not hang the suite.
    alarm(8);

    int to_child[2], to_parent[2];
    if (pipe(to_child) != 0 || pipe(to_parent) != 0) return 31;

    // Premise, checked before the fork so it cannot race the unlink: the
    // child is born while `doomed` is still visible.
    struct stat pre;
    if (stat(doomed, &pre) != 0) return 30;

    pid_t pid = fork();
    if (pid < 0) return 32;
    if (pid == 0) {
        // Closing the unused ends turns any early death into an EOF for the
        // peer instead of a deadlock.
        close(to_child[1]);
        close(to_parent[0]);
        char t;
        struct stat st;
        if (read(to_child[0], &t, 1) != 1) _exit(41);
        // The parent has unlinked it; the whiteout must reach us.
        if (stat(doomed, &st) == 0 || errno != ENOENT) _exit(42);
        // Our copy-up must reach the parent.
        int fd = open(shared, O_RDWR);
        if (fd < 0) _exit(43);
        if (pwrite(fd, "child", 5, 0) != 5) _exit(44);
        close(fd);
        if (write(to_parent[1], "y", 1) != 1) _exit(45);
        _exit(0);
    }

    close(to_child[0]);
    close(to_parent[1]);
    if (unlink(doomed) != 0) return 33;
    if (write(to_child[1], "x", 1) != 1) return 34;
    char t;
    if (read(to_parent[0], &t, 1) != 1) return 35;
    if (read_file(shared, buf, sizeof(buf)) != 0) return 36;
    if (strcmp(buf, "child") != 0) return 37;

    int status;
    if (waitpid(pid, &status, 0) != pid) return 38;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        return WIFEXITED(status) ? WEXITSTATUS(status) : 39;
    return 0;
}

static int verify(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX], probe[PATH_MAX], buf[64];
    if (lower_canon(fixture, lower) != 0) return 51;
    snprintf(probe, sizeof(probe), "%s/delta/data%s", fixture, lower);

    struct stat st;
    if (stat(probe, &st) != 0) {
        // Native check run: the mutations landed on the host.
        snprintf(path, sizeof(path), "%s/doomed", lower);
        if (stat(path, &st) == 0) return 52;
        return 0;
    }

    // The host survived both processes' mutations…
    snprintf(path, sizeof(path), "%s/doomed", lower);
    if (read_file(path, buf, sizeof(buf)) != 0 || strcmp(buf, "bye") != 0)
        return 53;
    snprintf(path, sizeof(path), "%s/shared", lower);
    if (read_file(path, buf, sizeof(buf)) != 0 || strcmp(buf, "lower") != 0)
        return 54;
    // …the parent's unlink is a whiteout, and the child's copy-up is real.
    snprintf(path, sizeof(path), "%s/doomed", probe);
    if (getxattr(path, "user.chimera.whiteout", buf, sizeof(buf)) <= 0)
        return 55;
    snprintf(path, sizeof(path), "%s/shared", probe);
    if (read_file(path, buf, sizeof(buf)) != 0 || strcmp(buf, "child") != 0)
        return 56;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) return 10;
    if (strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (strcmp(argv[1], "check") == 0) return check(argv[2]);
    if (strcmp(argv[1], "verify") == 0) return verify(argv[2]);
    return 10;
}
