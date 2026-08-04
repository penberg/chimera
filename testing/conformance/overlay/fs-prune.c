// RUN: %cc %s -o %t && rm -rf %t.fixture
// RUN: test -z "%runner" || %t drive %t.fixture "%runner"
// UNSUPPORTED: darwin -- drives the workspace CLI, a Linux-only feature
//
// `fs prune` sweeps the state directory: a filesystem no session
// holds is removed (including one from before the lock file existed), a
// filesystem some live session's tree still holds survives, and without -f
// a prompt fed from a closed stdin declines and removes nothing. The test
// runs natively against a private XDG_STATE_HOME, extracting the chimera
// binary from the runner prefix.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

static int write_file(const char *path, const char *content) {
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) return -1;
    ssize_t n = (ssize_t)strlen(content);
    if (write(fd, content, n) != n) return -1;
    return close(fd);
}

/// Lay out one fake kept filesystem: `data/` with a delta file, `meta`, and
/// (unless lockless) the `lock` file sessions hold shares on.
static int fake_filesystem(const char *base, const char *id, int lockless) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", base, id);
    if (mkdir(path, 0755) != 0) return -1;
    snprintf(path, sizeof(path), "%s/%s/data", base, id);
    if (mkdir(path, 0755) != 0) return -1;
    snprintf(path, sizeof(path), "%s/%s/data/delta", base, id);
    if (write_file(path, "unapplied") != 0) return -1;
    snprintf(path, sizeof(path), "%s/%s/meta", base, id);
    if (write_file(path, "command = fake\ncreated = 0\n") != 0) return -1;
    if (lockless) return 0;
    snprintf(path, sizeof(path), "%s/%s/lock", base, id);
    return write_file(path, "");
}

static int exists(const char *base, const char *id) {
    char path[PATH_MAX];
    struct stat st;
    snprintf(path, sizeof(path), "%s/%s", base, id);
    return stat(path, &st) == 0;
}

static int prune(const char *chim, const char *state, const char *flags) {
    char cmd[PATH_MAX * 2];
    snprintf(cmd, sizeof(cmd),
             "XDG_STATE_HOME=%s %s fs prune %s </dev/null "
             ">/dev/null 2>&1",
             state, chim, flags);
    return system(cmd);
}

static int drive(const char *fixture, const char *runner) {
    char chim[PATH_MAX];
    if (sscanf(runner, "%s", chim) != 1) return 11;

    char state[PATH_MAX], base[PATH_MAX], path[PATH_MAX];
    snprintf(state, sizeof(state), "%s/state", fixture);
    snprintf(base, sizeof(base), "%s/chimera/fs", state);
    if (mkdir(fixture, 0755) != 0) return 12;
    snprintf(path, sizeof(path), "%s/chimera", state);
    if (mkdir(state, 0755) != 0 || mkdir(path, 0755) != 0 ||
        mkdir(base, 0755) != 0)
        return 13;
    if (fake_filesystem(base, "stale", 0) != 0) return 14;
    if (fake_filesystem(base, "prelock", 1) != 0) return 15;
    if (fake_filesystem(base, "busy", 0) != 0) return 16;

    // A live session's share of the busy filesystem's tree-wide hold.
    snprintf(path, sizeof(path), "%s/busy/lock", base);
    int held = open(path, O_RDWR);
    if (held < 0 || flock(held, LOCK_SH) != 0) return 17;

    // Declined prompt (stdin is closed): exits cleanly, removes nothing.
    if (prune(chim, state, "") != 0) return 18;
    if (!exists(base, "stale") || !exists(base, "prelock") ||
        !exists(base, "busy"))
        return 19;

    // Forced: the unheld filesystems go, lock file or not; busy survives.
    if (prune(chim, state, "-f") != 0) return 20;
    if (exists(base, "stale") || exists(base, "prelock")) return 21;
    if (!exists(base, "busy")) return 22;

    // Released, the busy filesystem is residue like any other.
    close(held);
    if (prune(chim, state, "-f") != 0) return 23;
    if (exists(base, "busy")) return 24;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 4 && strcmp(argv[1], "drive") == 0)
        return drive(argv[2], argv[3]);
    return 10;
}
