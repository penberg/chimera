// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: test -z "%runner" || CHIMERA_FS=%t.fixture/ws %runner %t mutate %t.fixture
// RUN: test -z "%runner" || %t drive %t.fixture "%runner"
//
// Branching filesystems: `fs branch` seeds a new filesystem with a copy of
// the source's change-set — the whiteout included — and prints its id; a
// path selector still opens a delta in place (the scripting escape hatch);
// and `run --from <id>` branches a kept change-set rather than mutating it.
// The drive step runs natively, redirects the state directory into the
// fixture, and skips itself when no delta materialized (native and --unsafe
// suite runs).

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

static int write_file(const char *path, const char *content) {
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) return -1;
    ssize_t n = (ssize_t)strlen(content);
    if (write(fd, content, n) != n) return -1;
    return close(fd);
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
    snprintf(path, sizeof(path), "%s/lower/modified", fixture);
    if (write_file(path, "host v1") != 0) return 13;
    snprintf(path, sizeof(path), "%s/lower/deleted", fixture);
    if (write_file(path, "bye") != 0) return 14;
    return 0;
}

static int mutate(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX];
    if (lower_canon(fixture, lower) != 0) return 21;
    snprintf(path, sizeof(path), "%s/modified", lower);
    if (write_file(path, "guest version") != 0) return 22;
    snprintf(path, sizeof(path), "%s/created", lower);
    if (write_file(path, "fresh") != 0) return 23;
    snprintf(path, sizeof(path), "%s/deleted", lower);
    if (unlink(path) != 0) return 24;
    return 0;
}

static int mutate2(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX];
    if (lower_canon(fixture, lower) != 0) return 26;
    snprintf(path, sizeof(path), "%s/branchling", lower);
    if (write_file(path, "branch only") != 0) return 27;
    return 0;
}

/// The filesystem's diff must be exactly the prep/mutate change-set — one
/// each of A /created, M /modified, D /deleted — plus A /branchling when
/// `branchlings` says the branch was mutated.
static int check_diff(const char *chim, const char *sel, int branchlings) {
    char cmd[PATH_MAX * 2];
    snprintf(cmd, sizeof(cmd), "%s fs diff %s 2>/dev/null", chim, sel);
    FILE *p = popen(cmd, "r");
    if (!p) return 1;
    int a = 0, m = 0, d = 0, b = 0, other = 0;
    char line[PATH_MAX];
    while (fgets(line, sizeof(line), p)) {
        if (strstr(line, "/branchling") && line[0] == 'A') b++;
        else if (strstr(line, "/created") && line[0] == 'A') a++;
        else if (strstr(line, "/modified") && line[0] == 'M') m++;
        else if (strstr(line, "/deleted") && line[0] == 'D') d++;
        else other++;
    }
    if (pclose(p) != 0) return 2;
    if (a != 1 || m != 1 || d != 1 || other != 0) return 3;
    if (b != branchlings) return 4;
    return 0;
}

static int drive(const char *fixture, const char *runner, const char *self) {
    char ws[PATH_MAX], probe[PATH_MAX], state[PATH_MAX];
    snprintf(ws, sizeof(ws), "%s/ws", fixture);
    snprintf(probe, sizeof(probe), "%s/data", ws);
    struct stat st;
    if (stat(probe, &st) != 0) return 0; // native or --unsafe: no delta

    // The chimera binary is the runner's first token.
    char chim[PATH_MAX];
    if (sscanf(runner, "%s", chim) != 1) return 31;
    snprintf(state, sizeof(state), "%s/state", fixture);
    if (setenv("XDG_STATE_HOME", state, 1) != 0) return 32;

    // branch: prints the new filesystem's id.
    char cmd[PATH_MAX * 4], b1[64];
    snprintf(cmd, sizeof(cmd), "%s fs branch %s 2>/dev/null", chim, ws);
    FILE *p = popen(cmd, "r");
    if (!p) return 33;
    if (!fgets(b1, sizeof(b1), p)) {
        pclose(p);
        return 34;
    }
    if (pclose(p) != 0) return 35;
    b1[strcspn(b1, "\n")] = 0;
    if (b1[0] == 0) return 36;

    // The branch carries the source's exact change-set.
    if (check_diff(chim, b1, 0) != 0) return 37;

    // A path selector opens the branch's delta in place: the session grows
    // the branch, never its source.
    snprintf(cmd, sizeof(cmd),
             "CHIMERA_FS=%s/chimera/fs/%s %s %s mutate2 %s >/dev/null 2>&1",
             state, b1, runner, self, fixture);
    if (system(cmd) != 0) return 38;
    if (check_diff(chim, b1, 1) != 0) return 39;
    if (check_diff(chim, ws, 0) != 0) return 40;

    // run --from an id: the kept change-set is branched, never resumed — the
    // run must leave b1 as it was and its own changes in a new filesystem.
    snprintf(cmd, sizeof(cmd), "%s --from %s %s mutate2 %s >/dev/null 2>&1",
             runner, b1, self, fixture);
    if (system(cmd) != 0) return 41;
    if (check_diff(chim, ws, 0) != 0) return 42;
    if (check_diff(chim, b1, 1) != 0) return 51;

    // The branch landed in the state dir as exactly one more filesystem.
    char fsdir[PATH_MAX], b2[64] = "";
    snprintf(fsdir, sizeof(fsdir), "%s/chimera/fs", state);
    DIR *dir = opendir(fsdir);
    if (!dir) return 43;
    struct dirent *e;
    int extra = 0;
    while ((e = readdir(dir))) {
        if (e->d_name[0] == '.' || strcmp(e->d_name, b1) == 0) continue;
        snprintf(b2, sizeof(b2), "%s", e->d_name);
        extra++;
    }
    closedir(dir);
    if (extra != 1) return 44;
    if (check_diff(chim, b2, 1) != 0) return 45;

    // list shows both branches; rm removes them.
    snprintf(cmd, sizeof(cmd), "%s fs list 2>/dev/null", chim);
    p = popen(cmd, "r");
    if (!p) return 46;
    int saw1 = 0, saw2 = 0, from = 0;
    char line[PATH_MAX];
    while (fgets(line, sizeof(line), p)) {
        // Match the id column, which opens the line: a branch's row names its
        // parent too, so a substring search would count b1 twice.
        if (strncmp(line, b1, strlen(b1)) == 0) saw1++;
        if (strncmp(line, b2, strlen(b2)) == 0) {
            saw2++;
            // b2 branched b1, and its row must report the branch point.
            if (strstr(line + strlen(b2), b1)) from++;
        }
    }
    if (pclose(p) != 0) return 47;
    if (saw1 != 1 || saw2 != 1 || from != 1) return 48;
    snprintf(cmd, sizeof(cmd), "%s fs rm %s %s >/dev/null 2>&1", chim, b1, b2);
    if (system(cmd) != 0) return 49;
    snprintf(cmd, sizeof(cmd), "%s/%s", fsdir, b1);
    if (stat(cmd, &st) == 0 || errno != ENOENT) return 50;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (argc == 3 && strcmp(argv[1], "mutate") == 0) return mutate(argv[2]);
    if (argc == 3 && strcmp(argv[1], "mutate2") == 0) return mutate2(argv[2]);
    if (argc == 4 && strcmp(argv[1], "drive") == 0)
        return drive(argv[2], argv[3], argv[0]);
    return 10;
}
