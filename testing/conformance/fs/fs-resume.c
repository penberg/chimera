// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: case "%runner" in *--unsafe*) : ;; *chimera*) %t drive %t.fixture "%runner" ;; *) : ;; esac
//
// Resume: `run --in <id>` works inside a kept filesystem — same id, no copy
// — so consecutive sessions accumulate one change-set instead of leaving a
// branch per return. The drive step, run natively against a private state
// directory, also pins the verbs' guard rails: `--from` and `--in` are
// exclusive, `--rm` is refused with `--in`, `--in host` is refused and
// points at --unsafe, and a scheme-shaped locator is reserved rather than
// silently read as a path. Every refusal happens before any guest runs, so
// the state directory afterward is exactly as the two sessions left it.

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

static int prep(const char *fixture) {
    char path[PATH_MAX];
    if (mkdir(fixture, 0755) != 0 && errno != EEXIST) return 11;
    snprintf(path, sizeof(path), "%s/lower", fixture);
    if (mkdir(path, 0755) != 0 && errno != EEXIST) return 12;
    return 0;
}

static int mark(const char *name, const char *fixture) {
    char lower[PATH_MAX], canon[PATH_MAX], path[PATH_MAX];
    snprintf(lower, sizeof(lower), "%s/lower", fixture);
    if (!realpath(lower, canon)) return 21;
    snprintf(path, sizeof(path), "%s/mark-%s", canon, name);
    return write_file(path, name) == 0 ? 0 : 22;
}

/// The single filesystem the state directory holds, or nonzero when there is
/// none or more than one.
static int only_entry(const char *state, char *id, size_t size) {
    char fsdir[PATH_MAX];
    snprintf(fsdir, sizeof(fsdir), "%s/chimera/fs", state);
    DIR *dir = opendir(fsdir);
    if (!dir) return 1;
    struct dirent *e;
    int count = 0;
    while ((e = readdir(dir))) {
        if (e->d_name[0] == '.') continue;
        snprintf(id, size, "%s", e->d_name);
        count++;
    }
    closedir(dir);
    return count == 1 ? 0 : 2;
}

/// The one change-set carries both sessions' marks and nothing else.
static int check_diff(const char *chim, const char *id) {
    char cmd[PATH_MAX * 2];
    snprintf(cmd, sizeof(cmd), "%s fs diff %s 2>/dev/null", chim, id);
    FILE *p = popen(cmd, "r");
    if (!p) return 1;
    int one = 0, two = 0, other = 0;
    char line[PATH_MAX];
    while (fgets(line, sizeof(line), p)) {
        if (strstr(line, "/mark-one") && line[0] == 'A') one++;
        else if (strstr(line, "/mark-two") && line[0] == 'A') two++;
        else other++;
    }
    if (pclose(p) != 0) return 2;
    return one == 1 && two == 1 && other == 0 ? 0 : 3;
}

static int drive(const char *fixture, const char *runner, const char *self) {
    char chim[PATH_MAX], state[PATH_MAX], cmd[PATH_MAX * 4];
    // The chimera binary is the runner's first token.
    if (sscanf(runner, "%s", chim) != 1) return 31;
    snprintf(state, sizeof(state), "%s/state", fixture);
    if (setenv("XDG_STATE_HOME", state, 1) != 0) return 32;

    // Session one: a plain run branches the host and keeps its filesystem.
    snprintf(cmd, sizeof(cmd), "%s %s mark one %s >/dev/null 2>&1", runner,
             self, fixture);
    if (system(cmd) != 0) return 33;
    char id[64];
    if (only_entry(state, id, sizeof(id)) != 0) return 34;

    // Session two resumes it: same id, no copy, no second filesystem.
    snprintf(cmd, sizeof(cmd), "%s --in %s %s mark two %s >/dev/null 2>&1",
             runner, id, self, fixture);
    if (system(cmd) != 0) return 35;
    char again[64];
    if (only_entry(state, again, sizeof(again)) != 0) return 36;
    if (strcmp(id, again) != 0) return 37;
    if (check_diff(chim, id) != 0) return 38;

    // The guard rails: each command must be refused.
    snprintf(cmd, sizeof(cmd), "%s --in host %s mark refused %s >/dev/null 2>&1",
             runner, self, fixture);
    if (system(cmd) == 0) return 39;
    snprintf(cmd, sizeof(cmd),
             "%s --rm --in %s %s mark refused %s >/dev/null 2>&1", runner, id,
             self, fixture);
    if (system(cmd) == 0) return 40;
    snprintf(cmd, sizeof(cmd),
             "%s --from %s --in %s %s mark refused %s >/dev/null 2>&1", runner,
             id, id, self, fixture);
    if (system(cmd) == 0) return 41;
    snprintf(cmd, sizeof(cmd),
             "%s --from bad:scheme %s mark refused %s >/dev/null 2>&1", runner,
             self, fixture);
    if (system(cmd) == 0) return 42;

    // No refusal ran a guest or minted a filesystem.
    if (only_entry(state, again, sizeof(again)) != 0) return 43;
    if (strcmp(id, again) != 0) return 44;
    if (check_diff(chim, id) != 0) return 45;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (argc == 4 && strcmp(argv[1], "mark") == 0) return mark(argv[2], argv[3]);
    if (argc == 4 && strcmp(argv[1], "drive") == 0)
        return drive(argv[2], argv[3], argv[0]);
    return 10;
}
