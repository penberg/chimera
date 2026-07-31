// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: test -z "%runner" || CHIMERA_WORKSPACE=%t.fixture/ws %runner %t mutate %t.fixture
// RUN: test -z "%runner" || %t drive %t.fixture "%runner"
//
// The workspace tooling, driven against a workspace a scripted run left
// behind: diff reports the added, modified, and deleted paths; apply adopts
// the changes onto the host but refuses a file whose host copy changed since
// the workspace copied it up (the origin check); rm removes the workspace;
// list exits cleanly. The drive step runs natively, extracts the chimera
// binary from the runner prefix, and skips itself when no delta materialized
// (native and --unsafe suite runs, where the mutations landed directly).

#define _GNU_SOURCE

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

static int append_file(const char *path, const char *content) {
    int fd = open(path, O_WRONLY | O_APPEND);
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
    snprintf(path, sizeof(path), "%s/lower/modified", fixture);
    if (write_file(path, "host v1") != 0) return 13;
    snprintf(path, sizeof(path), "%s/lower/conflicted", fixture);
    if (write_file(path, "host v1") != 0) return 14;
    snprintf(path, sizeof(path), "%s/lower/deleted", fixture);
    if (write_file(path, "bye") != 0) return 15;
    return 0;
}

static int mutate(const char *fixture) {
    char lower[PATH_MAX], path[PATH_MAX];
    if (lower_canon(fixture, lower) != 0) return 21;
    snprintf(path, sizeof(path), "%s/modified", lower);
    if (write_file(path, "guest version") != 0) return 22;
    snprintf(path, sizeof(path), "%s/conflicted", lower);
    if (write_file(path, "guest version") != 0) return 23;
    snprintf(path, sizeof(path), "%s/created", lower);
    if (write_file(path, "fresh") != 0) return 24;
    snprintf(path, sizeof(path), "%s/deleted", lower);
    if (unlink(path) != 0) return 25;
    return 0;
}

/// Run `<chimera> workspace <args>`, capturing stdout; the exit status goes
/// to *status.
static FILE *tool(const char *chim, const char *args, const char *ws) {
    char cmd[PATH_MAX * 2];
    snprintf(cmd, sizeof(cmd), "%s workspace %s %s 2>/dev/null", chim, args, ws);
    return popen(cmd, "r");
}

static int drive(const char *fixture, const char *runner) {
    char ws[PATH_MAX], probe[PATH_MAX], lower[PATH_MAX], buf[256];
    snprintf(ws, sizeof(ws), "%s/ws", fixture);
    snprintf(probe, sizeof(probe), "%s/data", ws);
    struct stat st;
    if (stat(probe, &st) != 0) return 0; // native or --unsafe: no delta
    if (lower_canon(fixture, lower) != 0) return 31;

    // The chimera binary is the runner's first token.
    char chim[PATH_MAX];
    if (sscanf(runner, "%s", chim) != 1) return 32;

    // diff: exactly the four changes, tagged correctly.
    FILE *p = tool(chim, "diff", ws);
    if (!p) return 33;
    int a = 0, m = 0, c = 0, d = 0, other = 0;
    char line[PATH_MAX];
    while (fgets(line, sizeof(line), p)) {
        if (strstr(line, "/created") && line[0] == 'A') a++;
        else if (strstr(line, "/modified") && line[0] == 'M') m++;
        else if (strstr(line, "/conflicted") && line[0] == 'M') c++;
        else if (strstr(line, "/deleted") && line[0] == 'D') d++;
        else other++;
    }
    if (pclose(p) != 0) return 34;
    if (a != 1 || m != 1 || c != 1 || d != 1 || other != 0) return 35;

    // Sour one file on the host so its origin no longer matches.
    snprintf(buf, sizeof(buf), "%s/conflicted", lower);
    if (append_file(buf, " + host edit") != 0) return 36;

    // apply: reports failure (the conflict) but adopts everything else.
    char cmd[PATH_MAX * 2];
    snprintf(cmd, sizeof(cmd), "%s workspace apply %s >/dev/null 2>&1", chim, ws);
    if (system(cmd) == 0) return 37; // must fail: one conflict

    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/modified", lower);
    if (read_file(path, buf, sizeof(buf)) != 0) return 38;
    if (strcmp(buf, "guest version") != 0) return 39;
    snprintf(path, sizeof(path), "%s/created", lower);
    if (read_file(path, buf, sizeof(buf)) != 0) return 40;
    if (strcmp(buf, "fresh") != 0) return 41;
    snprintf(path, sizeof(path), "%s/deleted", lower);
    if (stat(path, &st) == 0 || errno != ENOENT) return 42;
    snprintf(path, sizeof(path), "%s/conflicted", lower);
    if (read_file(path, buf, sizeof(buf)) != 0) return 43;
    if (strcmp(buf, "host v1 + host edit") != 0) return 44; // refused, intact

    // list: runs cleanly (the path workspace is not under the state dir).
    snprintf(cmd, sizeof(cmd), "%s workspace list >/dev/null 2>&1", chim);
    if (system(cmd) != 0) return 45;

    // rm: the workspace is gone.
    snprintf(cmd, sizeof(cmd), "%s workspace rm %s >/dev/null 2>&1", chim, ws);
    if (system(cmd) != 0) return 46;
    if (stat(ws, &st) == 0 || errno != ENOENT) return 47;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (argc == 3 && strcmp(argv[1], "mutate") == 0) return mutate(argv[2]);
    if (argc == 4 && strcmp(argv[1], "drive") == 0)
        return drive(argv[2], argv[3]);
    return 10;
}
