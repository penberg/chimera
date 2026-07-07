// RUN: %cc %s -o %t && rm -rf %t.fixture && %t prep %t.fixture
// RUN: CHIMERA_WORKSPACE=%t.fixture/delta timeout 10 %runner %t check %t.fixture
// RUN: %t verify %t.fixture
//
// Exec coherence: a guest that rewrites a binary and execve's the path must
// run the rewritten version — host_path() answers from the serving layer, so
// the loader reads the delta's copy. The fixture binary starts as /bin/false
// (old version: exit 1); the guest overwrites it with this test binary
// (which, in exec-proof mode, exits 77) and execs it. The exec'd image also
// proves the close-on-exec sweep is unchanged under the overlay: a plain fd
// survives, a CLOEXEC fd does not. Natively the same flow rewrites the host
// file directly; verify branches and asserts the host kept /bin/false's
// bytes when the overlay ran.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static int copy_file(const char *from, const char *to, mode_t mode) {
    int src = open(from, O_RDONLY);
    if (src < 0) return -1;
    int dst = open(to, O_CREAT | O_TRUNC | O_WRONLY, mode);
    if (dst < 0) {
        close(src);
        return -1;
    }
    char buf[65536];
    ssize_t n;
    while ((n = read(src, buf, sizeof(buf))) > 0) {
        for (ssize_t off = 0; off < n;) {
            ssize_t w = write(dst, buf + off, n - off);
            if (w <= 0) return -1;
            off += w;
        }
    }
    close(src);
    if (fchmod(dst, mode) != 0) return -1;
    return n < 0 ? -1 : close(dst);
}

static void prog_path(const char *fixture, char *out) {
    char lower[PATH_MAX], canon[PATH_MAX];
    snprintf(lower, sizeof(lower), "%s/lower", fixture);
    if (!realpath(lower, canon)) canon[0] = 0;
    snprintf(out, PATH_MAX, "%s/prog", canon);
}

static int prep(const char *fixture) {
    char path[PATH_MAX];
    if (mkdir(fixture, 0755) != 0 && errno != EEXIST) return 11;
    snprintf(path, sizeof(path), "%s/lower", fixture);
    if (mkdir(path, 0755) != 0 && errno != EEXIST) return 12;
    snprintf(path, sizeof(path), "%s/lower/prog", fixture);
    if (copy_file("/bin/false", path, 0755) != 0) return 13;
    // Record the old version's size so verify can prove the host kept it.
    struct stat st;
    if (stat(path, &st) != 0) return 14;
    snprintf(path, sizeof(path), "%s/oldsize", fixture);
    FILE *f = fopen(path, "w");
    if (!f || fprintf(f, "%lld", (long long)st.st_size) < 0 || fclose(f))
        return 15;
    return 0;
}

static int check(const char *fixture, const char *self) {
    char prog[PATH_MAX];
    prog_path(fixture, prog);

    // Rewrite the binary through the overlay: old version out, this test
    // binary in.
    if (copy_file(self, prog, 0755) != 0) return 31;

    // One fd to survive the exec, one to be swept by close-on-exec.
    int keep[2], gone[2];
    if (pipe(keep) != 0 || pipe(gone) != 0) return 32;
    if (fcntl(gone[1], F_SETFD, FD_CLOEXEC) != 0) return 33;

    pid_t pid = fork();
    if (pid < 0) return 34;
    if (pid == 0) {
        char keep_str[16], gone_str[16];
        snprintf(keep_str, sizeof(keep_str), "%d", keep[1]);
        snprintf(gone_str, sizeof(gone_str), "%d", gone[1]);
        char *args[] = {prog, "exec-proof", keep_str, gone_str, NULL};
        execve(prog, args, environ);
        _exit(90); // exec failed
    }
    close(keep[1]);
    close(gone[1]);

    // The rewritten version ran (77), not /bin/false's 1.
    int status;
    if (waitpid(pid, &status, 0) != pid) return 35;
    if (!WIFEXITED(status)) return 36;
    if (WEXITSTATUS(status) != 77) return WEXITSTATUS(status) == 1 ? 37 : 38;

    // The kept pipe carried the child's byte; the CLOEXEC end delivered EOF.
    char t;
    if (read(keep[0], &t, 1) != 1 || t != 'k') return 39;
    if (read(gone[0], &t, 1) != 0) return 40;
    return 0;
}

static int exec_proof(const char *keep_str, const char *gone_str) {
    int keep_fd = atoi(keep_str);
    int gone_fd = atoi(gone_str);
    if (fcntl(gone_fd, F_GETFD) != -1 || errno != EBADF) return 60;
    if (write(keep_fd, "k", 1) != 1) return 61;
    return 77;
}

static int verify(const char *fixture) {
    char prog[PATH_MAX], path[PATH_MAX], probe[PATH_MAX], buf[32];
    prog_path(fixture, prog);
    snprintf(probe, sizeof(probe), "%s/delta/data%s", fixture, prog);

    struct stat st;
    if (stat(probe, &st) != 0) return 0; // native run: rewrite hit the host

    // The delta holds the rewritten binary; the host still holds the old
    // version, byte for byte in size.
    snprintf(path, sizeof(path), "%s/oldsize", fixture);
    FILE *f = fopen(path, "r");
    if (!f || !fgets(buf, sizeof(buf), f)) return 51;
    fclose(f);
    long long oldsize = atoll(buf);
    struct stat host;
    if (stat(prog, &host) != 0) return 52;
    if (host.st_size != oldsize) return 53;
    if (st.st_size == oldsize) return 54;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 4 && strcmp(argv[1], "exec-proof") == 0)
        return exec_proof(argv[2], argv[3]);
    if (argc != 3) return 10;
    if (strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (strcmp(argv[1], "check") == 0) return check(argv[2], "/proc/self/exe");
    if (strcmp(argv[1], "verify") == 0) return verify(argv[2]);
    return 10;
}
