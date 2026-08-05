// RUN: %cc %s -o %t && rm -rf %t.d && %t prep %t.d
// RUN: %runner %t race %t.d
//
// Copy-up publication race: two processes that both still see only the lower
// file each build a staging copy, and publication must be first-wins. The
// early opener holds a descriptor to the inode it just published and writes
// through it; a late publisher replacing that inode would unlink it, and the
// acknowledged write would vanish when the descriptor closes. Each round
// forks two writers that open the same never-touched lower file for writing
// almost simultaneously — the second delayed a moment so the first has
// published, opened, and written while the second's staging copy is still in
// flight — and each writes its own byte. Both bytes must then be readable by
// pathname, whichever copy-up won. The lower files are made large so the
// staging copy dominates the open, keeping both writers inside copy-up at
// once. Natively both writers share one inode and the assertion is trivial.

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define ROUNDS 5
#define FILE_MB 32

static void path_of(char *out, size_t n, const char *dir, int round) {
    snprintf(out, n, "%s/f%d", dir, round);
}

static int prep(const char *dir) {
    if (mkdir(dir, 0755) != 0) return 11;
    static char chunk[1 << 20];
    memset(chunk, 'x', sizeof(chunk));
    for (int r = 0; r < ROUNDS; r++) {
        char path[4096];
        path_of(path, sizeof(path), dir, r);
        int fd = open(path, O_CREAT | O_EXCL | O_WRONLY, 0644);
        if (fd < 0) return 12;
        for (int i = 0; i < FILE_MB; i++)
            if (write(fd, chunk, sizeof(chunk)) != sizeof(chunk)) return 13;
        if (close(fd) != 0) return 14;
    }
    return 0;
}

static int writer(const char *path, int go_fd, useconds_t delay, char byte,
                  off_t off) {
    char t;
    if (read(go_fd, &t, 1) != 1) return 51;
    if (delay) usleep(delay);
    int fd = open(path, O_RDWR);
    if (fd < 0) return 52;
    if (pwrite(fd, &byte, 1, off) != 1) return 53;
    return close(fd) == 0 ? 0 : 54;
}

static int race(const char *dir) {
    for (int r = 0; r < ROUNDS; r++) {
        char path[4096];
        path_of(path, sizeof(path), dir, r);

        int go[2];
        if (pipe(go) != 0) return 21;
        pid_t a = fork();
        if (a < 0) return 22;
        if (a == 0) _exit(writer(path, go[0], 0, 'A', 0));
        pid_t b = fork();
        if (b < 0) return 23;
        if (b == 0) _exit(writer(path, go[0], 2000, 'B', 1));

        if (write(go[1], "gg", 2) != 2) return 24;
        int status;
        if (waitpid(a, &status, 0) != a) return 25;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            return WIFEXITED(status) ? WEXITSTATUS(status) : 26;
        if (waitpid(b, &status, 0) != b) return 27;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            return WIFEXITED(status) ? WEXITSTATUS(status) : 28;
        close(go[0]);
        close(go[1]);

        char got[2] = {0, 0};
        int fd = open(path, O_RDONLY);
        if (fd < 0) return 29;
        if (pread(fd, got, 2, 0) != 2) return 30;
        close(fd);
        if (got[0] != 'A' || got[1] != 'B') {
            fprintf(stderr, "round %d: expected \"AB\", got \"%c%c\"\n", r,
                    got[0], got[1]);
            return 31;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) return 10;
    if (strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (strcmp(argv[1], "race") == 0) return race(argv[2]);
    return 10;
}
