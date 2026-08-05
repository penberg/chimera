// RUN: %cc %s -o %t && rm -rf %t.probeA %t.probeB %t.childA %t.childB %t.ws2
// RUN: %runner %t scenario %t.probeA > %t.childA
// RUN: %t await %t.childA
// RUN: %t verify-kept "$XDG_STATE_HOME/chimera/fs" %t.probeA
// RUN: case "%runner" in *--unsafe*) : ;; *chimera*) CHIMERA_FS=%t.ws2 %runner --rm %t scenario %t.probeB > %t.childB && %t await %t.childB && %t verify-rm %t.ws2 %t.probeB ;; *) : ;; esac
//
// The session lifetime is the guest process tree, not the root command: a
// forked guest that outlives the session root must keep a usable filesystem.
// The scenario forks a child that waits (by pipe EOF) until its parent — the
// session root — has fully exited, then creates and reads back an upper
// file, reporting through the inherited host stdout. Disposal decisions are
// deferred to the last process out: the fresh-and-still-empty filesystem of
// the first run must not be garbage-collected at root exit (the child's
// write lands in it and keeps it), and the --rm of the second run must not
// remove the filesystem before the child is done — but must remove it after.
// Natively the child simply writes to disk after its parent exits.

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static int scenario(const char *probe) {
    int p[2];
    if (pipe(p) != 0) return 21;
    pid_t pid = fork();
    if (pid < 0) return 22;
    if (pid == 0) {
        close(p[1]);
        char t;
        // EOF: the parent, the only writer, is completely gone.
        if (read(p[0], &t, 1) != 0) _exit(31);
        int fd = open(probe, O_CREAT | O_EXCL | O_RDWR, 0644);
        if (fd < 0) {
            dprintf(1, "child-fail-open\n");
            _exit(32);
        }
        if (write(fd, "held", 4) != 4) _exit(33);
        char buf[8] = {0};
        if (pread(fd, buf, 4, 0) != 4 || memcmp(buf, "held", 4) != 0)
            _exit(34);
        if (close(fd) != 0) _exit(35);
        dprintf(1, "child-ok\n");
        _exit(0);
    }
    // The root exits immediately; process exit closes the pipe's write end
    // only after the CLI has run its end-of-session disposition.
    return 0;
}

static int await(const char *report) {
    for (int i = 0; i < 100; i++) {
        char buf[64] = {0};
        FILE *f = fopen(report, "r");
        if (f) {
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            fclose(f);
            (void)n;
            if (strstr(buf, "child-ok")) return 0;
            if (strstr(buf, "child-fail")) return 41;
        }
        usleep(100 * 1000);
    }
    return 42;
}

// Overlay runs keep the (now non-empty) filesystem: some filesystem under the
// state directory holds the probe, and the host path stays absent. Native
// and --unsafe runs wrote straight to disk.
static int verify_kept(const char *statedir, const char *probe) {
    char path[4096];
    struct stat st;
    DIR *dir = opendir(statedir);
    if (dir) {
        struct dirent *e;
        while ((e = readdir(dir))) {
            if (e->d_name[0] == '.') continue;
            snprintf(path, sizeof(path), "%s/%s/data%s", statedir, e->d_name,
                     probe);
            if (stat(path, &st) == 0) {
                closedir(dir);
                return stat(probe, &st) == 0 ? 51 : 0;
            }
        }
        closedir(dir);
    }
    return stat(probe, &st) == 0 ? 0 : 52;
}

static int verify_rm(const char *ws, const char *probe) {
    struct stat st;
    if (stat(ws, &st) == 0) return 61;
    if (stat(probe, &st) == 0) return 62;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "scenario") == 0) return scenario(argv[2]);
    if (argc == 3 && strcmp(argv[1], "await") == 0) return await(argv[2]);
    if (argc == 4 && strcmp(argv[1], "verify-kept") == 0)
        return verify_kept(argv[2], argv[3]);
    if (argc == 4 && strcmp(argv[1], "verify-rm") == 0)
        return verify_rm(argv[2], argv[3]);
    return 10;
}
