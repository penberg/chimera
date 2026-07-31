// RUN: %cc %s -o %t && %cc -DNEWVER %s -o %t.new && rm -rf %t.d %t.ws && %t prep %t.d
// RUN: CHIMERA_FS=%t.ws %runner %t rewrite %t.new %t.d/prog
// RUN: CHIMERA_FS=%t.ws %runner %t.d/prog; test $? -eq 77
// RUN: CHIMERA_FS=%t.ws %runner %t.d/script; test $? -eq 77
// RUN: CHIMERA_FS=%t.ws %runner %t unlink %t.d/prog
// RUN: CHIMERA_FS=%t.ws %runner %t.d/prog; c=$?; test $c -ne 0 && test $c -ne 11
//
// Initial-exec coherence: the executable that starts a session must come
// from the same merged view the guest's own syscalls see. One session
// rewrites `prog` (old version exits 11, new version 77); a second session
// attached to the same filesystem then runs `prog` as its *initial*
// executable and must get 77 — the filesystem's copy, not the stale host
// bytes. The same holds one level down: `script`'s shebang names `prog`, so
// the interpreter lookup must resolve through the filesystem too. Finally a
// session deletes `prog`; launching it afterward must fail rather than
// resurrect the lower file (and in particular must not run the old version,
// exit 11). Natively the rewrite and unlink hit the host file itself, so
// every assertion holds unchanged.

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef NEWVER
#define VERSION_EXIT 77
#else
#define VERSION_EXIT 11
#endif

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

static int prep(const char *dir) {
    char path[4096], line[4096];
    if (mkdir(dir, 0755) != 0) return 21;
    snprintf(path, sizeof(path), "%s/prog", dir);
    if (copy_file("/proc/self/exe", path, 0755) != 0) return 22;
    int n = snprintf(line, sizeof(line), "#!%s\n", path);
    snprintf(path, sizeof(path), "%s/script", dir);
    int fd = open(path, O_CREAT | O_EXCL | O_WRONLY, 0755);
    if (fd < 0) return 23;
    if (write(fd, line, n) != n) return 24;
    return close(fd) == 0 ? 0 : 25;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (argc == 4 && strcmp(argv[1], "rewrite") == 0)
        return copy_file(argv[2], argv[3], 0755) == 0 ? 0 : 31;
    if (argc == 3 && strcmp(argv[1], "unlink") == 0)
        return unlink(argv[2]) == 0 ? 0 : 41;
    // Any other invocation — including as `script`'s interpreter — reports
    // which version of the binary actually ran.
    return VERSION_EXIT;
}
