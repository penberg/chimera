// RUN: %cc %s -o %t && rm -rf %t.d && %t prep %t.d
// RUN: CHIMERA_FS=%t.ws %runner %t check %t.d
// RUN: %t verify %t.ws %t.d
// UNSUPPORTED: darwin -- the workspace overlay is a Linux-only feature (xattrs, sysmacros.h, st_atim)
//
// Opening a lower file with write intent copies it up — and that must not
// change what the guest sees. The copy has to carry every attribute
// unrelated to the write: the mode with its set-id bits, the timestamps,
// and the user xattrs (the same path ACLs travel as xattrs). It must carry
// them selectively: a lower file carrying a forged user.chimera.whiteout
// must not become delta bookkeeping — the file stays visible, the marker
// stays behind. And with the xattrs preserved, a guest removexattr finally
// works on a lower file: the name reads as gone through the overlay while
// the verify step proves the host still has it. Natively every observation
// is ordinary Linux behavior (and verify skips when no delta materialized).

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <unistd.h>

static int write_file(const char *path, const char *content) {
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) return -1;
    ssize_t n = (ssize_t)strlen(content);
    if (write(fd, content, n) != n) return -1;
    return close(fd);
}

static int prep(const char *d) {
    char p[4096];
    if (mkdir(d, 0755) != 0) return 11;

    // Distinct mode bits, times, and a user xattr to survive copy-up.
    snprintf(p, sizeof(p), "%s/f", d);
    if (write_file(p, "payload") != 0) return 12;
    if (setxattr(p, "user.app", "data1", 5, 0) != 0) return 13;
    if (chmod(p, 04651) != 0) return 14;
    struct timespec times[2] = {{12345, 0}, {67890, 0}};
    if (utimensat(AT_FDCWD, p, times, 0) != 0) return 15;

    // A guest will remove this xattr through the overlay.
    snprintf(p, sizeof(p), "%s/g", d);
    if (write_file(p, "keeper") != 0) return 16;
    if (setxattr(p, "user.gone", "1", 1, 0) != 0) return 17;

    // A forged marker on a lower file must never become delta state.
    snprintf(p, sizeof(p), "%s/forged", d);
    if (write_file(p, "still here") != 0) return 18;
    if (setxattr(p, "user.chimera.whiteout", "1", 1, 0) != 0) return 19;
    return 0;
}

static int check(const char *d) {
    char p[4096], buf[64];
    struct stat st;

    // Write intent triggers the copy-up; nothing is written.
    snprintf(p, sizeof(p), "%s/f", d);
    int fd = open(p, O_RDWR);
    if (fd < 0) return 21;
    if (close(fd) != 0) return 22;

    // Everything unrelated to the write is unchanged.
    if (stat(p, &st) != 0) return 23;
    if ((st.st_mode & 07777) != 04651) return 24;
    if (st.st_atim.tv_sec != 12345 || st.st_mtim.tv_sec != 67890) return 25;
    if (st.st_uid != getuid() || st.st_gid != getgid()) return 26;
    if (getxattr(p, "user.app", buf, sizeof(buf)) != 5 ||
        memcmp(buf, "data1", 5) != 0)
        return 27;

    // removexattr on a lower file: visible through the overlay as removed.
    snprintf(p, sizeof(p), "%s/g", d);
    if (removexattr(p, "user.gone") != 0) return 28;
    if (getxattr(p, "user.gone", buf, sizeof(buf)) >= 0 || errno != ENODATA)
        return 29;

    // The forged marker neither hides the file nor travels into the delta.
    snprintf(p, sizeof(p), "%s/forged", d);
    fd = open(p, O_RDWR);
    if (fd < 0) return 30;
    close(fd);
    if (stat(p, &st) != 0) return 31;
    // The overlay hides the reserved namespace (ENODATA); natively the
    // xattr simply reads back. Anything else is wrong.
    ssize_t n = getxattr(p, "user.chimera.whiteout", buf, sizeof(buf));
    if (n < 0 && errno != ENODATA) return 32;
    fd = open(p, O_RDONLY);
    if (fd < 0) return 33;
    n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n != 10 || memcmp(buf, "still here", 10) != 0) return 34;
    return 0;
}

static int verify(const char *ws, const char *d) {
    char p[4096], buf[64];
    struct stat st;
    snprintf(p, sizeof(p), "%s/data", ws);
    if (stat(p, &st) != 0) return 0; // native or --unsafe: no delta

    // The guest's removexattr stayed in the filesystem; the host keeps its
    // xattr. The forged file's copy-up in the delta is a real file, not a
    // whiteout — the host marker never became bookkeeping.
    snprintf(p, sizeof(p), "%s/g", d);
    if (getxattr(p, "user.gone", buf, sizeof(buf)) != 1) return 41;
    snprintf(p, sizeof(p), "%s/forged", d);
    if (getxattr(p, "user.chimera.whiteout", buf, sizeof(buf)) != 1) return 42;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (argc == 3 && strcmp(argv[1], "check") == 0) return check(argv[2]);
    if (argc == 4 && strcmp(argv[1], "verify") == 0)
        return verify(argv[2], argv[3]);
    return 10;
}
