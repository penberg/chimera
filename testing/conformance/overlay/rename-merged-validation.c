// RUN: %cc %s -o %t && rm -rf %t.d && %t prep %t.d
// RUN: timeout 20 %runner %t check %t.d
//
// Rename preconditions are merged-view questions. A directory renamed over a
// lower regular file must fail ENOTDIR and over a non-empty lower directory
// ENOTEMPTY, even though the physical upper destination is absent or an
// empty scaffold; a file over a merged directory is EISDIR; NOREPLACE sees
// the merged destination. And the merged view cuts the other way too: a name
// deleted by the guest is free, so a rename onto a whiteout must succeed —
// plain, with NOREPLACE, and with a directory source — and a directory
// destination whose remaining children are all deleted is empty, so
// replacing it must succeed with none of the deleted lower children ever
// reappearing inside the replacement. Natively every one of these is
// ordinary Linux behavior.

#define _GNU_SOURCE

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

static int write_file(const char *path, const char *content) {
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) return -1;
    ssize_t n = (ssize_t)strlen(content);
    if (write(fd, content, n) != n) return -1;
    return close(fd);
}

static int rename2(const char *from, const char *to, unsigned flags) {
    return (int)syscall(SYS_renameat2, AT_FDCWD, from, AT_FDCWD, to, flags);
}

static int prep(const char *d) {
    char p[4096];
    if (mkdir(d, 0755) != 0) return 11;
    snprintf(p, sizeof(p), "%s/file1", d);
    if (write_file(p, "one") != 0) return 12;
    snprintf(p, sizeof(p), "%s/file2", d);
    if (write_file(p, "two") != 0) return 13;
    snprintf(p, sizeof(p), "%s/emptyd", d);
    if (mkdir(p, 0755) != 0) return 14;
    snprintf(p, sizeof(p), "%s/fulld", d);
    if (mkdir(p, 0755) != 0) return 15;
    snprintf(p, sizeof(p), "%s/fulld/child", d);
    if (write_file(p, "c") != 0) return 16;
    snprintf(p, sizeof(p), "%s/wfile", d);
    if (write_file(p, "w") != 0) return 17;
    snprintf(p, sizeof(p), "%s/wfile2", d);
    if (write_file(p, "w2") != 0) return 18;
    snprintf(p, sizeof(p), "%s/fulld2", d);
    if (mkdir(p, 0755) != 0) return 19;
    snprintf(p, sizeof(p), "%s/fulld2/child", d);
    if (write_file(p, "c2") != 0) return 20;
    snprintf(p, sizeof(p), "%s/xfile", d);
    if (write_file(p, "swapme") != 0) return 21;
    return 0;
}

static int count_entries(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) return -1;
    int n = 0;
    struct dirent *e;
    while ((e = readdir(dir))) {
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0)
            continue;
        n++;
    }
    closedir(dir);
    return n;
}

static int check(const char *d) {
    char src[4096], dst[4096], p[4096];
    struct stat st;

    // Fresh directories (upper-only under the overlay) as rename sources.
    for (int i = 1; i <= 4; i++) {
        snprintf(src, sizeof(src), "%s/src%d", d, i);
        if (mkdir(src, 0755) != 0) return 30 + i;
    }
    snprintf(p, sizeof(p), "%s/src3/inner", d);
    if (write_file(p, "moved") != 0) return 35;

    // A directory over a lower regular file: ENOTDIR.
    snprintf(src, sizeof(src), "%s/src1", d);
    snprintf(dst, sizeof(dst), "%s/file1", d);
    if (rename(src, dst) == 0 || errno != ENOTDIR) return 41;

    // A directory over a non-empty lower directory: ENOTEMPTY.
    snprintf(dst, sizeof(dst), "%s/fulld", d);
    if (rename(src, dst) == 0 || (errno != ENOTEMPTY && errno != EEXIST))
        return 42;

    // A file over a merged directory: EISDIR.
    snprintf(src, sizeof(src), "%s/file1", d);
    snprintf(dst, sizeof(dst), "%s/emptyd", d);
    if (rename(src, dst) == 0 || errno != EISDIR) return 43;

    // NOREPLACE judges the merged destination.
    snprintf(dst, sizeof(dst), "%s/file2", d);
    if (rename2(src, dst, RENAME_NOREPLACE) == 0 || errno != EEXIST)
        return 44;

    // A directory over an empty lower directory: lands, contents intact.
    snprintf(src, sizeof(src), "%s/src3", d);
    snprintf(dst, sizeof(dst), "%s/emptyd", d);
    if (rename(src, dst) != 0) return 45;
    snprintf(p, sizeof(p), "%s/emptyd/inner", d);
    if (stat(p, &st) != 0) return 46;

    // A whiteout frees the name: plain rename of a directory onto it...
    snprintf(p, sizeof(p), "%s/wfile", d);
    if (unlink(p) != 0) return 47;
    snprintf(src, sizeof(src), "%s/src1", d);
    if (rename(src, p) != 0) return 48;
    if (stat(p, &st) != 0 || !S_ISDIR(st.st_mode)) return 49;

    // ...and a NOREPLACE rename of a file onto it both succeed.
    snprintf(p, sizeof(p), "%s/wfile2", d);
    if (unlink(p) != 0) return 50;
    snprintf(src, sizeof(src), "%s/newf", d);
    if (write_file(src, "fresh") != 0) return 51;
    if (rename2(src, p, RENAME_NOREPLACE) != 0) return 52;

    // Deleting a directory's last child makes it merged-empty: replacing it
    // succeeds and the deleted child never reappears in the replacement.
    snprintf(p, sizeof(p), "%s/fulld2/child", d);
    if (unlink(p) != 0) return 53;
    snprintf(src, sizeof(src), "%s/src2", d);
    snprintf(dst, sizeof(dst), "%s/fulld2", d);
    if (rename(src, dst) != 0) return 54;
    if (count_entries(dst) != 0) return 55;

    // EXCHANGE swaps a directory with a lower file.
    snprintf(src, sizeof(src), "%s/src4", d);
    snprintf(dst, sizeof(dst), "%s/xfile", d);
    if (rename2(src, dst, RENAME_EXCHANGE) != 0) return 56;
    if (stat(dst, &st) != 0 || !S_ISDIR(st.st_mode)) return 57;
    if (stat(src, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size != 6)
        return 58;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) return 10;
    if (strcmp(argv[1], "prep") == 0) return prep(argv[2]);
    if (strcmp(argv[1], "check") == 0) return check(argv[2]);
    return 10;
}
