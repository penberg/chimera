// RUN: %cc %s -o %t && rm -rf %t.a %t.b %t.c && %t prep %t.a %t.b %t.c
// RUN: %runner %t check %t.a %t.b %t.c
// UNSUPPORTED: darwin -- the workspace overlay is a Linux-only feature (xattrs, sysmacros.h, st_atim)
//
// Scaffold directories are structure, not content: creating a file beneath a
// lower directory materializes upper parents, but those exist only so the
// child has a path. The directory's visible identity — mode (the sticky bit
// here), owner, inode, device — must keep coming from the lower directory it
// merges with. A mutation aimed at the directory itself is different: chmod,
// utimensat, or an fd-level fchmod must become visible, and must preserve
// every attribute it did not change (a utimens must not reset the sticky
// 1777 to a umask default; a later utimens must not undo an earlier fchmod).
// Natively all of this is ordinary Linux behavior.

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int prep(char **dirs) {
    for (int i = 0; i < 3; i++) {
        if (mkdir(dirs[i], 0755) != 0) return 11 + i;
        if (chmod(dirs[i], 01777) != 0) return 14 + i;
    }
    return 0;
}

static int make_child(const char *dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/child", dir);
    int fd = open(path, O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd < 0) return -1;
    return close(fd);
}

static int check(char **dirs) {
    const char *a = dirs[0], *b = dirs[1], *c = dirs[2];
    struct stat before, after;

    // A: an upper child appears; the directory's identity must not move.
    if (stat(a, &before) != 0) return 21;
    if ((before.st_mode & 07777) != 01777) return 22;
    if (make_child(a) != 0) return 23;
    if (stat(a, &after) != 0) return 24;
    if ((after.st_mode & 07777) != 01777) return 25;
    if (after.st_ino != before.st_ino || after.st_dev != before.st_dev)
        return 26;
    if (after.st_uid != before.st_uid || after.st_gid != before.st_gid)
        return 27;
    int fd = open(a, O_RDONLY | O_DIRECTORY);
    if (fd < 0) return 28;
    if (fstat(fd, &after) != 0) return 29;
    close(fd);
    if ((after.st_mode & 07777) != 01777 || after.st_ino != before.st_ino ||
        after.st_dev != before.st_dev)
        return 30;

    // B (still scaffolded): chmod aimed at the directory becomes visible.
    if (chmod(a, 0750) != 0) return 31;
    if (stat(a, &after) != 0) return 32;
    if ((after.st_mode & 07777) != 0750) return 33;

    // C (never touched): utimensat must set the times and nothing else —
    // the sticky 1777 must survive.
    struct timespec times[2] = {{55555, 0}, {66666, 0}};
    if (utimensat(AT_FDCWD, b, times, 0) != 0) return 41;
    if (stat(b, &after) != 0) return 42;
    if ((after.st_mode & 07777) != 01777) return 43;
    if (after.st_atim.tv_sec != 55555 || after.st_mtim.tv_sec != 66666)
        return 44;

    // D: fd-level fchmod on a scaffolded directory becomes visible, and a
    // later utimens preserves it rather than re-mirroring the lower mode.
    if (make_child(c) != 0) return 51;
    fd = open(c, O_RDONLY | O_DIRECTORY);
    if (fd < 0) return 52;
    if (fchmod(fd, 0700) != 0) return 53;
    close(fd);
    if (stat(c, &after) != 0) return 54;
    if ((after.st_mode & 07777) != 0700) return 55;
    struct timespec later[2] = {{77777, 0}, {88888, 0}};
    if (utimensat(AT_FDCWD, c, later, 0) != 0) return 56;
    if (stat(c, &after) != 0) return 57;
    if ((after.st_mode & 07777) != 0700) return 58;
    if (after.st_mtim.tv_sec != 88888) return 59;

    // The merged contents were never disturbed by any of the claims.
    char path[4096];
    snprintf(path, sizeof(path), "%s/child", c);
    if (stat(path, &after) != 0) return 61;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 5) return 10;
    if (strcmp(argv[1], "prep") == 0) return prep(&argv[2]);
    if (strcmp(argv[1], "check") == 0) return check(&argv[2]);
    return 10;
}
