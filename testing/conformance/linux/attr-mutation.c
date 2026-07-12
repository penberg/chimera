// RUN: %cc %s -o %t && %runner %t
//
// Attribute mutators must route through the VFS. The test discovers which
// world it runs in by trying to create a scratch file: in a writable world
// (native, or a read-write sandbox) every mutator succeeds and stat observes
// the change; on Chimera's read-only mount the create fails with EROFS and
// every mutator must then return EROFS too, leaving the file untouched — a
// mutator slipping past the VFS to the host is exactly the bug this guards
// against.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/xattr.h>
#include <unistd.h>

#ifndef SYS_fchmodat2
#define SYS_fchmodat2 452
#endif

static char scratch[4096];

// Argument decoding must be answered by Chimera itself, identically in both
// worlds, before any mount policy applies.
static int rejected_arguments(const char *self) {
    struct stat before, after;
    if (stat(self, &before) != 0) return 1;

    if (chmod("", 0600) == 0 || errno != ENOENT) return 2;
    if (syscall(SYS_chmod, NULL, 0600) == 0 || errno != EFAULT) return 3;
    // AT_REMOVEDIR is not a chmod flag; a pre-6.6 native kernel answers the
    // unknown syscall with ENOSYS instead.
    errno = 0;
    if (syscall(SYS_fchmodat2, AT_FDCWD, self, 0600, AT_REMOVEDIR) == 0 ||
        (errno != EINVAL && errno != ENOSYS))
        return 4;

    if (chown("", (uid_t)-1, (gid_t)-1) == 0 || errno != ENOENT) return 5;
    if (syscall(SYS_fchownat, AT_FDCWD, self, -1, -1, AT_REMOVEDIR) == 0 ||
        errno != EINVAL)
        return 6;

    if (utimes("", NULL) == 0 || errno != ENOENT) return 7;
    if (syscall(SYS_utimensat, AT_FDCWD, self, NULL, AT_REMOVEDIR) == 0 ||
        errno != EINVAL)
        return 8;
    struct timeval invalid_tv[2] = {{0, 1000000}, {0, 0}};
    if (syscall(SYS_utimes, self, invalid_tv) == 0 || errno != EINVAL) return 9;
    // A NULL path against AT_FDCWD is a bad address, not the futimesat fd
    // form: utimes(NULL, tv) answers EFAULT, never EBADF.
    if (utimes(NULL, invalid_tv) == 0 || errno != EFAULT) return 94;

    if (mknod("", S_IFIFO | 0644, 0) == 0 || errno != ENOENT) return 10;

    if (setxattr("", "user.chimera-test", "v", 1, 0) == 0 || errno != ENOENT)
        return 93;

    if (stat(self, &after) != 0) return 91;
    if (before.st_mode != after.st_mode) return 92;
    return 0;
}

static int read_only_world(const char *self) {
    struct stat before, after;
    if (stat(self, &before) != 0) return 11;

    // Path forms: each must be denied with EROFS, not forwarded.
    if (chmod(self, 0600) == 0 || errno != EROFS) return 12;
    if (chown(self, (uid_t)-1, (gid_t)-1) == 0 || errno != EROFS) return 13;
    if (lchown(self, (uid_t)-1, (gid_t)-1) == 0 || errno != EROFS) return 14;
    struct timespec ts[2] = {{12345, 0}, {54321, 0}};
    if (utimensat(AT_FDCWD, self, ts, 0) == 0 || errno != EROFS) return 15;
    if (mknod(scratch, S_IFIFO | 0644, 0) == 0 || errno != EROFS) return 23;
    if (setxattr(self, "user.chimera-test", "v", 1, 0) == 0 || errno != EROFS)
        return 24;
    if (removexattr(self, "user.chimera-test") == 0 || errno != EROFS)
        return 26;

    // Fd forms on a read-only open of the same file.
    int fd = open(self, O_RDONLY);
    if (fd < 0) return 16;
    if (fchmod(fd, 0600) == 0 || errno != EROFS) return 17;
    if (fchown(fd, (uid_t)-1, (gid_t)-1) == 0 || errno != EROFS) return 18;
    if (futimens(fd, ts) == 0 || errno != EROFS) return 19;
    if (fsetxattr(fd, "user.chimera-test", "v", 1, 0) == 0 || errno != EROFS)
        return 25;
    if (fremovexattr(fd, "user.chimera-test") == 0 || errno != EROFS)
        return 27;
    if (fallocate(fd, 0, 0, 1) == 0 || errno != EROFS) return 28;
    close(fd);

    // Nothing leaked through to the host.
    if (stat(self, &after) != 0) return 20;
    if (after.st_mode != before.st_mode) return 21;
    if (after.st_mtim.tv_sec != before.st_mtim.tv_sec) return 22;
    if (after.st_size != before.st_size) return 29;
    return 0;
}

static int writable_world(int fd) {
    struct stat st;

    // Fd forms, observed through fstat on the same descriptor.
    if (fchmod(fd, 0640) != 0) return 31;
    if (fstat(fd, &st) != 0 || (st.st_mode & 07777) != 0640) return 32;
    if (fchown(fd, (uid_t)-1, (gid_t)-1) != 0) return 33;
    struct timespec fts[2] = {{111, 0}, {222, 0}};
    if (futimens(fd, fts) != 0) return 51;
    if (fstat(fd, &st) != 0 || st.st_mtim.tv_sec != 222) return 52;
    // UTIME_OMIT holds one timestamp while the other moves.
    struct timespec omit[2] = {{0, UTIME_NOW}, {0, UTIME_OMIT}};
    if (futimens(fd, omit) != 0) return 53;
    if (fstat(fd, &st) != 0 || st.st_mtim.tv_sec != 222) return 54;

    // fallocate manipulates file space through the descriptor; punching a
    // hole with KEEP_SIZE leaves the size alone (EOPNOTSUPP: a filesystem
    // without hole punching).
    if (fallocate(fd, 0, 0, 8192) != 0) return 70;
    if (fstat(fd, &st) != 0 || st.st_size != 8192) return 71;
    errno = 0;
    if (fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, 0, 4096) !=
            0 &&
        errno != EOPNOTSUPP)
        return 72;
    if (fstat(fd, &st) != 0 || st.st_size != 8192) return 73;

    // Path forms, observed through stat by path.
    if (chmod(scratch, 0600) != 0) return 34;
    if (stat(scratch, &st) != 0 || (st.st_mode & 07777) != 0600) return 35;
    if (chown(scratch, (uid_t)-1, (gid_t)-1) != 0) return 36;
    struct timespec ts[2] = {{12345, 0}, {54321, 0}};
    if (utimensat(AT_FDCWD, scratch, ts, 0) != 0) return 55;
    if (stat(scratch, &st) != 0 || st.st_atim.tv_sec != 12345 ||
        st.st_mtim.tv_sec != 54321)
        return 56;

    // No-follow chmod of a symlink answers EOPNOTSUPP the way the kernel
    // does, leaving the target alone (ENOSYS: pre-6.6 native kernel).
    char link[sizeof(scratch) + 8];
    snprintf(link, sizeof(link), "%s.link", scratch);
    unlink(link);
    if (symlink(scratch, link) != 0) return 37;
    errno = 0;
    if (syscall(SYS_fchmodat2, AT_FDCWD, link, 0400, AT_SYMLINK_NOFOLLOW) ==
            0 ||
        (errno != EOPNOTSUPP && errno != ENOSYS))
        return 38;
    if (stat(scratch, &st) != 0 || (st.st_mode & 07777) != 0600) return 39;
    unlink(link);

    // The l-variants name a final symlink itself, so they work even on a
    // dangling one.
    char dangling[sizeof(scratch) + 12];
    snprintf(dangling, sizeof(dangling), "%s.dangling", scratch);
    unlink(dangling);
    if (symlink("nope", dangling) != 0) return 40;
    if (lchown(dangling, (uid_t)-1, (gid_t)-1) != 0) return 41;

    // Xattrs roundtrip where the filesystem supports them. The user
    // namespace is denied on a symlink with EPERM — not ENOENT, which
    // proves the no-follow resolution reached the dangling link itself.
    if (setxattr(scratch, "user.chimera-test", "v", 1, 0) == 0) {
        char buf[8];
        if (getxattr(scratch, "user.chimera-test", buf, sizeof(buf)) != 1 ||
            buf[0] != 'v')
            return 60;
        if (fsetxattr(fd, "user.chimera-test", "w", 1, 0) != 0) return 61;
        if (getxattr(scratch, "user.chimera-test", buf, sizeof(buf)) != 1 ||
            buf[0] != 'w')
            return 62;
        if (lsetxattr(dangling, "user.chimera-test", "v", 1, 0) == 0 ||
            errno != EPERM)
            return 63;
        if (removexattr(scratch, "user.chimera-test") != 0) return 65;
        if (removexattr(scratch, "user.chimera-test") == 0 ||
            errno != ENODATA)
            return 66;
        if (fsetxattr(fd, "user.chimera-test", "x", 1, 0) != 0) return 67;
        if (fremovexattr(fd, "user.chimera-test") != 0) return 68;
    } else if (errno != ENOTSUP) {
        return 64;
    }

    // mknod creates a FIFO the filesystem then reports as one; through a
    // trailing symlink — even a dangling one — it answers EEXIST instead.
    char fifo[sizeof(scratch) + 8];
    snprintf(fifo, sizeof(fifo), "%s.fifo", scratch);
    unlink(fifo);
    if (mknod(fifo, S_IFIFO | 0644, 0) != 0) return 57;
    if (stat(fifo, &st) != 0 || !S_ISFIFO(st.st_mode)) return 58;
    unlink(fifo);
    if (mknod(dangling, S_IFIFO | 0644, 0) == 0 || errno != EEXIST) return 59;
    unlink(dangling);

    close(fd);
    unlink(scratch);
    return 0;
}

int main(int argc, char **argv) {
    (void)argc;
    int rejected = rejected_arguments(argv[0]);
    if (rejected != 0) return rejected;
    snprintf(scratch, sizeof(scratch), "%s.scratch", argv[0]);
    int fd = open(scratch, O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
        if (errno != EROFS) return 10;
        return read_only_world(argv[0]);
    }
    return writable_world(fd);
}
