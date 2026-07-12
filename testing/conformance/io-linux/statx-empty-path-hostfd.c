// RUN: %cc %s -o %t && %runner %t
//
// The AT_EMPTY_PATH fd form must work on a descriptor that never went
// through a path resolution — a pipe end or a memfd, the shape of an
// inherited or SCM_RIGHTS-passed descriptor. Rust's `File::metadata` is
// exactly `statx(fd, "", AT_EMPTY_PATH)`; glycin's sandboxed loader calls it
// on the image fd its parent passed over a socketpair, so a sandbox that
// answers EBADF here kills every GNOME image load.

#define _GNU_SOURCE

#include <fcntl.h>
#include <linux/stat.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

int main(void) {
    int mfd = memfd_create("statx-probe", 0);
    if (mfd < 0) return 1;
    if (ftruncate(mfd, 4096) != 0) return 2;

    struct statx sx;
    if (syscall(SYS_statx, mfd, "", AT_EMPTY_PATH, STATX_BASIC_STATS, &sx) != 0) return 3;
    if (!S_ISREG(sx.stx_mode) || sx.stx_size != 4096) return 4;

    struct stat st;
    if (syscall(SYS_newfstatat, mfd, "", &st, AT_EMPTY_PATH) != 0) return 5;
    if (!S_ISREG(st.st_mode) || st.st_size != 4096) return 6;

    int pfd[2];
    if (pipe(pfd) != 0) return 7;
    if (syscall(SYS_statx, pfd[0], "", AT_EMPTY_PATH, STATX_BASIC_STATS, &sx) != 0) return 8;
    if (!S_ISFIFO(sx.stx_mode)) return 9;
    return 0;
}
