// RUN: %cc %s -o %t && %runner %t
// UNSUPPORTED: darwin -- Linux-only host interfaces
//
// close_range is how spawn paths scrub "every fd from N up" before an
// exec — Bun marks 3.. CLOSE_RANGE_CLOEXEC before every subprocess. The
// guest's own descriptors in the range must be closed (or marked), and —
// under a sandbox that keeps runtime descriptors in the same table — the
// sweep must not destroy the runtime's own files: opening and reading a
// file must still work afterwards. Exits 0 only if both hold.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/close_range.h>
#include <sys/syscall.h>
#include <unistd.h>

int main(void) {
    int a = open("/etc/hostname", O_RDONLY);
    if (a < 0) return 1;

    // CLOEXEC-mark only: the fd stays open and readable.
    if (syscall(SYS_close_range, 3, ~0U, CLOSE_RANGE_CLOEXEC) != 0) return 2;
    char c;
    if (read(a, &c, 1) != 1) return 3;

    // Plain close: the fd dies...
    if (syscall(SYS_close_range, 3, ~0U, 0) != 0) return 4;
    errno = 0;
    if (read(a, &c, 1) >= 0 || errno != EBADF) return 5;

    // ...but the world keeps working: a fresh open still resolves and reads.
    int b = open("/etc/hostname", O_RDONLY);
    if (b < 0) return 6;
    if (read(b, &c, 1) != 1) return 7;

    return 0;
}
