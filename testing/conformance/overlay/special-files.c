// RUN: %cc %s -o %t && rm -rf %t.delta %t.fifo && CHIMERA_COW=%t.delta %runner %t check %t.fifo
// RUN: %t verify %t.delta %t.fifo
//
// Special files under the overlay: opening a device passes through to the
// lower without copy-up — writing /dev/null sends bytes to the object behind
// the node, not to any filesystem — while mknod of a FIFO lands in the upper
// and behaves like a real FIFO because it is one. Device-node mknod fails
// exactly as it does for an unprivileged user natively. Every check
// assertion is plain Linux semantics, so the test passes natively too; the
// verify step then proves no /dev copy-up materialized and the FIFO never
// touched the host.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

static int check(const char *fifo) {
    // Writing a character device works and reaches the real device.
    int null = open("/dev/null", O_WRONLY);
    if (null < 0) return 31;
    if (write(null, "sink", 4) != 4) return 32;
    struct stat st;
    if (fstat(null, &st) != 0 || !S_ISCHR(st.st_mode)) return 33;
    if (major(st.st_rdev) != 1 || minor(st.st_rdev) != 3) return 34;
    close(null);

    // Reading /dev/zero produces zeros — the node, not a copied file.
    int zero = open("/dev/zero", O_RDONLY);
    if (zero < 0) return 35;
    char buf[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    if (read(zero, buf, sizeof(buf)) != sizeof(buf)) return 36;
    for (unsigned i = 0; i < sizeof(buf); i++)
        if (buf[i] != 0) return 37;
    close(zero);

    // A FIFO the guest creates is a real FIFO: bytes written come back.
    if (mkfifo(fifo, 0644) != 0) return 38;
    if (stat(fifo, &st) != 0 || !S_ISFIFO(st.st_mode)) return 39;
    int rw = open(fifo, O_RDWR); // O_RDWR so one process can do both ends
    if (rw < 0) return 40;
    if (write(rw, "ping", 4) != 4) return 41;
    if (read(rw, buf, 4) != 4 || memcmp(buf, "ping", 4) != 0) return 42;
    close(rw);

    // An unprivileged mknod of a device node fails the same way natively.
    char dev[4096];
    snprintf(dev, sizeof(dev), "%s.dev", fifo);
    if (mknod(dev, S_IFCHR | 0644, makedev(1, 3)) == 0 || errno != EPERM)
        return 43;
    return 0;
}

static int verify(const char *delta, const char *fifo) {
    char probe[4096];
    struct stat st;
    snprintf(probe, sizeof(probe), "%s/data", delta);
    if (stat(probe, &st) != 0) return 0; // native run: nothing to prove

    // No device was copied up…
    snprintf(probe, sizeof(probe), "%s/data/dev", delta);
    if (stat(probe, &st) == 0) return 51;
    // …and the guest's FIFO lives in the delta, not on the host.
    if (stat(fifo, &st) == 0) return 52;
    snprintf(probe, sizeof(probe), "%s/data%s", delta, fifo);
    if (stat(probe, &st) != 0 || !S_ISFIFO(st.st_mode)) return 53;
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "check") == 0) return check(argv[2]);
    if (argc == 4 && strcmp(argv[1], "verify") == 0)
        return verify(argv[2], argv[3]);
    return 10;
}
