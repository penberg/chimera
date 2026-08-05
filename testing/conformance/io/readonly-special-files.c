// RUN: %cc %s -o %t && %runner %t
//
// The read-only mount restricts the filesystem, not the devices on it, the
// way a kernel `mount -o ro` does: a write-open of a special file sends bytes
// to the object behind the node, mutating no filesystem state, so it must
// succeed — /dev/null must accept writes, a pty must be allocatable and its
// ioctls (grantpt/unlockpt, isatty's TCGETS) must reach the real device.
// Exits 0 only if both hold. (Native runs pass trivially: the assertions are
// plain Linux semantics, which is the point — the sandbox must not change
// them.)

#define _XOPEN_SOURCE 700

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    // Writing a character device on the read-only mount works.
    int null = open("/dev/null", O_WRONLY);
    if (null < 0) return 1;
    if (write(null, "x", 1) != 1) return 2;
    close(null);

    // A pty allocates: the O_RDWR open of /dev/ptmx and the ioctls behind
    // grantpt/unlockpt and isatty all reach the device.
    int master = posix_openpt(O_RDWR | O_NOCTTY);
    if (master < 0) return 3;
    if (grantpt(master) != 0 || unlockpt(master) != 0) return 4;
    if (!isatty(master)) return 5;
    close(master);

    return 0;
}
