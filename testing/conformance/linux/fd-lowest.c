// RUN: %cc %s -o %t && %runner %t
// UNSUPPORTED: darwin -- Linux-only host interfaces
//
// POSIX gives every new descriptor the lowest available number, and real
// programs depend on it: bash indexes a per-fd array by the number (a sparse
// descriptor like 0x40000000 once made it allocate and zero 8 GiB per
// redirect), and FD_SET corrupts memory beyond 1023. A sandbox that
// virtualizes descriptors must keep them in the same low, dense namespace.
// Exits 0 only if open/dup/F_DUPFD/dup2 all number the way the kernel does.

#include <fcntl.h>
#include <sys/select.h>
#include <unistd.h>

int main(void) {
    int a = open("/etc/hostname", O_RDONLY);
    if (a < 0 || a >= FD_SETSIZE) return 1;

    // Consecutive opens take consecutive numbers.
    int b = open("/etc/hostname", O_RDONLY);
    if (b != a + 1) return 2;

    // A closed number is the next one handed out.
    close(a);
    int c = open("/etc/hostname", O_RDONLY);
    if (c != a) return 3;

    // dup takes the lowest free number; F_DUPFD honors its floor.
    int d = dup(b);
    if (d != b + 1) return 4;
    int e = fcntl(b, F_DUPFD, 100);
    if (e != 100) return 5;

    // dup2 lands exactly on its target, and reads through it work.
    int f = dup2(b, 200);
    if (f != 200) return 6;
    char buf[4];
    if (read(f, buf, sizeof(buf)) != sizeof(buf)) return 7;

    return 0;
}
