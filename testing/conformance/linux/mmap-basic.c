// RUN: %cc %s -o %t && %runner %t
//
// Basic anonymous mmap/munmap flow. This exercises the runtime-owned mmap path
// and verifies that the returned mapping is writable and survives ordinary
// guest-side use.

#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 1;

    size_t len = (size_t) page_size * 2;
    uint8_t *p = mmap(0, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 2;

    p[0] = 0x12;
    p[page_size - 1] = 0x34;
    p[page_size] = 0x56;
    p[len - 1] = 0x78;

    if (p[0] != 0x12) return 3;
    if (p[page_size - 1] != 0x34) return 4;
    if (p[page_size] != 0x56) return 5;
    if (p[len - 1] != 0x78) return 6;

    if (munmap(p, len) != 0) return 7;
    return 0;
}
