// RUN: %cc %s -o %t && %runner %t
//
// Partially unmap the middle page of a three-page mapping, then map a fresh
// page back at the same address. The surrounding pages must retain their
// contents and the hole must be reusable.

#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 1;

    size_t len = (size_t) page_size * 3;
    uint8_t *base = mmap(0, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return 2;

    base[0] = 0x11;
    base[page_size] = 0x22;
    base[page_size * 2] = 0x33;

    if (munmap(base + page_size, (size_t) page_size) != 0) return 3;

    uint8_t *middle = mmap(base + page_size, (size_t) page_size, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (middle == MAP_FAILED) return 4;
    if (middle != base + page_size) return 5;

    middle[0] = 0x44;

    if (base[0] != 0x11) return 6;
    if (middle[0] != 0x44) return 7;
    if (base[page_size * 2] != 0x33) return 8;

    if (munmap(base, (size_t) page_size) != 0) return 9;
    if (munmap(middle, (size_t) page_size) != 0) return 10;
    if (munmap(base + (page_size * 2), (size_t) page_size) != 0) return 11;
    return 0;
}
