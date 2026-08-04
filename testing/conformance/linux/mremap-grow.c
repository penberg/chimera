// RUN: %cc %s -o %t && %runner %t
// UNSUPPORTED: darwin -- mremap(2) is Linux-only
//
// Grow an anonymous mapping with mremap(MREMAP_MAYMOVE). The existing contents
// must survive and the grown tail must be writable.

#define _GNU_SOURCE
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 1;

    size_t old_len = (size_t) page_size;
    size_t new_len = (size_t) page_size * 2;
    uint8_t *p = mmap(0, old_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 2;

    p[0] = 0x5a;
    p[old_len - 1] = 0xa5;

    uint8_t *grown = mremap(p, old_len, new_len, MREMAP_MAYMOVE);
    if (grown == MAP_FAILED) return 3;

    if (grown[0] != 0x5a) return 4;
    if (grown[old_len - 1] != 0xa5) return 5;

    grown[new_len - 1] = 0x3c;
    if (grown[new_len - 1] != 0x3c) return 6;

    if (munmap(grown, new_len) != 0) return 7;
    return 0;
}
