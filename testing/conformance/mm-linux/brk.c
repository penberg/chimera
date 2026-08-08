// RUN: %cc %s -o %t && %runner %t
//
// The kernel's brk contract, through raw syscalls so libc's sbrk caching
// cannot mask the runtime's answers: brk(0) reads the break back, a grow
// returns the requested break and the new pages are writable, a shrink
// followed by a regrow reads zeroes, and a request no break can satisfy
// (below the arena, or absurdly large) leaves the break unchanged.

#define _GNU_SOURCE
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

static uintptr_t raw_brk(uintptr_t addr) {
    return (uintptr_t) syscall(SYS_brk, addr);
}

int main(void) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 1;
    size_t page = (size_t) page_size;

    uintptr_t base = raw_brk(0);
    if (base == 0) return 2;
    if (raw_brk(0) != base) return 3;

    // Work from a page-aligned break: the initial one may sit mid-page, and a
    // shrink only discards whole pages past the rounded-up break.
    uintptr_t start = (base + page - 1) & ~(uintptr_t) (page - 1);
    uintptr_t grown = start + 2 * page;
    if (raw_brk(grown) != grown) return 4;
    if (raw_brk(0) != grown) return 5;
    uint8_t *heap = (uint8_t *) start;
    memset(heap, 0x5a, 2 * page);
    if (heap[0] != 0x5a || heap[2 * page - 1] != 0x5a) return 6;

    // Shrink away the second page, then regrow: the page must come back
    // zeroed, and the surviving page must keep its contents.
    if (raw_brk(start + page) != start + page) return 7;
    if (raw_brk(grown) != grown) return 8;
    if (heap[0] != 0x5a || heap[page - 1] != 0x5a) return 9;
    for (size_t i = page; i < 2 * page; i++)
        if (heap[i] != 0) return 10;

    // A request below the break's floor fails with the break unchanged.
    if (raw_brk(page) != grown) return 11;
    // So does one no address space can satisfy.
    if (raw_brk(UINTPTR_MAX & ~(uintptr_t) (page - 1)) != grown) return 12;
    if (raw_brk(0) != grown) return 13;

    return 0;
}
