// RUN: %cc %s -o %t && %runner %t
//
// madvise(MADV_DONTNEED) on an initialized data segment must not lose the
// segment's contents. On a private file-backed mapping DONTNEED drops the
// resident pages and the next access re-reads the file's original bytes -- so
// the values must reappear, not zero-fill. A translator that maps file-backed
// PT_LOAD segments as anonymous copies instead zero-fills them here, silently
// corrupting the segment: this is what zeroed a real guest's embedded bytecode
// (JavaScriptCore in Bun DONTNEEDs its own data pages), producing a blank code
// block and a wild crash much later.
//
// `data` is a multi-page initialized global with sentinels on its first and
// last page; after DONTNEED both must still read their original values.

#include <stdint.h>
#include <sys/mman.h>

#if defined(__x86_64__) && defined(__linux__)

#define N 4096 // 16 KiB = four pages
static uint32_t data[N] = {[0] = 0xdeadbeef, [N - 1] = 0xcafef00d};

int main(void) {
    if (data[0] != 0xdeadbeef || data[N - 1] != 0xcafef00d) return 1;

    uintptr_t start = (uintptr_t) data & ~4095UL;
    uintptr_t end = ((uintptr_t) &data[N] + 4095) & ~4095UL;
    if (madvise((void *) start, end - start, MADV_DONTNEED) != 0) return 2;

    // File-backed: the originals must be re-read, not zero-filled.
    if (data[0] != 0xdeadbeef) return 3;
    if (data[N - 1] != 0xcafef00d) return 4;
    return 0;
}

#else
#error "unsupported host for madvise-dontneed test"
#endif
