// RUN: %cc %s -o %t && %runner %t
//
// Code whose last byte is the last byte of its mapping must still run: the
// translator's fixed-size decode window straddles past the function into the
// unmapped page, so the guarded fetch is cut short and only the readable
// prefix reaches the decoder. A translator that demands the full window --
// or that rounds the readable prefix wrong -- either refuses to translate
// the block or crashes on the fetch. The function is placed so it ends
// exactly at the mapping's end, the worst case for the window.

#include <string.h>
#include <sys/mman.h>

#if defined(__x86_64__) && defined(__linux__)

typedef int (*fn)(void);

int main(void) {
    long pg = 4096;
    // Two pages, the second dropped to PROT_NONE so the readable mapping ends
    // at a boundary the decode window is guaranteed to straddle.
    unsigned char *code = mmap(0, pg * 2, PROT_READ | PROT_WRITE | PROT_EXEC,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (code == MAP_FAILED) return 1;
    if (mprotect(code + pg, pg, PROT_NONE) != 0) return 2;

    static const unsigned char ret42[] = {0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3}; // mov eax,42; ret
    unsigned char *f = code + pg - sizeof ret42;
    memcpy(f, ret42, sizeof ret42);
    __builtin___clear_cache((char *) code, (char *) code + pg);

    if (((fn) f)() != 42) return 3;
    return 0;
}

#else
#error "unsupported host for code-at-mapping-end test"
#endif
