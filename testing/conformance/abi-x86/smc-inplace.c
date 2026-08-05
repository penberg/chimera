// RUN: %cc %s -o %t && %runner %t
//
// Self-modifying code: a guest overwrites already-executed machine code in
// place and runs it again, expecting the new behavior. A dynamic translator
// that caches translations must notice the write and re-translate, or it runs
// the stale code. Chimera maps a writable+executable (JIT) page read-only once
// it has translated from it; the overwrite then traps, the page's translations
// are dropped, and the next call re-translates from the new bytes.
//
// `f` is built twice in the same page: first `mov eax, 1; ret`, then
// `mov eax, 2; ret`. The first call returns 1, the in-place rewrite must take
// effect, and the second call must return 2 rather than a cached 1.

#include <string.h>
#include <sys/mman.h>

#if defined(__x86_64__) && defined(__linux__)

typedef int (*fn)(void);

int main(void) {
    long pg = 4096;
    unsigned char *code = mmap(0, pg, PROT_READ | PROT_WRITE | PROT_EXEC,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (code == MAP_FAILED) return 1;

    static const unsigned char ret1[] = {0xB8, 0x01, 0x00, 0x00, 0x00, 0xC3}; // mov eax,1; ret
    static const unsigned char ret2[] = {0xB8, 0x02, 0x00, 0x00, 0x00, 0xC3}; // mov eax,2; ret

    memcpy(code, ret1, sizeof ret1);
    __builtin___clear_cache((char *) code, (char *) code + pg);
    int first = ((fn) code)();

    memcpy(code, ret2, sizeof ret2); // self-modifying write to translated code
    __builtin___clear_cache((char *) code, (char *) code + pg);
    int second = ((fn) code)();

    if (first != 1) return 2;
    if (second != 2) return 3; // a stale cached translation returns 1 here
    return 0;
}

#else
#error "unsupported host for smc-inplace test"
#endif
