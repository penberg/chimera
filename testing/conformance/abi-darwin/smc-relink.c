// RUN: %cc %s -o %t && %runner %t
//
// Self-modifying code reached through a *linked* edge. Once the arm64 cache
// links direct branches and caches indirect-branch targets, a block can be
// entered without the run loop resolving it — so dropping a translation on a
// self-modifying write is not enough: every link naming it must be broken too,
// or a predecessor keeps jumping into the stale block.
//
// The guest builds a function in a JIT page, calls it through a function
// pointer (an indirect branch, so its target is cached) in a loop (a hot,
// linked back edge), rewrites it in place, and calls it again the same way.
// The second round must observe the new code.

#include <libkern/OSCacheControl.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/mman.h>

typedef int (*fn)(void);

// `mov w0, #imm; ret` — the whole function, rebuilt in place each round.
static void build(uint32_t *code, unsigned value) {
    pthread_jit_write_protect_np(0);
    code[0] = 0x52800000 | (value << 5); // mov w0, #value
    code[1] = 0xD65F03C0;                // ret
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(code, 8);
}

// Call through a function pointer, repeatedly: the indirect branch caches its
// target and the loop's back edge is linked, so both fast paths are warm on
// the block that is about to be rewritten.
static int call_hot(fn f) {
    int last = 0;
    for (int i = 0; i < 1000; i++) last = f();
    return last;
}

int main(void) {
    size_t pg = 16384;
    uint32_t *code = mmap(0, pg, PROT_READ | PROT_WRITE | PROT_EXEC,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT, -1, 0);
    if (code == MAP_FAILED) return 1;

    build(code, 1);
    if (call_hot((fn) code) != 1) return 2;

    build(code, 2); // self-modifying write to already-translated, linked code
    if (call_hot((fn) code) != 2) return 3; // a stale translation returns 1

    return 0;
}
