// RUN: %cc %s -o %t && %runner %t
//
// Thread-local storage rides the FS segment base on x86-64 Linux: a `__thread`
// access compiles to a `mov` through `fs:`. Chimera keeps its own FS base for
// the runtime's Rust TLS and installs the guest's only lazily -- the first
// block of a cache residency that actually reads `fs:` swaps it in, and the
// exit swaps Chimera's back. A bug in that swap (installing the wrong base,
// failing to install it, or failing to restore Chimera's afterward) corrupts
// every TLS access. This test exercises it across the grain of the laziness:
// it reads and mutates thread-locals, forces many cache round-trips with bare
// getpid syscalls in between, and checks the values survived. Several distinct
// thread-locals catch a swap that installs a stale or aliased base.

#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(__x86_64__) && defined(__linux__)

static __thread uint64_t a = 0x1111111111111111ull;
static __thread uint64_t b = 0x2222222222222222ull;
static __thread uint64_t c = 0x3333333333333333ull;

// Defeat the optimizer so each call really re-reads the thread-local through
// fs: rather than caching it in a register across the loop.
static uint64_t __attribute__((noinline)) load(volatile uint64_t *p) { return *p; }

int main(void) {
    // Initial values must read back through fs: correctly.
    if (load(&a) != 0x1111111111111111ull) return 1;
    if (load(&b) != 0x2222222222222222ull) return 2;
    if (load(&c) != 0x3333333333333333ull) return 3;

    // Mutate the thread-locals, then interleave TLS reads with syscalls. Each
    // getpid leaves and re-enters the code cache (an FS-free residency that
    // must leave the saved base alone); each load() re-enters and must install
    // the guest base again. Accumulating into a thread-local keeps fs: traffic
    // on every iteration.
    a = 0xaaaaaaaaaaaaaaaaull;
    b = 0xbbbbbbbbbbbbbbbbull;
    for (int i = 0; i < 1000; i++) {
        (void)syscall(SYS_getpid);
        c = load(&a) ^ load(&b) ^ (uint64_t)i;
    }

    if (load(&a) != 0xaaaaaaaaaaaaaaaaull) return 4;
    if (load(&b) != 0xbbbbbbbbbbbbbbbbull) return 5;
    if (load(&c) != (0xaaaaaaaaaaaaaaaaull ^ 0xbbbbbbbbbbbbbbbbull ^ 999ull)) return 6;
    return 0;
}

#else

int main(void) { return 0; }

#endif
