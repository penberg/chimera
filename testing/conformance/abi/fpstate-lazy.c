// RUN: %cc %s -o %t && %runner %t
//
// Chimera saves and restores the guest's vector state lazily: a cache
// residency that touches no FP/SIMD register skips the XSAVE/XRSTOR around the
// boundary entirely, leaving the canonical copy in the thread's save area
// untouched. The save is taken only on entry to a block that actually uses FP,
// and written back only on exit from a residency that did. This test exercises
// the seam directly: it parks a 256-bit sentinel in ymm0, then issues many
// bare getpid syscalls in a tight loop. Each syscall leaves and re-enters the
// code cache, but the loop body uses no vector register, so every one of those
// round-trips must preserve ymm0 purely by leaving the save area alone. A bug
// that saved garbage on an FP-free exit, or failed to restore on the final
// FP block, corrupts the sentinel. The setup, the loop, and the readback are a
// single asm block so the compiler cannot spill ymm0 across the syscalls
// itself (which would hide the runtime's mistake) or slip in a vzeroupper.

#include <stdint.h>
#include <string.h>
#include <unistd.h>

#if defined(__x86_64__) && defined(__linux__)

int main(void) {
    if (!__builtin_cpu_supports("avx")) return 0; // nothing to preserve

    static const uint8_t pat[32] __attribute__((aligned(32))) = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
        0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
    };
    uint8_t out[32] __attribute__((aligned(32)));
    memset(out, 0, sizeof out);

    // ebx holds the loop count: it is callee-saved, so the kernel preserves it
    // across each syscall, and the loop body names no vector register. ymm0
    // stays live across all 1000 round-trips with no instruction touching it.
    asm volatile(
        "vmovdqa %[p], %%ymm0\n\t"
        "mov $1000, %%ebx\n\t"
        "1:\n\t"
        "mov $39, %%rax\n\t" // SYS_getpid
        "syscall\n\t"
        "dec %%ebx\n\t"
        "jnz 1b\n\t"
        "vmovdqu %%ymm0, %[o]\n\t"
        : [o] "=m"(out)
        : [p] "m"(pat)
        : "rax", "rbx", "rcx", "r11", "ymm0", "memory");

    if (memcmp(out, pat, 32) != 0) {
        // 1: the upper 128 bits were lost; 2: the low lane differs too.
        if (memcmp(out + 16, pat + 16, 16) != 0) return 1;
        return 2;
    }
    return 0;
}

#else

int main(void) { return 0; }

#endif
