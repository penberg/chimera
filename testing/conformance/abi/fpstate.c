// RUN: %cc %s -o %t && %runner %t
//
// Every transition between guest and runtime saves and restores the guest's
// vector state. A translator that saves only the legacy x87+SSE state (via
// FXSAVE/FXRSTOR on x86) drops the upper 128 bits of the AVX YMM registers,
// because those bits live outside the FXSAVE area and need XSAVE. The loss is
// invisible until guest code keeps live state in the upper lane across a
// transition -- exactly what glibc's AVX2 string routines do, with a runtime
// round-trip on every loop back-edge. This test parks distinct 256-bit
// sentinels in several YMM registers, forces a round-trip with a harmless
// syscall, and checks that all 256 bits survived. The whole sequence is one
// asm block so the compiler cannot insert a `vzeroupper` between setup and
// check, and the syscall is the only thing standing between them.
//
// The inline asm uses AVX mnemonics directly, so the test needs no `-mavx`:
// the assembler always knows them, and a runtime CPUID check skips hosts
// without AVX rather than executing an illegal instruction there.

#include <stdint.h>
#include <string.h>
#include <unistd.h>

#if defined(__x86_64__) && defined(__linux__)

int main(void) {
    // A fresh process must start with every SSE floating-point exception
    // masked: MXCSR = 0x1f80, the value the kernel seeds. A translator that
    // restores guest vector state from a zeroed save area loads MXCSR = 0
    // instead, unmasking every exception so the first inexact multiply or
    // divide traps with SIGFPE. This holds regardless of AVX, so check it
    // before the AVX gate below.
    uint32_t mxcsr = 0;
    asm volatile("stmxcsr %0" : "=m"(mxcsr));
    if ((mxcsr & 0x1f80) != 0x1f80) return 5;

    if (!__builtin_cpu_supports("avx")) return 0; // nothing more to preserve

    // Distinct sentinels per register: distinct values catch a translator
    // that swaps or aliases vector registers, not just one that drops the
    // upper lane. The high 16 bytes of each are the part FXSAVE omits.
    static const uint8_t pat[3][32] __attribute__((aligned(32))) = {
        {0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
         0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
         0xf0,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,
         0xf8,0xf9,0xfa,0xfb,0xfc,0xfd,0xfe,0xff},
        {0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,
         0x28,0x29,0x2a,0x2b,0x2c,0x2d,0x2e,0x2f,
         0xd0,0xd1,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,
         0xd8,0xd9,0xda,0xdb,0xdc,0xdd,0xde,0xdf},
        {0x40,0x41,0x42,0x43,0x44,0x45,0x46,0x47,
         0x48,0x49,0x4a,0x4b,0x4c,0x4d,0x4e,0x4f,
         0xb0,0xb1,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,
         0xb8,0xb9,0xba,0xbb,0xbc,0xbd,0xbe,0xbf},
    };
    uint8_t out[3][32] __attribute__((aligned(32)));
    memset(out, 0, sizeof out);

    // ymm0, ymm7, ymm15 span the register file so a partial save is caught.
    asm volatile(
        "vmovdqa %[p0], %%ymm0\n\t"
        "vmovdqa %[p7], %%ymm7\n\t"
        "vmovdqa %[p15], %%ymm15\n\t"
        "mov $39, %%rax\n\t"          // SYS_getpid: a harmless round-trip
        "syscall\n\t"
        "vmovdqu %%ymm0,  %[o0]\n\t"
        "vmovdqu %%ymm7,  %[o7]\n\t"
        "vmovdqu %%ymm15, %[o15]\n\t"
        : [o0] "=m"(out[0]), [o7] "=m"(out[1]), [o15] "=m"(out[2])
        : [p0] "m"(pat[0]), [p7] "m"(pat[1]), [p15] "m"(pat[2])
        : "rax", "rcx", "r11", "ymm0", "ymm7", "ymm15", "memory");

    for (int i = 0; i < 3; i++) {
        if (memcmp(out[i], pat[i], 32) == 0) continue;
        // 10-12: the upper 128 bits were lost (the FXSAVE-only signature);
        // 20-22: the low 128 bits differ too (an unrelated failure).
        if (memcmp(out[i] + 16, pat[i] + 16, 16) != 0) return 10 + i;
        return 20 + i;
    }
    return 0;
}

#elif defined(__aarch64__) && defined(__APPLE__)

// Darwin arm64 has no FXSAVE/YMM split -- the NEON V registers are 128 bits
// and the trampoline saves each whole. The analogous contract still holds:
// a full 128-bit vector register must survive a syscall round-trip. We seed
// q0, issue BSD getpid (#20 in x16, entry via `svc #0x80`), and check it.
int main(void) {
    static const uint8_t pat[16] __attribute__((aligned(16))) = {
        0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
        0x88,0x99,0xaa,0xbb,0xcc,0xdd,0xee,0xff,
    };
    uint8_t out[16] __attribute__((aligned(16)));
    memset(out, 0, sizeof out);

    asm volatile(
        "ldr q0, [%[p]]\n\t"
        "mov x16, #20\n\t"            // SYS_getpid (BSD #20) on Darwin
        "svc #0x80\n\t"
        "str q0, [%[o]]\n\t"
        :
        : [p] "r"(pat), [o] "r"(out)
        : "x0", "x1", "x16", "v0", "memory", "cc");

    if (memcmp(out, pat, 16) != 0) return 1;
    return 0;
}

#else
#error "unsupported host for fpstate test"
#endif
