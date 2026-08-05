// RUN: %cc %s -o %t && %runner %t
//
// The safepoint poll on a linked conditional back-edge must not perturb the
// guest's arithmetic flags. A multi-limb add carries the carry flag (CF) across
// the loop's back-edge: each iteration's `adc` consumes the previous one's
// carry, the index is stepped with `lea` (no flag effect), and the counter with
// `dec` (which preserves CF and sets ZF for the `jnz`). The loop is a single
// linked block, so the back-edge poll runs every iteration; a flag-clobbering
// poll breaks the carry chain and corrupts the sum. No signal is needed — the
// bug surfaces purely from the poll running on linked iterations.
//
// Adding all-ones + 1 across N limbs must yield all-zero limbs and a carry out.

#include <stdint.h>

#define N 64

int main(void) {
    static uint64_t a[N], b[N], r[N];
    for (int i = 0; i < N; i++) {
        a[i] = ~0ull;
        b[i] = 0;
        r[i] = 0;
    }
    b[0] = 1;

    uint64_t idx = 0;
    uint64_t cnt = N;
    unsigned char carry;
    __asm__ volatile(
        "clc\n\t"
        "1:\n\t"
        "mov (%[a],%[i],8), %%rax\n\t"
        "adc (%[b],%[i],8), %%rax\n\t" // consumes + produces CF
        "mov %%rax, (%[r],%[i],8)\n\t"
        "lea 1(%[i]), %[i]\n\t" // i++ without touching flags
        "dec %[c]\n\t"          // preserves CF, sets ZF
        "jnz 1b\n\t"            // back-edge; CF must reach the next adc
        "setc %[carry]\n\t"
        : [i] "+r"(idx), [c] "+r"(cnt), [carry] "=q"(carry)
        : [a] "r"(a), [b] "r"(b), [r] "r"(r)
        : "rax", "cc", "memory");

    if (carry != 1) return 1;
    for (int i = 0; i < N; i++)
        if (r[i] != 0) return 2;
    return 0;
}
