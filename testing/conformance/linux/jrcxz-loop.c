// RUN: %cc -O2 %s -o %t && %runner %t
// UNSUPPORTED: darwin -- x86 inline asm (rel8-only jrcxz/loop have no arm64 analogue)
//
// The rel8-only conditional branches: jrcxz/jecxz (branch if rcx/ecx is zero)
// and the loop family (decrement rcx/ecx, branch while nonzero; loope/loopne
// additionally require ZF set/clear). They have no rel32 form, so a translator
// that lowers conditional branches as jcc rel32 must treat them specially, and
// none of them may clobber flags: a block fallthrough can branch on flags set
// before the jrcxz. Bun/JSC's runtime executes jrcxz on its startup path.

#include <stdint.h>

// jrcxz taken: rcx == 0 leaves the marker at 0.
static int jrcxz_taken(void) {
    uint64_t out;
    __asm__ volatile("xor %[out], %[out]\n\t"
                     "jrcxz 1f\n\t"
                     "mov $1, %[out]\n\t"
                     "1:"
                     : [out] "=&r"(out)
                     : "c"(0ULL)
                     : "cc");
    return out == 0;
}

// jrcxz not taken: rcx != 0 runs the fall-through.
static int jrcxz_not_taken(void) {
    uint64_t out;
    __asm__ volatile("xor %[out], %[out]\n\t"
                     "jrcxz 1f\n\t"
                     "mov $1, %[out]\n\t"
                     "1:"
                     : [out] "=&r"(out)
                     : "c"(7ULL)
                     : "cc");
    return out == 1;
}

// jecxz tests only the low 32 bits: rcx = 1<<32 has ecx == 0, so it branches.
static int jecxz_width(void) {
    uint64_t out;
    __asm__ volatile("xor %[out], %[out]\n\t"
                     "jecxz 1f\n\t"
                     "mov $1, %[out]\n\t"
                     "1:"
                     : [out] "=&r"(out)
                     : "c"(1ULL << 32)
                     : "cc");
    return out == 0;
}

// jrcxz must not clobber flags: the ZF from the cmp before it decides the jne
// after it (in the successor block).
static int jrcxz_preserves_flags(void) {
    uint64_t out;
    __asm__ volatile("xor %[out], %[out]\n\t"
                     "cmp %[c], %[c]\n\t" // ZF = 1
                     "jrcxz 1f\n\t"       // rcx != 0: not taken, ZF must survive
                     "1:\n\t"
                     "jne 2f\n\t" // ZF still 1: not taken
                     "mov $1, %[out]\n\t"
                     "2:"
                     : [out] "=&r"(out)
                     : [c] "c"(5ULL)
                     : "cc");
    return out == 1;
}

// loop decrements rcx and iterates while it is nonzero: 5 rounds.
static int loop_counts(void) {
    uint64_t iters = 0, cnt = 5;
    __asm__ volatile("1:\n\t"
                     "inc %[it]\n\t"
                     "loop 1b"
                     : [it] "+r"(iters), [c] "+c"(cnt)
                     :
                     : "cc");
    return iters == 5 && cnt == 0;
}

// loope continues while ZF is set: the body's cmp x,x keeps ZF = 1, so it
// runs rcx down to 0.
static int loope_runs_while_equal(void) {
    uint64_t iters = 0, cnt = 3;
    __asm__ volatile("1:\n\t"
                     "inc %[it]\n\t"
                     "cmp %[it], %[it]\n\t" // ZF = 1
                     "loope 1b"
                     : [it] "+r"(iters), [c] "+c"(cnt)
                     :
                     : "cc");
    return iters == 3;
}

// loopne continues while ZF is clear: the body's never-equal cmp keeps ZF = 0,
// so it also runs rcx down to 0.
static int loopne_runs_while_unequal(void) {
    uint64_t iters = 0, cnt = 4;
    __asm__ volatile("1:\n\t"
                     "inc %[it]\n\t"
                     "cmp $0x1234567, %[it]\n\t" // never equal: ZF = 0
                     "loopne 1b"
                     : [it] "+r"(iters), [c] "+c"(cnt)
                     :
                     : "cc");
    return iters == 4;
}

// loopne also stops early when ZF becomes set: equality on iteration 2 ends a
// count of 6 after 2 rounds.
static int loopne_stops_on_equal(void) {
    uint64_t iters = 0, cnt = 6;
    __asm__ volatile("1:\n\t"
                     "inc %[it]\n\t"
                     "cmp $2, %[it]\n\t"
                     "loopne 1b"
                     : [it] "+r"(iters), [c] "+c"(cnt)
                     :
                     : "cc");
    return iters == 2 && cnt == 4;
}

int main(void) {
    if (!jrcxz_taken())
        return 1;
    if (!jrcxz_not_taken())
        return 2;
    if (!jecxz_width())
        return 3;
    if (!jrcxz_preserves_flags())
        return 4;
    if (!loop_counts())
        return 5;
    if (!loope_runs_while_equal())
        return 6;
    if (!loopne_runs_while_unequal())
        return 7;
    if (!loopne_stops_on_equal())
        return 8;
    return 0;
}
