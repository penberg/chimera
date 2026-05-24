// RUN: %cc %s -o %t && %runner %t
//
// A translator rewrites every block terminator into a stub that computes the
// next guest PC. Those stubs must not write below the guest's stack pointer.
// On x86-64 the System V ABI reserves the 128 bytes below rsp -- the "red
// zone" -- for a leaf function's own use, so a guest may keep live data there
// without adjusting rsp; clobbering it corrupts the guest. A translator that
// brackets a conditional branch with a scratch `push`/`pop` on the guest
// stack writes [rsp-8] and destroys whatever the guest parked there.
//
// The test fills the red zone with a sentinel, executes a conditional branch
// in each direction, then reads the bytes back -- all in one asm block so the
// compiler never spills into the red zone between the writes and the reads.
// The read-back is branch-free (xor/or into an accumulator) so the only block
// terminators between the fill and the check are the branches under test.

#include <stdint.h>

#if defined(__x86_64__) && defined(__linux__)

int main(void) {
    uint64_t acc;
    asm volatile(
        "movabs $0x5eed1e55c0ffee11, %%r10\n\t"
        // Seed several slots across the 128-byte red zone.
        "mov %%r10,  -8(%%rsp)\n\t"
        "mov %%r10, -16(%%rsp)\n\t"
        "mov %%r10, -64(%%rsp)\n\t"
        "mov %%r10, -128(%%rsp)\n\t"
        // A conditional branch that is NOT taken (ZF=1, so jnz falls through),
        // then one that IS taken. Both are block terminators.
        "xor %%ecx, %%ecx\n\t"
        "test %%ecx, %%ecx\n\t"
        "jnz 8f\n\t"
        "jz 1f\n\t"
        "8:\n\t"
        "ud2\n\t"                       // unreachable
        "1:\n\t"
        // acc = OR of (slot ^ sentinel) over every slot: zero iff all intact.
        "mov  -8(%%rsp), %%rdx\n\t" "xor %%r10, %%rdx\n\t" "mov %%rdx, %[acc]\n\t"
        "mov -16(%%rsp), %%rdx\n\t" "xor %%r10, %%rdx\n\t" "or  %%rdx, %[acc]\n\t"
        "mov -64(%%rsp), %%rdx\n\t" "xor %%r10, %%rdx\n\t" "or  %%rdx, %[acc]\n\t"
        "mov -128(%%rsp), %%rdx\n\t" "xor %%r10, %%rdx\n\t" "or  %%rdx, %[acc]\n\t"
        : [acc] "=r"(acc)
        :
        : "rcx", "rdx", "r10", "cc", "memory");

    return acc != 0; // 0 = red zone intact, 1 = a terminator clobbered it
}

#elif defined(__aarch64__) && defined(__APPLE__)

// AAPCS64 defines no red zone, so this is not about preserving ABI-reserved
// scratch -- but the same invariant holds: a translator must not write below
// the guest's sp when it rewrites a branch, since a guest may park sp one word
// above a guard page. Stash a sentinel below sp, branch in each direction, and
// confirm it survives.
int main(void) {
    uint64_t acc = 0;
    asm volatile(
        "movz x10, #0xee11\n\t"
        "movk x10, #0xc0ff, lsl #16\n\t"
        "movk x10, #0x1e55, lsl #32\n\t"
        "movk x10, #0x5eed, lsl #48\n\t"
        "stur x10, [sp, #-8]\n\t"
        "stur x10, [sp, #-64]\n\t"
        "cmp xzr, xzr\n\t"              // Z=1
        "b.ne 8f\n\t"                   // not taken
        "b.eq 1f\n\t"                   // taken
        "8:\n\t"
        "brk #0\n\t"                    // unreachable
        "1:\n\t"
        "ldur x11, [sp, #-8]\n\t"  "eor x11, x11, x10\n\t" "orr %[acc], %[acc], x11\n\t"
        "ldur x11, [sp, #-64]\n\t" "eor x11, x11, x10\n\t" "orr %[acc], %[acc], x11\n\t"
        : [acc] "+r"(acc)
        :
        : "x10", "x11", "cc", "memory");

    return acc != 0; // 0 = nothing written below sp, 1 = a terminator wrote there
}

#else
#error "unsupported host for red-zone test"
#endif
