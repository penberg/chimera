// RUN: %cc %s -o %t && %runner %t
//
// A basic block longer than the translator's fixed decode window. The window
// is read as a fixed-size slice of guest memory, so an instruction that
// straddles its end is handed to the decoder truncated and misreported as
// invalid -- which crashed translation of a long jump-table initializer in a
// real binary (`translate: unhandled terminator ... flow_control=Exception
// code=INVALID`). The translator must instead split an over-long block at an
// instruction boundary and continue.
//
// The `.rept` emits one straight-line run of ~8000 bytes with no branches, well
// past the 4096-byte window, forcing at least one split. The running sum is
// carried across the split point, so a split that drops or duplicates an
// instruction -- or resumes decode at the wrong guest PC -- yields the wrong
// total rather than merely not crashing.

#include <stdint.h>

#define REPS 2000

#if defined(__x86_64__) && defined(__linux__)

int main(void) {
    uint64_t acc = 0;
    asm volatile(
        ".rept 2000\n\t"
        "add $1, %0\n\t"
        ".endr\n\t"
        : "+r"(acc)
        :
        : "cc");

    return acc == REPS ? 0 : 1;
}

#elif defined(__aarch64__) && defined(__APPLE__)

int main(void) {
    uint64_t acc = 0;
    asm volatile(
        ".rept 2000\n\t"
        "add %0, %0, #1\n\t"
        ".endr\n\t"
        : "+r"(acc)
        :
        : "cc");

    return acc == REPS ? 0 : 1;
}

#else
#error "unsupported host for long-block test"
#endif
