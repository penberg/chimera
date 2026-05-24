// RUN: %cc %s -o %t && %runner %t
//
// An indirect call must read its target operand using the stack pointer the
// guest had *before* the call -- exactly as hardware does: `CALL m` computes
// the target from memory, then pushes the return address. A translator that
// rewrites the call by pushing the return address first and only then reading
// the operand evaluates an rsp-relative target against the post-push rsp,
// reading 8 bytes off and branching to garbage. `call qword ptr [rsp+disp]`
// is common in real code (it is how bash crashed: `ff 54 24 58`).
//
// We stage the correct target at [rsp+0x58] and a decoy at [rsp+0x50] (the
// slot a translator that pushed first would read instead), then call through
// [rsp+0x58]. The right target returns one sentinel, the decoy another, so
// the bug is caught as a clean wrong answer rather than relying on a crash.

#include <stdint.h>

#if defined(__x86_64__) && defined(__linux__)

int main(void) {
    uint64_t result = 0;
    asm volatile(
        "sub $0x100, %%rsp\n\t"            // scratch frame (16-byte multiple)
        "lea 1f(%%rip), %%rax\n\t"
        "mov %%rax, 0x58(%%rsp)\n\t"       // correct target
        "lea 3f(%%rip), %%rax\n\t"
        "mov %%rax, 0x50(%%rsp)\n\t"       // decoy at the off-by-8 slot
        "call *0x58(%%rsp)\n\t"           // indirect call through the stack
        "jmp 2f\n\t"
        "1:\n\t" "movq $0x600d, %[r]\n\t" "ret\n\t"   // reached iff rsp was right
        "3:\n\t" "movq $0x0bad, %[r]\n\t" "ret\n\t"   // reached iff read was off by 8
        "2:\n\t"
        "add $0x100, %%rsp\n\t"
        : [r] "=r"(result)
        :
        : "rax", "cc", "memory");

    return result == 0x600d ? 0 : 1;
}

#elif defined(__aarch64__) && defined(__APPLE__)

// AArch64 has no memory-operand call and BL/BLR records the return address in
// the link register, not on the stack -- so adjusting sp is never part of a
// call and the hazard above cannot arise. The invariant still holds: an
// indirect call through a value loaded from the stack must reach that target.
// Load a function pointer from a stack slot and `blr` it.
int main(void) {
    uint64_t result = 0;
    asm volatile(
        "sub sp, sp, #0x100\n\t"
        "adr x9, 1f\n\t"
        "str x9, [sp, #0x58]\n\t"
        "ldr x9, [sp, #0x58]\n\t"
        "blr x9\n\t"
        "b 2f\n\t"
        "1:\n\t" "movz %[r], #0x600d\n\t" "ret\n\t"
        "2:\n\t"
        "add sp, sp, #0x100\n\t"
        : [r] "=r"(result)
        :
        : "x9", "x30", "cc", "memory");

    return result == 0x600d ? 0 : 1;
}

#else
#error "unsupported host for indirect-call test"
#endif
