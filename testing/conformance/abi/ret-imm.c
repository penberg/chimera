// RUN: %cc -O2 %s -o %t && %runner %t
// UNSUPPORTED: darwin -- x86 inline asm (`ret imm16` has no arm64 analogue)
//
// `ret imm16`: pops the return address, then releases imm16 more bytes of
// stack — the callee-cleans-arguments convention V8's builtins use. A
// translator that lowers it as a bare `ret` leaves the caller's stack pointer
// short by imm16 after every such return; the drift eventually feeds a stale
// stack slot to a later `ret` and control flies into the weeds. Like `ret`,
// the form must not touch the arithmetic flags (an `add rsp, imm` lowering
// would). The callee returns with `ret $16` holding ZF set; the caller checks
// the returned value, that ZF survived, and that %rsp came back exactly where
// it started.

unsigned long call_and_check(void);

__asm__(
    ".text\n"
    // callee_ret16(a=%rdi, b=%rsi): result in %rax, ZF set, returns with
    // `ret $16`, releasing the two qwords its caller pushed for it.
    "callee_ret16:\n"
    "  lea (%rdi,%rsi), %rax\n"
    "  cmp %rdi, %rdi\n"
    "  ret $16\n"

    // call_and_check(): pushes two qwords, calls callee_ret16, and verifies
    // ZF is still set and %rsp is back to its pre-push value. Returns the sum
    // on success, 0 on a flags or stack mismatch.
    "call_and_check:\n"
    "  mov %rsp, %r11\n"
    "  push $2\n"
    "  push $3\n"
    "  mov $40, %rdi\n"
    "  mov $2, %rsi\n"
    "  call callee_ret16\n"
    "  jne 1f\n"
    "  cmp %rsp, %r11\n"
    "  jne 1f\n"
    "  ret\n"
    "1:\n"
    "  xor %eax, %eax\n"
    "  ret\n");

int main(void) {
    return call_and_check() == 42 ? 0 : 1;
}
