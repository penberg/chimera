// RUN: %cc %s -o %t && %runner %t
//
// Verifies that the syscall instruction's architectural side-effects match
// the host ABI. Each host has its own conventions, and the test asserts
// the ones a translator must preserve.

#include <stdint.h>
#include <unistd.h>

#if defined(__x86_64__) && defined(__linux__)

// Linux x86_64: the `syscall` instruction clobbers rcx (with the address
// of the following instruction) and r11 (with the caller's rflags). A
// translator that handles `syscall` itself must reproduce both side
// effects so libc's stub layout keeps working.
int main(void) {
    unsigned long rcx_after = 0;
    unsigned long expected_rcx = 0;
    unsigned long r11_after = 0;
    unsigned long rflags_before = 0;
    long ret;

    asm volatile(
        "pushfq\n\t"
        "pop %[rflags_before]\n\t"
        "leaq 1f(%%rip), %%rax\n\t"
        "mov %%rax, %[expected_rcx]\n\t"
        "mov $39, %%rax\n\t"              // SYS_getpid on Linux
        "movabs $0x1111111122222222, %%rcx\n\t"
        "movabs $0x3333333344444444, %%r11\n\t"
        "syscall\n\t"
        "1:\n\t"
        "mov %%rcx, %[rcx_after]\n\t"
        "mov %%r11, %[r11_after]\n\t"
        : [rcx_after] "=m"(rcx_after),
          [expected_rcx] "=m"(expected_rcx),
          [r11_after] "=m"(r11_after),
          [rflags_before] "=m"(rflags_before),
          "=a"(ret)
        :
        : "rcx", "r11", "memory", "cc");

    if (ret <= 0) return 1;
    if (rcx_after != expected_rcx) return 2;
    if (r11_after != rflags_before) return 3;
    return 0;
}

#elif defined(__aarch64__) && defined(__APPLE__)

// Darwin arm64: `svc #0x80` with the syscall number in x16. On return, x0
// holds the result and the NZCV carry flag indicates error (set on error,
// clear on success). The kernel is free to clobber x16/x17 but every
// other callee-saved register and the rest of NZCV must survive. A
// translator that fakes the syscall must reproduce that exact contract.
int main(void) {
    long ret = 0;
    unsigned long carry = 1;
    unsigned long preserved_x19 = 0x1111111122222222UL;
    unsigned long preserved_x20 = 0x3333333344444444UL;
    unsigned long x19_after = 0;
    unsigned long x20_after = 0;

    asm volatile(
        "mov x19, %[in19]\n\t"
        "mov x20, %[in20]\n\t"
        "mov x16, #20\n\t"                // SYS_getpid (BSD #20) on Darwin
        "svc #0x80\n\t"
        "cset %[carry], cs\n\t"
        "mov %[x19_after], x19\n\t"
        "mov %[x20_after], x20\n\t"
        "mov %[ret], x0\n\t"
        : [ret] "=r"(ret),
          [carry] "=r"(carry),
          [x19_after] "=r"(x19_after),
          [x20_after] "=r"(x20_after)
        : [in19] "r"(preserved_x19),
          [in20] "r"(preserved_x20)
        : "x0", "x16", "x19", "x20", "memory", "cc");

    if (ret <= 0) return 1;
    if (carry != 0) return 2;             // success must clear carry
    if (x19_after != preserved_x19) return 3;
    if (x20_after != preserved_x20) return 4;
    return 0;
}

#else
#error "unsupported host for syscall-clobber test"
#endif
