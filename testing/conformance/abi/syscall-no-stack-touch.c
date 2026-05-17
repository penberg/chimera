// RUN: %cc %s -o %t && %runner %t
//
// A translator that fakes the syscall instruction must not touch the
// caller's stack: real kernels switch to their own per-thread stack and
// never push to the user stack. We park sp one byte above a `PROT_NONE`
// guard page so any user-stack write inside the syscall would fault.
// The syscall we issue exits with status 0 — if we get back to the C
// code after `syscall`, the exit didn't take and we return non-zero.

#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 1;

    void *mapping = mmap(0, (size_t) page_size * 2, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) return 2;

    if (mprotect(mapping, (size_t) page_size, PROT_NONE) != 0) return 3;

    uintptr_t guest_sp = (uintptr_t) mapping + (uintptr_t) page_size;

#if defined(__x86_64__) && defined(__linux__)
    // Linux: exit_group is syscall #231.
    asm volatile(
        "mov %[guest_sp], %%rsp\n\t"
        "xor %%edi, %%edi\n\t"
        "mov $231, %%rax\n\t"
        "syscall\n\t"
        "ud2\n\t"
        :
        : [guest_sp] "r"(guest_sp)
        : "rax", "rdi", "rcx", "r11", "memory", "cc");
#elif defined(__aarch64__) && defined(__APPLE__)
    // Darwin: `exit` is BSD syscall #1; number goes in x16, arg0 in x0,
    // entry via `svc #0x80`. Use a `brk` after as the "shouldn't return"
    // marker (the syscall doesn't return; if it does, we trap hard).
    asm volatile(
        "mov sp, %[guest_sp]\n\t"
        "mov x0, #0\n\t"
        "mov x16, #1\n\t"
        "svc #0x80\n\t"
        "brk #0\n\t"
        :
        : [guest_sp] "r"(guest_sp)
        : "x0", "x16", "memory", "cc");
#else
#error "unsupported host for syscall-no-stack-touch test"
#endif

    return 4;
}
