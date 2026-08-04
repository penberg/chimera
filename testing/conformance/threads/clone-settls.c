// RUN: %cc %s -o %t && %runner %t
// UNSUPPORTED: darwin -- clone is a Linux thread primitive; Darwin uses bsdthread_create.
//
// clone(CLONE_SETTLS): the new thread gets its own thread pointer (%fs base),
// taken from the `tls` clone argument — this is how each pthread gets a private
// TLS block. Chimera virtualizes %fs per guest thread, so the child must run
// with the TLS base the clone requested, distinct from the parent's. The child
// reads a sentinel through %fs and the parent confirms it saw the child's block
// (not its own). Exits 0 only on a match.

#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <sys/mman.h>

// The child's TLS block. [0] is the TCB self-pointer (the %fs:0 convention, so
// nothing that peeks at it dereferences garbage); [1] is the sentinel the child
// reads back through %fs:0x8.
static uint64_t child_tls[2];
static volatile uint64_t child_saw;

#define SENTINEL 0xDEADBEEFCAFEBABEULL

static int child_fn(void *arg) {
    (void) arg;
    uint64_t v;
    __asm__ volatile("movq %%fs:0x8, %0" : "=r"(v));
    child_saw = v;
    // Exit just this thread, touching no TLS of our own beyond the read above.
    __asm__ volatile("movl $60, %%eax\n\t" // __NR_exit
                     "xorl %%edi, %%edi\n\t"
                     "syscall\n\t"
                     :
                     :
                     : "rax", "rdi", "rcx", "r11", "memory");
    return 0; // unreachable
}

int main(void) {
    child_tls[0] = (uint64_t) &child_tls; // self-pointer at %fs:0
    child_tls[1] = SENTINEL; // sentinel at %fs:0x8

    const size_t stack_size = 256 * 1024;
    char *stack = mmap(NULL, stack_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
    if (stack == MAP_FAILED) return 1;

    int flags = CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD |
                CLONE_SETTLS;
    // glibc clone(): fn, stack, flags, arg, parent_tid, tls, child_tid.
    int tid = clone(child_fn, stack + stack_size, flags, NULL, NULL, &child_tls,
                    NULL);
    if (tid < 0) return 2;

    for (int i = 0; i < 1000000 && child_saw != SENTINEL; i++) {
        sched_yield();
    }
    return child_saw == SENTINEL ? 0 : 3;
}
