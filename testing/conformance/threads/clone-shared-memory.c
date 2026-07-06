// RUN: %cc %s -o %t && %runner %t
//
// clone(CLONE_VM): a new thread that shares the creator's address space.
// Chimera cannot forward such a clone to the host (the new task would run guest
// code natively, outside the translator); instead it spawns a host thread that
// runs the child guest in the shared address space and code cache. This test
// proves that happened: the child writes a global and the parent, on its own
// host thread, observes the write through the shared memory. Exits 0 only if
// the parent saw it.
//
// Deliberately minimal — no TLS, no join/futex (those come later). The child
// terminates itself with the raw exit syscall rather than returning (a return
// would run glibc's exit_group and tear the whole process down), and the parent
// rendezvouses by polling the shared word.

#define _GNU_SOURCE
#include <sched.h>
#include <sys/mman.h>

// Shared with the child via CLONE_VM. `volatile` so the parent reloads it from
// memory on each spin rather than hoisting the read out of the loop.
static volatile int shared;

static int child_fn(void *arg) {
    (void) arg;
    shared = 42;
    // Exit just this thread. Inline asm so the child touches no TLS (it has none
    // of its own here) and does not fall into glibc's process-wide exit.
    __asm__ volatile("movl $60, %%eax\n\t" // __NR_exit
                     "xorl %%edi, %%edi\n\t" // status 0
                     "syscall\n\t"
                     :
                     :
                     : "rax", "rdi", "rcx", "r11", "memory");
    return 0; // unreachable
}

int main(void) {
    const size_t stack_size = 256 * 1024;
    char *stack = mmap(NULL, stack_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
    if (stack == MAP_FAILED) return 1;

    int flags = CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD;
    int tid = clone(child_fn, stack + stack_size, flags, NULL);
    if (tid < 0) return 2;

    // Yield until the child's write lands (or give up after an ample bound).
    for (int i = 0; i < 1000000 && shared != 42; i++) {
        sched_yield();
    }
    return shared == 42 ? 0 : 3;
}
