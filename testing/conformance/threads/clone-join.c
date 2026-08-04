// RUN: %cc %s -o %t && %runner %t
// UNSUPPORTED: darwin -- clone is a Linux thread primitive; Darwin uses bsdthread_create.
//
// The join mechanism, on which pthread_join is built. CLONE_PARENT_SETTID
// delivers the child's TID into a parent word at clone time; CLONE_CHILD_CLEARTID
// makes the runtime zero a child word and futex-wake it when the child exits, so
// a thread blocked in a futex on that word wakes. Chimera spawns the child on a
// host thread, so it must reproduce both itself. Exits 0 only if the parent saw
// its TID word set and then saw the child word cleared via the wake.

#define _GNU_SOURCE
#include <linux/futex.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static int child_word;  // CLONE_CHILD_CLEARTID target: zeroed + woken on exit.
static int parent_word; // CLONE_PARENT_SETTID target: child TID at clone time.

static int child_fn(void *arg) {
    (void) arg;
    // Exit just this thread; the runtime clears child_word and wakes its futex.
    __asm__ volatile("movl $60, %%eax\n\t" // __NR_exit
                     "xorl %%edi, %%edi\n\t"
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

    child_word = -1;
    int flags = CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD |
                CLONE_PARENT_SETTID | CLONE_CHILD_CLEARTID;
    // glibc clone(): fn, stack, flags, arg, parent_tid, tls, child_tid.
    int tid = clone(child_fn, stack + stack_size, flags, NULL, &parent_word, NULL,
                    &child_word);
    if (tid < 0) return 2;
    if (parent_word != tid) return 3; // CLONE_PARENT_SETTID delivered the TID

    // Join: block until the child's exit clears child_word and wakes us. Bounded
    // so a missing wake fails the test rather than hanging.
    struct timespec ts = {0, 50 * 1000 * 1000}; // 50ms
    for (int i = 0; i < 60 && child_word != 0; i++) {
        int v = child_word;
        if (v == 0) break;
        syscall(SYS_futex, &child_word, FUTEX_WAIT, v, &ts, NULL, 0);
    }
    return child_word == 0 ? 0 : 4;
}
