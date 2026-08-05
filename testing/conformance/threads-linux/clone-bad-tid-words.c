// RUN: %cc %s -o %t && %runner %t
//
// Bogus TID words must not crash the runtime — and must not fail the clone.
// The kernel's put_user for CLONE_PARENT_SETTID / CLONE_CHILD_SETTID is
// unchecked: a clone whose TID pointers are unmapped succeeds and the writes
// are silently skipped (verified natively). The exit-time
// CLONE_CHILD_CLEARTID store is equally best-effort: the kernel zeroes and
// futex-wakes the word if it can, and the thread exits either way. A guest
// aiming any of these at a PROT_NONE page therefore runs to completion.

#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

static volatile int settid_child_ran;
static volatile int cleartid_child_ran;
static char stack1[64 * 1024];
static char stack2[64 * 1024];

static int settid_child(void *arg) {
    (void) arg;
    settid_child_ran = 1;
    return 0;
}

static int cleartid_child(void *arg) {
    (void) arg;
    cleartid_child_ran = 1;
    return 0;
}

#define THREAD_FLAGS (CLONE_VM | CLONE_THREAD | CLONE_SIGHAND | CLONE_FS | CLONE_FILES)

int main(void) {
    long page = sysconf(_SC_PAGESIZE);
    char *none = mmap(NULL, page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (none == MAP_FAILED) return 1;

    // Set-TID words in a PROT_NONE page: the clone still succeeds and the
    // child still runs; the writes are skipped.
    int tid = clone(settid_child, stack1 + sizeof(stack1),
                    THREAD_FLAGS | CLONE_PARENT_SETTID | CLONE_CHILD_SETTID, NULL,
                    (pid_t *) none, NULL, (pid_t *) (none + 64));
    if (tid < 0) {
        fprintf(stderr, "clone with bad set-TID words failed\n");
        return 2;
    }

    // Clear-TID word in the PROT_NONE page: the child's exit-time store and
    // wake are best-effort; teardown must not crash.
    tid = clone(cleartid_child, stack2 + sizeof(stack2),
                THREAD_FLAGS | CLONE_CHILD_CLEARTID, NULL, NULL, NULL,
                (pid_t *) (none + 128));
    if (tid < 0) {
        fprintf(stderr, "clone with bad clear-TID word failed\n");
        return 3;
    }

    // Let both children run and fully exit (the clear-TID store happens at
    // exit); if either store went through a raw pointer, the process died
    // before reaching here.
    usleep(200 * 1000);
    if (!settid_child_ran || !cleartid_child_ran) {
        fprintf(stderr, "children did not run: settid=%d cleartid=%d\n",
                settid_child_ran, cleartid_child_ran);
        return 4;
    }
    return 0;
}
