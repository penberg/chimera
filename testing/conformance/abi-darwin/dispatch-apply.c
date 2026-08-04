// RUN: %cc %s -o %t && %runner %t
//
// Grand Central Dispatch runs a client's callback on its own worker threads —
// host threads a translator never created and has no context for. Left alone,
// the guest's code would run there natively: off the translator, on pages that
// are not executable, and outside the sandbox if they were. A translator must
// take those callbacks back and run them itself.
//
// `dispatch_apply` is the parallel-for, whose iterations the API allows an
// implementation to run on the calling thread. The queueing calls cannot be
// run inline like that — libdispatch's own bookkeeping notices — so the queue
// keeps the work and the thread, and gets a runtime shim in place of the
// guest's pointer; the callback still arrives on a worker thread, and is run
// there through the translator.

#include <dispatch/dispatch.h>
#include <stdint.h>

#define N 64

static uint32_t seen[N];
static int sync_ran, async_block, async_f, group_ran;

static void set_flag(void *p) { *(int *) p = 1; }

int main(void) {
    dispatch_queue_t q = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

    // Parallel-for: every index must run exactly once, and the writes must be
    // visible to whoever called dispatch_apply once it returns.
    dispatch_apply(N, q, ^(size_t i) {
        seen[i] = (uint32_t) (i + 1);
    });
    for (size_t i = 0; i < N; i++)
        if (seen[i] != (uint32_t) (i + 1)) return 1;

    // dispatch_sync runs its block on the calling thread, so it stays inside
    // the translator the way any ordinary call does.
    dispatch_sync(q, ^{
        sync_ran = 1;
    });
    if (!sync_ran) return 2;

    // Queued work: a block, and the function form that carries a context.
    // Both land on a queue's own worker thread.
    dispatch_queue_t serial = dispatch_queue_create("chimera.test", 0);
    dispatch_async(serial, ^{
        async_block = 1;
    });
    dispatch_async_f(serial, &async_f, set_flag);
    dispatch_sync(serial, ^{
    }); // a serial queue orders this after both
    if (!async_block || !async_f) return 3;

    // The group forms, which the system linker uses.
    dispatch_group_t group = dispatch_group_create();
    dispatch_group_async(group, q, ^{
        group_ran = 1;
    });
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    if (!group_ran) return 4;

    return 0;
}
