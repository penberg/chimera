// RUN: %cc %s -pthread -o %t && %runner %t
//
// exit_group must terminate a sibling that is pure computation — spinning in
// a fully linked guest loop that never issues a syscall and so never reaches
// a run-loop boundary on its own. The hard direction is a worker calling
// exit_group while the *main* thread spins: the process ends only when the
// main thread's run returns, so if the stop cannot pop it out of the code
// cache, the process outlives the exit_group forever where native dies
// instantly (caught here by the suite's per-test timeout). The stop request
// arms the spinner's safepoint slot, which the translated back-edge polls
// read.

#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

static volatile unsigned long counter;

static void *exit_after_nap(void *arg) {
    (void) arg;
    // Let the main thread's spin loop translate, link, and get hot: it has no
    // reason to leave the cache again.
    usleep(100 * 1000);
    syscall(SYS_exit_group, 0);
    return 0;
}

int main(void) {
    pthread_t t;
    if (pthread_create(&t, NULL, exit_after_nap, NULL) != 0) return 1;
    for (;;) counter++; // never reaches another syscall boundary
}
