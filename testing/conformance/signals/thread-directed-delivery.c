// RUN: %cc %s -o %t -pthread && %runner %t
//
// A thread-directed signal (pthread_kill) must be delivered ON the thread it
// targets, even while another thread is busy passing through delivery points.
// Chimera once kept one process-wide pending set: whichever thread reached a
// block boundary first consumed everyone's signals, so a sibling could steal a
// tgkill'd signal -- JavaScriptCore's SIGPWR suspend/resume protocol deadlocked
// exactly that way, the resume swallowed by a busy sibling while the suspended
// thread waited in sigsuspend holding the heap lock. The handler records which
// thread it ran on; it must be the targeted worker, never main. The runner's
// own timeout turns a lost delivery into a deterministic failure.

#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

#if defined(__APPLE__)
static long self_tid(void) {
    uint64_t tid = 0;
    pthread_threadid_np(0, &tid);
    return (long) tid;
}
#else
#include <sys/syscall.h>
static long self_tid(void) {
    return (long) syscall(SYS_gettid);
}
#endif

static atomic_long handler_tid = 0;
static atomic_int worker_ready = 0;

static void on_usr1(int sig) {
    (void) sig;
    atomic_store(&handler_tid, self_tid());
}

static void *worker(void *arg) {
    (void) arg;
    atomic_store(&worker_ready, 1);
    // Park in nanosleep so delivery interrupts a blocking syscall; loop in
    // case an unrelated wakeup arrives first.
    while (atomic_load(&handler_tid) == 0) {
        struct timespec ts = {0, 50 * 1000 * 1000};
        nanosleep(&ts, 0);
    }
    return 0;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = on_usr1;
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 1;

    pthread_t t;
    if (pthread_create(&t, 0, worker, 0) != 0) return 2;
    while (!atomic_load(&worker_ready))
        ;

    long worker_tid = 0;
    if (pthread_kill(t, SIGUSR1) != 0) return 3;

    // Main keeps taking syscalls (delivery points) while the signal is in
    // flight; with a shared pending set it would steal the delivery here.
    for (int i = 0; i < 1000000 && atomic_load(&handler_tid) == 0; i++)
        (void) getpid();

    if (pthread_join(t, 0) != 0) return 4;
    worker_tid = atomic_load(&handler_tid);
    if (worker_tid == 0) return 5; // never delivered
    if (worker_tid == self_tid()) return 6; // stolen by main
    return 0;
}
