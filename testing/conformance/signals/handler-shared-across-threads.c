// RUN: %cc %s -pthread -o %t && timeout 10 %runner %t
//
// Signal dispositions are process-wide (POSIX): a handler one thread installs
// with sigaction is in force for every thread, including a thread-directed
// signal delivered to a sibling. The main thread installs a SIGUSR1 handler,
// then pthread_kills a worker; the worker's handler must run on the worker
// rather than the signal taking its default action (Term) and killing the
// process.
//
// With a per-thread disposition table the worker started with SIGUSR1 at its
// default, so the delivered signal terminated the whole group — exactly how
// JSC's SIGPWR stop-the-world (a handler installed on the main thread, fired
// thread-directed at each mutator) brought the process down. The `timeout`
// turns a lost-signal hang into a failure too.

#include <pthread.h>
#include <signal.h>
#include <unistd.h>

static volatile sig_atomic_t handled;       // set by the worker's handler
static pthread_t worker_self;               // the thread the handler ran on

static void handler(int sig) {
    (void) sig;
    worker_self = pthread_self();
    handled = 1;
}

static void *worker(void *arg) {
    (void) arg;
    // Spin until our own handler fires. If the disposition were the per-thread
    // default, this signal would terminate the process instead of arriving.
    while (!handled)
        usleep(1000);
    return 0;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handler;
    if (sigaction(SIGUSR1, &sa, 0) != 0) return 1;

    pthread_t t;
    if (pthread_create(&t, 0, worker, 0) != 0) return 2;

    // Give the worker time to reach its spin loop, then direct the signal at it.
    usleep(50000);
    if (pthread_kill(t, SIGUSR1) != 0) return 3;

    if (pthread_join(t, 0) != 0) return 4;
    if (!handled) return 5;                       // the handler must have run
    if (!pthread_equal(worker_self, t)) return 6; // and on the worker thread
    return 0;
}
