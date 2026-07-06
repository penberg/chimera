// RUN: %cc %s -pthread -o %t && %runner %t
//
// Heavy mutex contention with a conservation invariant. Eight threads perform
// many random one-unit transfers between sixteen accounts, each transfer taking
// the one mutex around a two-account read-modify-write. The sum of all balances
// must be exactly conserved: if the mutex ever failed to serialize the paired
// updates (a dropped LOCK prefix through translation, a lost wakeup on the
// contended slow path, or a code-cache race), a transfer would be torn and the
// total would drift. Threads are created once and loop internally, so this leans
// on the contended fast/slow paths without create/join churn.

#include <pthread.h>
#include <stdint.h>

#define THREADS 8
#define ACCOUNTS 16
#define ITERS 200000
#define START 1000

static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
static long acct[ACCOUNTS];

static void *worker(void *arg) {
    uint64_t r = (uint64_t) (uintptr_t) arg * 2654435761u + 12345u;
    for (int i = 0; i < ITERS; i++) {
        r = r * 6364136223846793005ull + 1442695040888963407ull; // LCG
        int a = (int) ((r >> 33) % ACCOUNTS);
        int b = (int) ((r >> 15) % ACCOUNTS);
        if (a == b) continue;
        pthread_mutex_lock(&lk);
        acct[a] -= 1;
        acct[b] += 1; // conserves the total
        pthread_mutex_unlock(&lk);
    }
    return 0;
}

int main(void) {
    for (int i = 0; i < ACCOUNTS; i++) acct[i] = START;

    pthread_t t[THREADS];
    for (long i = 0; i < THREADS; i++)
        if (pthread_create(&t[i], NULL, worker, (void *) i) != 0) return 1;
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);

    long total = 0;
    for (int i = 0; i < ACCOUNTS; i++) total += acct[i];
    return total == (long) ACCOUNTS * START ? 0 : 2;
}
