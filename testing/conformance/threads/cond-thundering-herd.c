// RUN: %cc %s -pthread -o %t && %runner %t
//
// Condition-variable thundering herd. Sixteen waiters block on one condvar; each
// round the controller bumps a round counter and pthread_cond_broadcast, which
// must wake every waiter exactly once — none lost, none doubled — across fifty
// rounds. Two condvars keep the directions separate (controller -> waiters via
// the round broadcast, waiters -> controller via an arrival signal), so a lost
// or coalesced wakeup on the futex slow path under the herd would leave the
// arrival count short and the controller (and the final tally) would not match.
// Threads are created once and rendezvous each round, avoiding create/join churn.

#include <pthread.h>

#define WAITERS 16
#define ROUNDS 50

static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv_round = PTHREAD_COND_INITIALIZER;   // controller -> waiters
static pthread_cond_t cv_arrived = PTHREAD_COND_INITIALIZER; // waiters -> controller
static int round_no;   // current round, bumped by the controller
static int arrived;    // waiters that have woken this round
static long woke_total; // total wakeups across all rounds

static void *waiter(void *arg) {
    (void) arg;
    int seen = 0;
    for (int r = 0; r < ROUNDS; r++) {
        pthread_mutex_lock(&m);
        while (round_no == seen) pthread_cond_wait(&cv_round, &m);
        seen = round_no;
        woke_total++;
        arrived++;
        pthread_cond_signal(&cv_arrived);
        pthread_mutex_unlock(&m);
    }
    return 0;
}

int main(void) {
    pthread_t t[WAITERS];
    for (int i = 0; i < WAITERS; i++)
        if (pthread_create(&t[i], NULL, waiter, NULL) != 0) return 1;

    for (int r = 1; r <= ROUNDS; r++) {
        pthread_mutex_lock(&m);
        arrived = 0;
        round_no = r;
        pthread_cond_broadcast(&cv_round); // wake the whole herd
        while (arrived < WAITERS) pthread_cond_wait(&cv_arrived, &m);
        pthread_mutex_unlock(&m);
    }

    for (int i = 0; i < WAITERS; i++) pthread_join(t[i], NULL);
    return woke_total == (long) WAITERS * ROUNDS ? 0 : 2;
}
