// RUN: %cc %s -pthread -o %t && %runner %t
//
// A condition variable carrying a bounded-queue producer/consumer. The producer
// sends 1..N through a 16-slot ring guarded by a mutex and two condvars
// (not_full / not_empty); the consumer receives each and sums it. The small
// queue forces both sides to block and be signalled repeatedly, so this leans on
// pthread_cond_wait (futex wait) and pthread_cond_signal (futex wake) under
// Chimera. If a wakeup were lost the test would hang; if an item were dropped or
// duplicated the sum would be wrong. Exits 0 only if the sum is exactly
// 1+2+...+N.

#include <pthread.h>

#define N 10000
#define QCAP 16

static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t not_empty = PTHREAD_COND_INITIALIZER;
static pthread_cond_t not_full = PTHREAD_COND_INITIALIZER;

static int queue[QCAP];
static int head, tail, count;
static long consumed_sum;

static void *producer(void *arg) {
    (void) arg;
    for (int i = 1; i <= N; i++) {
        pthread_mutex_lock(&mtx);
        while (count == QCAP) pthread_cond_wait(&not_full, &mtx);
        queue[tail] = i;
        tail = (tail + 1) % QCAP;
        count++;
        pthread_cond_signal(&not_empty);
        pthread_mutex_unlock(&mtx);
    }
    return 0;
}

static void *consumer(void *arg) {
    (void) arg;
    for (int i = 0; i < N; i++) {
        pthread_mutex_lock(&mtx);
        while (count == 0) pthread_cond_wait(&not_empty, &mtx);
        int v = queue[head];
        head = (head + 1) % QCAP;
        count--;
        consumed_sum += v;
        pthread_cond_signal(&not_full);
        pthread_mutex_unlock(&mtx);
    }
    return 0;
}

int main(void) {
    pthread_t p, c;
    if (pthread_create(&p, NULL, producer, NULL) != 0) return 1;
    if (pthread_create(&c, NULL, consumer, NULL) != 0) return 2;
    if (pthread_join(p, NULL) != 0) return 3;
    if (pthread_join(c, NULL) != 0) return 4;

    long expected = (long) N * (N + 1) / 2;
    return consumed_sum == expected ? 0 : 5;
}
