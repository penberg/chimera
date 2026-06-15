// RUN: %cc %s -pthread -o %t && %runner %t
//
// A counting semaphore bounds how many threads may be inside a region at once.
// Twelve threads repeatedly sem_wait / hold / sem_post on a semaphore with 3
// permits; an instrumentation mutex tracks how many are inside. The occupancy
// must never exceed 3 (the semaphore really bounds entry) and must reach at
// least 2 (it genuinely admits several at once rather than acting like a mutex).
// sem_wait/sem_post ride a futex under Chimera. Exits 0 only if both hold.

#include <pthread.h>
#include <semaphore.h>

#define THREADS 12
#define PERMITS 3
#define ITERS 2000

static sem_t sem;
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
static int inside;
static int max_inside;
static int violations;

static void *worker(void *arg) {
    (void) arg;
    for (int i = 0; i < ITERS; i++) {
        sem_wait(&sem);

        pthread_mutex_lock(&mtx);
        inside++;
        if (inside > max_inside) max_inside = inside;
        if (inside > PERMITS) violations++;
        pthread_mutex_unlock(&mtx);

        for (volatile int s = 0; s < 4000; s++) { /* brief hold so threads overlap */
        }

        pthread_mutex_lock(&mtx);
        inside--;
        pthread_mutex_unlock(&mtx);

        sem_post(&sem);
    }
    return 0;
}

int main(void) {
    if (sem_init(&sem, 0, PERMITS) != 0) return 1;
    pthread_t t[THREADS];
    for (int i = 0; i < THREADS; i++) {
        if (pthread_create(&t[i], NULL, worker, NULL) != 0) return 2;
    }
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
    sem_destroy(&sem);

    if (violations != 0) return 3;  // occupancy exceeded the permit count
    if (max_inside < 2) return 4;   // semaphore admitted real concurrency
    return 0;
}
