// RUN: %cc %s -pthread -o %t && %runner %t
//
// Concurrent indirect-branch dispatch over the shared, lock-free inline
// indirect-branch table. Each thread runs the same data-dependent chain of
// indirect calls — picking one of several functions per step from a value that
// depends on the previous result — so distinct call and return PCs churn the
// one shared IB cache from several threads at once. Every thread must
// reproduce the single-threaded reference checksum exactly; a wrong-block jump
// from a torn read of a `{guest_pc, host_pc}` slot would corrupt it. A torn
// read needs a hash collision and a precise interleave, so this exercises the
// torn-read-safe publication path rather than deterministically reproducing a
// race.

#include <pthread.h>
#include <stdint.h>

#define THREADS 8
#define ITERS 3000000

typedef uint64_t (*op_t)(uint64_t);
static uint64_t op_add(uint64_t x) { return x + 0x9e3779b97f4a7c15ull; }
static uint64_t op_xor(uint64_t x) { return x ^ (x >> 17); }
static uint64_t op_mul(uint64_t x) { return x * 0xff51afd7ed558ccdull; }
static uint64_t op_rot(uint64_t x) { return (x << 13) | (x >> 51); }
static uint64_t op_sub(uint64_t x) { return x - 0x7f4a7c159e3779b9ull; }
static uint64_t op_not(uint64_t x) { return ~x; }
static uint64_t op_shl(uint64_t x) { return x + (x << 7); }
static uint64_t op_mix(uint64_t x) {
    x ^= x >> 33;
    return x * 0xc4ceb9fe1a85ec53ull;
}
static op_t ops[8] = {op_add, op_xor, op_mul, op_rot, op_sub, op_not, op_shl, op_mix};

static uint64_t run(uint64_t seed) {
    uint64_t x = seed;
    for (int i = 0; i < ITERS; i++) {
        op_t f = ops[x & 7]; // indirect call; target varies data-dependently
        x = f(x) + (uint64_t) i;
    }
    return x;
}

static uint64_t expected[THREADS];
static uint64_t got[THREADS];

static void *worker(void *arg) {
    long i = (long) arg;
    got[i] = run((uint64_t) i * 0x1234567u + 1);
    return 0;
}

int main(void) {
    for (long i = 0; i < THREADS; i++)
        expected[i] = run((uint64_t) i * 0x1234567u + 1); // single-threaded reference

    pthread_t t[THREADS];
    for (long i = 0; i < THREADS; i++)
        if (pthread_create(&t[i], NULL, worker, (void *) i) != 0) return 1;
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);

    for (int i = 0; i < THREADS; i++)
        if (got[i] != expected[i]) return 2;
    return 0;
}
