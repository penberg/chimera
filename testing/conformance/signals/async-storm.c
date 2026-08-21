// RUN: %cc -O2 %s -o %t && %runner %t
//
// A storm of asynchronous signals must leave the interrupted computation
// untouched. Chimera preempts a thread out of translated code at the exact
// host instruction a signal lands on and reconstructs the guest register state
// from whatever the translator had borrowed there — a parked rax across a call
// push, a scratch register across a far RIP-relative access, the flags across
// a lazy FP/TLS prologue, the target of an indirect branch mid-lookup. Each
// shape below is a loop that lives in one of those code regions, computed once
// quietly as the reference and once under a 200µs interval timer; the results
// must agree bit for bit, and every shape must actually have been interrupted.
// Also fails (via the suite's timeout) if any shape never gets its signal.

#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

static volatile sig_atomic_t hits;

static void on_alarm(int sig) {
    (void) sig;
    hits++;
}

// Linked conditional back-edge: a serial ALU chain.
static uint64_t shape_alu(void) {
    uint64_t acc = 0x9e3779b97f4a7c15ull;
    for (volatile uint64_t i = 0; i < 2000000; i++) {
        acc = acc * 6364136223846793005ull + 1442695040888963407ull;
        acc ^= acc >> 29;
    }
    return acc;
}

// Flags carried across the back-edge: a multi-limb add whose carry flows from
// one iteration's `adc` into the next, closed by a flag-preserving counter.
static uint64_t shape_carry(void) {
    static uint64_t a[256], b[256], r[256];
    uint64_t out = 0;
    for (int rep = 0; rep < 4000; rep++) {
        for (int i = 0; i < 256; i++) {
            a[i] = ~0ull - (uint64_t) rep;
            b[i] = (uint64_t) i + rep;
        }
        unsigned __int128 carry = 0;
        for (int i = 0; i < 256; i++) {
            carry += (unsigned __int128) a[i] + b[i];
            r[i] = (uint64_t) carry;
            carry >>= 64;
        }
        out = out * 31 + r[255] + (uint64_t) carry;
    }
    return out;
}

// FP/SIMD: the block prologue lazily restores the vector state; the loop keeps
// live values in xmm registers across the back-edge.
static uint64_t shape_fp(void) {
    double x = 1.0, y = 0.5;
    for (volatile int i = 0; i < 1500000; i++) {
        x = x * 1.0000001 + y;
        y = y * 0.9999999 - x * 1e-9;
    }
    uint64_t bits;
    memcpy(&bits, &x, sizeof bits);
    uint64_t ybits;
    memcpy(&ybits, &y, sizeof ybits);
    return bits ^ (ybits << 1);
}

// Guest TLS: every iteration touches `fs:`-relative storage, so the block
// prologue installs the guest FS base.
static __thread uint64_t tls_counter;
static uint64_t shape_tls(void) {
    tls_counter = 7;
    for (volatile int i = 0; i < 2000000; i++) {
        tls_counter = tls_counter * 3 + 1;
    }
    return tls_counter;
}

// Indirect branch: a computed-goto interpreter loop that closes through the
// inline indirect-branch lookup.
static uint64_t shape_indirect(void) {
    static const void *ops[] = {&&op_add, &&op_xor, &&op_rot, &&op_done};
    uint64_t acc = 1, n = 3000000;
    unsigned pc = 0;
    goto *ops[pc];
op_add:
    acc += 0x1234567;
    pc = 1;
    goto *ops[pc];
op_xor:
    acc ^= acc << 13;
    pc = 2;
    goto *ops[pc];
op_rot:
    acc = (acc << 7) | (acc >> 57);
    pc = --n ? 0 : 3;
    goto *ops[pc];
op_done:
    return acc;
}

// Call and return: direct calls link through a pushed return address, returns
// resolve through the lookup; both sit in the recipe regions.
static uint64_t __attribute__((noinline)) leaf(uint64_t x) {
    return x * 2654435761u + 1;
}
static uint64_t shape_calls(void) {
    uint64_t acc = 0;
    for (volatile int i = 0; i < 1500000; i++) {
        acc = leaf(acc);
    }
    return acc;
}

// Function-pointer calls: indirect call pushes the return address with the
// target parked in a context slot.
static uint64_t __attribute__((noinline)) leaf2(uint64_t x) {
    return (x ^ 0xdeadbeef) * 3;
}
static uint64_t shape_indirect_call(void) {
    uint64_t (*volatile fns[2])(uint64_t) = {leaf, leaf2};
    uint64_t acc = 5;
    for (volatile int i = 0; i < 1500000; i++) {
        acc = fns[i & 1](acc);
    }
    return acc;
}

// String instructions: large copies run as `rep movsb`/vector loops that a
// signal interrupts mid-instruction.
static uint64_t shape_memcpy(void) {
    static unsigned char src[1 << 16], dst[1 << 16];
    for (int i = 0; i < (1 << 16); i++) src[i] = (unsigned char) (i * 7);
    uint64_t acc = 0;
    for (volatile int i = 0; i < 600; i++) {
        memcpy(dst, src, sizeof dst);
        acc = acc * 131 + dst[(i * 977) & 0xffff] + dst[0xffff];
    }
    return acc;
}

typedef uint64_t (*shape_fn)(void);
static const struct {
    const char *name;
    shape_fn fn;
} shapes[] = {
    {"alu", shape_alu},
    {"carry", shape_carry},
    {"fp", shape_fp},
    {"tls", shape_tls},
    {"indirect", shape_indirect},
    {"calls", shape_calls},
    {"indirect-call", shape_indirect_call},
    {"memcpy", shape_memcpy},
};
#define NSHAPES (sizeof shapes / sizeof shapes[0])

static void timer(long usec) {
    struct itimerval it = {0};
    it.it_value.tv_usec = usec;
    it.it_interval.tv_usec = usec;
    setitimer(ITIMER_REAL, &it, 0);
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = on_alarm;
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGALRM, &sa, 0) != 0) return 1;

    uint64_t reference[NSHAPES];
    for (unsigned i = 0; i < NSHAPES; i++) reference[i] = shapes[i].fn();

    for (unsigned i = 0; i < NSHAPES; i++) {
        hits = 0;
        timer(200);
        uint64_t got = shapes[i].fn();
        timer(0);
        if (got != reference[i]) {
            write(2, "mismatch: ", 10);
            write(2, shapes[i].name, strlen(shapes[i].name));
            write(2, "\n", 1);
            return 10 + (int) i;
        }
        // Spin until the shape has been interrupted at least once, so a shape
        // too fast for the timer does not pass vacuously — and a delivery that
        // never comes hangs into the suite's timeout rather than passing.
        timer(200);
        while (hits == 0) {
            shapes[i].fn();
        }
        timer(0);
    }
    return 0;
}
