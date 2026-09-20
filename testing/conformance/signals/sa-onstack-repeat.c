// RUN: %cc %s -o %t && timeout 10 %runner %t
//
// An SA_ONSTACK handler runs on the alternate stack every time it is
// delivered, not just the first time. Delivering a signal switches stacks and
// blocks the signal for the handler's duration, and the handler's return
// undoes both -- but the return is an rt_sigreturn straight into the kernel,
// which a runtime that virtualizes signals never observes. One that tracks
// "on the alternate stack" or "currently blocked" by remembering what it did
// at delivery, rather than by re-reading what the kernel restored, gets stuck
// in the delivered state: the second raise is filtered out as blocked, or is
// delivered onto the interrupted stack instead of the alternate one.
//
// The handler raises nothing itself, so each delivery is a fresh one from the
// main flow. It checks both halves: that its own frame really is inside the
// registered stack, and that sigaltstack agrees by reporting SS_ONSTACK.
// Exits 0 only if all three deliveries ran, on the alternate stack, each time.

#define _GNU_SOURCE

#include <signal.h>
#include <stdio.h>
#include <unistd.h>

#define ALT_SIZE (256 * 1024)

static char alt[ALT_SIZE];
static volatile sig_atomic_t runs;
static volatile sig_atomic_t on_stack;
static volatile sig_atomic_t reported_onstack;

static void handler(int sig) {
    (void) sig;
    char probe;
    runs++;
    if (&probe >= alt && &probe < alt + sizeof alt) on_stack++;
    stack_t cur;
    if (sigaltstack(0, &cur) == 0 && (cur.ss_flags & SS_ONSTACK)) reported_onstack++;
}

int main(void) {
    stack_t ss;
    ss.ss_sp = alt;
    ss.ss_size = sizeof alt;
    ss.ss_flags = 0;
    if (sigaltstack(&ss, 0) != 0) {
        perror("sigaltstack");
        return 1;
    }

    struct sigaction sa;
    sa.sa_handler = handler;
    sa.sa_flags = SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGUSR1, &sa, 0) != 0) {
        perror("sigaction");
        return 1;
    }

    for (int i = 0; i < 3; i++) {
        if (raise(SIGUSR1) != 0) return 1;
    }

    if (runs != 3) {
        fprintf(stderr, "handler ran %d times, expected 3\n", runs);
        return 1;
    }
    if (on_stack != 3) {
        fprintf(stderr, "handler ran on the alternate stack %d of 3 times\n", on_stack);
        return 1;
    }
    if (reported_onstack != 3) {
        fprintf(stderr, "sigaltstack reported SS_ONSTACK %d of 3 times\n", reported_onstack);
        return 1;
    }
    return 0;
}
