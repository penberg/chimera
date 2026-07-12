// RUN: %cc %s -pthread -o %t && %runner %t
//
// Concurrent code-page writes while a sibling thread executes from the page.
// A concurrent JIT (JavaScriptCore compiling on worker threads) routinely
// stores into a page that holds code other threads are hot in — new functions,
// inline-cache data — without stopping the world; the stores land next to the
// executed bytes, never over them. A dynamic translator that invalidates its
// translations on such a write must drop and re-link them with the executing
// thread still running full speed out of its code cache: any non-atomic
// rewrite of reachable cache bytes is a window where the executor decodes a
// spliced instruction and computes garbage.
//
// One thread calls a function built in a JIT page in a tight loop, checking
// its result on every call; another hammers a data byte elsewhere in the same
// page, so every re-translation arms the page and the next store invalidates
// it again. The function's small internal loop gives the translation linked
// blocks and a back-edge inside the invalidated page, so each round severs and
// relinks real edges. Any torn decode shows up as a wrong return value (or a
// crash); the test fails loudly rather than hanging thanks to the alarm.
//
// The function, at offset 0 (data byte at offset 2048):
//   sum(n): eax = 0; do { eax += n; n -= 1 } while (n > 0); ret
// so sum(n) = n(n+1)/2 for n >= 1.

#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__x86_64__) && defined(__linux__)

typedef int (*fn)(int);

static unsigned char *page;
static atomic_int writer_done;

// mov eax, 0        ; b8 00 00 00 00
// .loop:
// add eax, edi      ; 01 f8
// sub edi, 1        ; 83 ef 01
// jg .loop          ; 7f f9
// ret               ; c3
static const unsigned char sum_code[] = {0xB8, 0x00, 0x00, 0x00, 0x00, 0x01,
                                         0xF8, 0x83, 0xEF, 0x01, 0x7F, 0xF9,
                                         0xC3};

static void *writer(void *arg) {
    (void) arg;
    for (int i = 0; i < 20000; i++) {
        page[2048] = (unsigned char) i; // same page as the code, not the code
        if ((i & 63) == 0)
            sched_yield();
    }
    atomic_store(&writer_done, 1);
    return 0;
}

// Several executors, so re-entry into a freshly invalidated or relinked block
// is always in flight on some core when the writer's store tears translations
// down.
static void *executor(void *arg) {
    (void) arg;
    fn sum = (fn) page;
    while (!atomic_load(&writer_done))
        for (int n = 1; n <= 8; n++)
            if (sum(n) != n * (n + 1) / 2)
                _exit(2); // a torn decode computed garbage
    return 0;
}

int main(void) {
    page = mmap(0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) return 1;
    memcpy(page, sum_code, sizeof sum_code);
    __builtin___clear_cache((char *) page, (char *) page + 4096);

    alarm(30); // a wedged executor fails the test instead of hanging it

    pthread_t w, x[4];
    if (pthread_create(&w, 0, writer, 0)) return 1;
    for (int i = 0; i < 4; i++)
        if (pthread_create(&x[i], 0, executor, 0)) return 1;

    for (int i = 0; i < 4; i++)
        if (pthread_join(x[i], 0)) return 1;
    if (pthread_join(w, 0)) return 1;
    return 0;
}

#else
#error "unsupported host for smc-concurrent-invalidate test"
#endif
