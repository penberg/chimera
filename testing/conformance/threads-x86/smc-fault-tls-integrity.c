// RUN: %cc %s -pthread -o %t && %runner %t
//
// Guest TLS must survive the runtime's self-modifying-code write trap. The
// trap's SIGSEGV handler interrupts translated code that may be running with
// the guest's FS base installed (the lazy per-residency `wrfsbase`), and the
// handler's own runtime code can reach libc stores through FS-relative TLS:
// under lock contention its mutex falls back to a `futex` system call whose
// wrapper writes `errno` on `EAGAIN`. Run with the guest's FS base, that
// store lands inside the *guest's* TLS block — this is how node's webpack
// build died: `errno = 11` overwrote the low half of a thread-local hash
// table's bucket pointer, and the first worker thread to exit crashed in the
// TLS destructor. The handler must swap its own FS base in before running
// any such code and restore the guest's on the way out.
//
// Each thread carpets its TLS block with a sentinel pattern large enough to
// cover any plausible host-`errno` offset below the thread pointer and reads
// it through `fs:` (behind a noinline call, so every trap really fires with
// the guest base installed). "Churn" threads rewrite and re-execute a
// page-sized straight-line function, so each round re-translates it with the
// runtime's address-space lock held for a long stretch; "trap" threads
// hammer one-trap-per-round stores on private pages, so their traps pile up
// on that lock. Any stray runtime store through the still-installed guest
// base lands in a sentinel — read back both by the next probe and by a final
// sweep — and fails the test. The `errno` race needs the unlock to land in a
// wait's syscall-entry window, so a run that exercises the path may still
// miss the write; the test is a canary that fails loudly whenever the race
// lands, and a deterministic regression check that the handler's FS swap
// keeps SMC recovery working under thread and translation pressure.
//
// Trap threads each own one 64-byte slot in the shared page holding
//   mov eax, imm32 ; ret
// and every round rewrite the immediate (the trap) and call it (the
// re-translation that re-arms the page for everyone). Churn threads own a
// private page filled with  mov eax, imm32 ; add eax, 1 × N ; ret.

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__x86_64__) && defined(__linux__)

enum { TRAPPERS = 8, CHURNERS = 2, CHURN_ROUNDS = 3000, SENTINEL_LEN = 0x400 };

// Big enough that the whole window below the thread pointer where the host
// libc's `errno` can live falls inside it, wherever the runtime's own TLS
// block ends.
static __thread unsigned char sentinel[SENTINEL_LEN];

static atomic_int churn_left;

typedef int (*fn)(void);

// The sentinel access must go through fs: on every call — a hoisted
// address-of-TLS computation would touch fs: once at loop entry and leave
// every later trap running on Chimera's own FS base, missing the bug — so
// the indexing lives behind a noinline boundary where the local-exec access
// compiles to a single fs:-relative load.
static unsigned char __attribute__((noinline)) probe(int i) {
    return sentinel[i];
}

static void *trap(void *arg) {
    unsigned char *code = arg;
    fn f = (fn) code;

    memset(sentinel, 0xA5, sizeof(sentinel));

    for (int i = 0; atomic_load_explicit(&churn_left, memory_order_relaxed) != 0;
         i++) {
        // Touch TLS first: the block that reads fs: installs the guest FS
        // base for this residency, so the trap below fires with it live. No
        // system calls in the loop, or the residency (and the installed
        // base) would be torn down before the store traps.
        if (probe(i & (SENTINEL_LEN - 1)) != 0xA5)
            return (void *) 2;
        // The store traps every round: the call just below re-translated
        // (and re-armed) this thread's private page.
        memcpy(code + 1, &i, 4);
        if (f() != i)
            return (void *) 3;
    }

    for (int i = 0; i < SENTINEL_LEN; i++)
        if (sentinel[i] != 0xA5)
            return (void *) 4;
    return 0;
}

static void *churn(void *arg) {
    unsigned char *code = arg;
    fn f = (fn) code;

    memset(sentinel, 0xA5, sizeof(sentinel));

    for (int i = 0; i < CHURN_ROUNDS; i++) {
        if (probe(i & (SENTINEL_LEN - 1)) != 0xA5)
            return (void *) 2;
        // The store traps (the previous call armed the page); the call then
        // re-translates the whole page-long block, holding the runtime's
        // address-space lock long enough for the trap threads to stack up
        // behind it.
        memcpy(code + 1, &i, 4);
        if (f() != i + (3800 / 3))
            return (void *) 3;
    }
    atomic_fetch_sub(&churn_left, 1);

    for (int i = 0; i < SENTINEL_LEN; i++)
        if (sentinel[i] != 0xA5)
            return (void *) 4;
    return 0;
}

int main(void) {
    alarm(120); // fail loudly rather than hang

    // Fragment the address space: tens of thousands of small VMAs make the
    // kernel side of the trap handler's `mprotect` slower and more variable,
    // so the handler's lock is held long enough for the other threads' traps
    // to really sleep on it — the regime where the `futex` wrappers see
    // contention (and, unfixed, write `errno` through the guest FS base).
    {
        size_t len = (size_t) 30000 * 4096;
        unsigned char *frag = mmap(0, len, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (frag != MAP_FAILED)
            for (size_t off = 0; off < len; off += 2 * 4096)
                mprotect(frag + off, 4096, PROT_READ);
    }

    unsigned char *trap_pages[TRAPPERS];
    for (int t = 0; t < TRAPPERS; t++) {
        unsigned char *code = mmap(0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (code == MAP_FAILED)
            return 10;
        code[0] = 0xB8; // mov eax, imm32
        code[5] = 0xC3; // ret
        trap_pages[t] = code;
    }

    unsigned char *churn_pages[CHURNERS];
    for (int t = 0; t < CHURNERS; t++) {
        unsigned char *code = mmap(0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (code == MAP_FAILED)
            return 10;
        code[0] = 0xB8; // mov eax, imm32
        size_t off = 5;
        for (int i = 0; i < 3800 / 3; i++) {
            code[off++] = 0x83; // add eax, 1
            code[off++] = 0xC0;
            code[off++] = 0x01;
        }
        code[off] = 0xC3; // ret
        churn_pages[t] = code;
    }

    atomic_store(&churn_left, CHURNERS);
    pthread_t trappers[TRAPPERS], churners[CHURNERS];
    for (int i = 0; i < CHURNERS; i++)
        if (pthread_create(&churners[i], 0, churn, churn_pages[i]) != 0)
            return 11;
    for (int i = 0; i < TRAPPERS; i++)
        if (pthread_create(&trappers[i], 0, trap, trap_pages[i]) != 0)
            return 11;

    int failed = 0;
    void *ret;
    for (int i = 0; i < CHURNERS; i++) {
        pthread_join(churners[i], &ret);
        if (ret != 0)
            failed = 1;
    }
    for (int i = 0; i < TRAPPERS; i++) {
        pthread_join(trappers[i], &ret);
        if (ret != 0)
            failed = 1;
    }
    return failed ? 12 : 0;
}

#else

int main(void) {
    return 0;
}

#endif
