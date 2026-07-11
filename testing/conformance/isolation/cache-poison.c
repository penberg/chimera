// RUN: %cc %s -o %t && timeout 10 %runner %t
//
// A guest must never be able to smuggle a raw host syscall past the embedder's
// syscall handler by writing into Chimera's translated-code cache. The cache
// backs the guest's translated instructions with a host RWX mapping, so a naive
// design leaves it writable to the very guest code it runs; the guest can then
// scan /proc/self/maps for that mapping and rewrite an already-translated block
// in place. Chimera protects the cache with a per-thread protection key that
// denies guest writes while translated code runs, so any such store faults.
//
// The check is the sandbox escape itself, run in a child so its fault can't take
// the harness down. `victim()` carries a 2-byte NOP wedged in front of a movabs
// whose immediate is a unique 8-byte marker; the child translates it once, hunts
// the marker down in every RWX mapping, and rewrites the NOP into a `syscall`.
// If that store succeeds, re-running `victim()` executes a raw `write(2)` into a
// pipe the parent watches. Correct behavior is the opposite: either the store
// faults (under Chimera, the child dies) or there is no such mapping to patch
// (native, where the guest's own code is not writable) -- in both cases nothing
// reaches the pipe. Bytes on the pipe mean the escape worked, and the test fails.

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static const char raw_msg[] = "SANDBOX ESCAPED\n";

// The marker doubles as the movabs immediate below and as the needle the patcher
// searches for. Its little-endian encoding is what shows up in the code stream.
#define MARKER 0x1122334455667788ULL

__attribute__((noinline, noclone))
static void victim(void) {
    asm volatile(
        "mov $1, %%eax\n\t"                    // SYS_write on x86-64
        "mov $1, %%edi\n\t"                    // fd 1: the pipe, after dup2 below
        "lea raw_msg(%%rip), %%rsi\n\t"
        "mov %[len], %%edx\n\t"
        ".byte 0x66, 0x90\n\t"                 // NOP the patcher rewrites to syscall
        "movabs %[marker], %%r11\n\t"
        :
        : [len] "i"(sizeof(raw_msg) - 1), [marker] "i"(MARKER)
        : "rax", "rdi", "rsi", "rdx", "rcx", "r11", "memory");
}

static unsigned char *read_file(const char *path, size_t *out_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return NULL;
    }

    size_t cap = 16384;
    size_t len = 0;
    unsigned char *buf = malloc(cap + 1);
    if (!buf) {
        close(fd);
        return NULL;
    }

    for (;;) {
        if (len == cap) {
            cap *= 2;
            unsigned char *next = realloc(buf, cap + 1);
            if (!next) {
                free(buf);
                close(fd);
                return NULL;
            }
            buf = next;
        }
        ssize_t n = read(fd, buf + len, cap - len);
        if (n < 0) {
            free(buf);
            close(fd);
            return NULL;
        }
        if (n == 0) {
            break;
        }
        len += (size_t)n;
    }

    close(fd);
    buf[len] = 0;
    *out_len = len;
    return buf;
}

// Rewrite the `66 90` NOP that precedes the marker into a `0f 05` syscall,
// wherever the marker appears in [lo, hi). Only the translated copy of victim()
// carries both, so a match is Chimera's code cache and nothing else.
static int patch_rwx_mapping(uintptr_t lo, uintptr_t hi) {
    static const unsigned char marker[] = {
        0x49, 0xbb, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11
    };

    int patched = 0;
    for (uintptr_t p = lo + 2; p + sizeof(marker) <= hi; p++) {
        unsigned char *q = (unsigned char *)p;
        if (memcmp(q, marker, sizeof(marker)) != 0) {
            continue;
        }
        if (q[-2] != 0x66 || q[-1] != 0x90) {
            continue;
        }
        q[-2] = 0x0f;
        q[-1] = 0x05;
        patched++;
    }
    return patched;
}

static int patch_code_cache(void) {
    size_t maps_len = 0;
    unsigned char *maps = read_file("/proc/self/maps", &maps_len);
    if (!maps) {
        return 0;
    }

    int patched = 0;
    char *save = NULL;
    for (char *line = strtok_r((char *)maps, "\n", &save);
         line;
         line = strtok_r(NULL, "\n", &save)) {
        uintptr_t lo = 0;
        uintptr_t hi = 0;
        char perms[5] = {0};

        if (sscanf(line, "%lx-%lx %4s", &lo, &hi, perms) != 3) {
            continue;
        }
        if (strcmp(perms, "rwxp") != 0) {
            continue;
        }

        patched += patch_rwx_mapping(lo, hi);
    }

    free(maps);
    return patched;
}

// The cache guard is a protection key; a host without CPU PKU (or a kernel
// without CONFIG_PKEYS) leaves Chimera nothing to arm, so it warns and runs the
// cache unguarded. This test can only assert the guard where the guard can
// exist, mirroring an lit `REQUIRES: mpk`. The guest sees the host's flags
// through the passed-through /proc/cpuinfo.
static int host_has_pku(void) {
    size_t len = 0;
    unsigned char *info = read_file("/proc/cpuinfo", &len);
    if (!info) {
        return 0;
    }
    int found = strstr((char *)info, " pku") != NULL;
    free(info);
    return found;
}

int main(void) {
    int fds[2];
    if (pipe(fds) != 0) {
        return 1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        return 2;
    }
    if (pid == 0) {
        close(fds[0]);
        victim();          // translate victim() while it still holds the NOP
        patch_code_cache(); // rewrite the cached NOP into a syscall, or fault
        dup2(fds[1], 1);   // route the smuggled write(2) to the parent's pipe
        victim();          // vulnerable: runs the raw syscall; fixed: never here
        _exit(0);
    }

    close(fds[1]);
    char buf[64];
    ssize_t n = read(fds[0], buf, sizeof(buf));
    close(fds[0]);

    int status = 0;
    waitpid(pid, &status, 0);

    // Any bytes on the pipe are the smuggled write(2): the guest patched the
    // cache and ran a raw host syscall. That is the escape, and it is a failure
    // wherever Chimera can arm the guard -- i.e. wherever the host has
    // protection keys. Without them Chimera cannot enforce it, so the escape is
    // out of this test's scope, not a failure.
    if (n > 0) {
        return host_has_pku() ? 3 : 0;
    }
    return 0;
}
