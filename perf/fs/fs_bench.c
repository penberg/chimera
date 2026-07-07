// Self-timing filesystem microbenchmarks for measuring the userspace VFS
// overhead. Each mode runs one filesystem operation in a tight loop and reports
// nanoseconds per operation; the self-timing (via perf.h) excludes process
// startup and one-time block translation, leaving the steady-state per-op cost.
//
//   fs_bench <stat|open|read> <iters> [path]
//
// Modes decompose where the VFS spends time:
//   stat -- stat(path) per op; exercises the path resolver (the per-component
//           walk that stats every component of the path).
//   open -- open(path, O_RDONLY) + close per op; resolver plus fd-table install.
//   read -- open once, then pread(fd, buf, 4096, 0) per op; isolates the per-fd
//           dispatch (File::pread, fd lookup, offset) with no path resolution.
//
// run.sh runs each mode native and under `chimera run` and reports the ratio.

#define _GNU_SOURCE
#include "../perf.h" // perf_now_ns(); same self-timing rationale as the suite

#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <stat|open|read> <iters> [path]\n", argv[0]);
        return 2;
    }
    const char *mode = argv[1];
    uint64_t iters = strtoull(argv[2], NULL, 0);
    const char *path = argc > 3 ? argv[3] : "/usr/lib/os-release";

    uint64_t t0, t1, ok = 0;
    char buf[4096];

    if (strcmp(mode, "stat") == 0) {
        struct stat st;
        t0 = perf_now_ns();
        for (uint64_t i = 0; i < iters; i++)
            if (stat(path, &st) == 0) ok++;
        t1 = perf_now_ns();
    } else if (strcmp(mode, "open") == 0) {
        t0 = perf_now_ns();
        for (uint64_t i = 0; i < iters; i++) {
            int fd = open(path, O_RDONLY);
            if (fd >= 0) { ok++; close(fd); }
        }
        t1 = perf_now_ns();
    } else if (strcmp(mode, "read") == 0) {
        int fd = open(path, O_RDONLY);
        if (fd < 0) { perror("open"); return 1; }
        t0 = perf_now_ns();
        for (uint64_t i = 0; i < iters; i++)
            if (pread(fd, buf, sizeof buf, 0) >= 0) ok++;
        t1 = perf_now_ns();
        close(fd);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        return 2;
    }

    printf("%-5s %8.1f ns/op  (n=%llu, ok=%llu)\n", mode,
           (double)(t1 - t0) / (double)iters, (unsigned long long)iters,
           (unsigned long long)ok);
    return 0;
}
