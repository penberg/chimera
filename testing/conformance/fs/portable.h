// The host spellings of the interfaces these tests reach for by name, so one
// source says the same thing on both hosts: Darwin's xattr calls carry a
// resource-fork position and an options word, its `struct stat` names its
// timestamps `st_*timespec`, its exclusive rename is `renamex_np`, and the
// running image is found through `_NSGetExecutablePath` rather than a `/proc`
// link. Include it after the system headers.

#ifndef CHIMERA_FS_PORTABLE_H
#define CHIMERA_FS_PORTABLE_H

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/xattr.h>
#include <unistd.h>

#ifdef __APPLE__

#include <mach-o/dyld.h>

#define getxattr(p, n, v, s) getxattr(p, n, v, s, 0, 0)
#define setxattr(p, n, v, s, f) setxattr(p, n, v, s, 0, f)
#define removexattr(p, n) removexattr(p, n, 0)
#define listxattr(p, b, s) listxattr(p, b, s, 0)
#define fgetxattr(fd, n, v, s) fgetxattr(fd, n, v, s, 0, 0)
#define fsetxattr(fd, n, v, s, f) fsetxattr(fd, n, v, s, 0, f)
#define fremovexattr(fd, n) fremovexattr(fd, n, 0)
#define flistxattr(fd, b, s) flistxattr(fd, b, s, 0)

// A missing attribute is ENOATTR here, where Linux says ENODATA.
#undef ENODATA
#define ENODATA ENOATTR

#define st_atim st_atimespec
#define st_mtim st_mtimespec

#define RENAME_NOREPLACE RENAME_EXCL
#define RENAME_EXCHANGE RENAME_SWAP

static inline int rename2(const char *from, const char *to, unsigned flags) {
    return renamex_np(from, to, flags);
}

static inline const char *self_exe(char *buf, size_t size) {
    uint32_t n = (uint32_t)size;
    return _NSGetExecutablePath(buf, &n) == 0 ? buf : NULL;
}

#else

#include <sys/syscall.h>
#include <sys/sysmacros.h>

static inline int rename2(const char *from, const char *to, unsigned flags) {
    return (int)syscall(SYS_renameat2, AT_FDCWD, from, AT_FDCWD, to, flags);
}

static inline const char *self_exe(char *buf, size_t size) {
    (void)buf;
    (void)size;
    return "/proc/self/exe";
}

#endif

// The upper entry for a guest path lives under `data/` at the path's
// *resolved* name: the namespace follows every symlink before a filesystem
// sees the path, so a test probing the delta for a path that passes through
// one (`/var` to `/private/var` on Darwin) must resolve the directory first.
// The entry itself need not exist on the host, so only its directory is
// resolved.
static inline const char *resolved_dir_path(char *out, size_t size, const char *path) {
    char dir[PATH_MAX], canon[PATH_MAX];
    const char *slash = strrchr(path, '/');
    if (!slash || (size_t)(slash - path) >= sizeof(dir)) return NULL;
    memcpy(dir, path, slash - path);
    dir[slash - path] = 0;
    if (!realpath(dir[0] ? dir : "/", canon)) return NULL;
    if (snprintf(out, size, "%s%s", canon, slash) >= (int)size) return NULL;
    return out;
}

#endif
