// RUN: %cc %s -o %t && %runner %t
// UNSUPPORTED: darwin -- clone is a Linux thread primitive; Darwin uses bsdthread_create.
//
// clone3 argument validation must come from the kernel, not crash the runtime.
// A guest handing clone3 a bogus `clone_args` pointer must get EFAULT — the
// pointer cannot be dereferenced to check its flags, so the call has to be
// forwarded for the kernel to reject, exactly as it would natively. Likewise a
// struct smaller than CLONE_ARGS_SIZE_VER0 (64 bytes) must get EINVAL even
// when the memory itself is readable. A PROT_NONE page provides a
// deterministically unreadable address; an address in the middle of nowhere
// would risk landing in a real mapping.

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#define CLONE_ARGS_SIZE_VER0 64

int main(void) {
    // Unreadable pointer, valid size: EFAULT.
    void *guard = mmap(NULL, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (guard == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    long r = syscall(SYS_clone3, guard, (size_t) CLONE_ARGS_SIZE_VER0);
    if (r != -1 || errno != EFAULT) {
        fprintf(stderr, "clone3(PROT_NONE ptr): got r=%ld errno=%d, want EFAULT\n", r, errno);
        return 2;
    }

    // NULL pointer: EFAULT, not a synthesized success or a crash.
    r = syscall(SYS_clone3, NULL, (size_t) CLONE_ARGS_SIZE_VER0);
    if (r != -1 || errno != EFAULT) {
        fprintf(stderr, "clone3(NULL): got r=%ld errno=%d, want EFAULT\n", r, errno);
        return 3;
    }

    // Readable struct, undersized: EINVAL. The buffer even asks for CLONE_VM,
    // so an interceptor that read the flags before checking the size would be
    // caught taking the thread-creation path on a call the kernel rejects.
    uint64_t args[8];
    memset(args, 0, sizeof(args));
    args[0] = 0x100; /* CLONE_VM */
    r = syscall(SYS_clone3, args, (size_t) 8);
    if (r != -1 || errno != EINVAL) {
        fprintf(stderr, "clone3(size=8): got r=%ld errno=%d, want EINVAL\n", r, errno);
        return 4;
    }

    return 0;
}
