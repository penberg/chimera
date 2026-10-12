// RUN: %cc %s -o %t && %runner %t
//
// Thread-local storage initializers. A `_Thread_local` with a static
// initializer must read as that value on first access, and a zero-init one as
// 0. On Darwin this exercises the in-process linker's TLV setup: the
// `S_THREAD_LOCAL_REGULAR` template is concatenated into the initial-value
// image and the descriptor's first-touch thunk hands back the per-thread copy
// (`setup_tlv` / `chimera_tlv_get_addr`); `S_THREAD_LOCAL_ZEROFILL` backs the
// zero-init one. On Linux it is the ELF `.tdata`/`.tbss` model. Single-threaded
// on purpose, so it is a pure TLS-init check independent of thread spawn.

#include <string.h>

static _Thread_local int z;                       // zero-init
static _Thread_local int r = 0x5eed;              // scalar template
static _Thread_local long arr[4] = {1, 2, 3, 4};  // aggregate template
static _Thread_local char msg[8] = "tls-ok";      // string template

int main(void) {
    if (z != 0) return 1;
    if (r != 0x5eed) return 2;
    if (arr[0] != 1 || arr[1] != 2 || arr[2] != 3 || arr[3] != 4) return 3;
    if (strcmp(msg, "tls-ok") != 0) return 4;
    // The storage is writable and stable across accesses.
    r = 0x1234;
    z = 99;
    if (r != 0x1234 || z != 99) return 5;
    return 0;
}
