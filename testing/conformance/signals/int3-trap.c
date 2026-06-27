// RUN: %cc %s -o %t && timeout 5 %runner %t
//
// A guest `int3` software breakpoint raises SIGTRAP. The translator does not run
// the breakpoint in the cache; it exits with a trap indication and the run loop
// raises SIGTRAP, which enters the guest's handler (and, with none, terminates
// the process with a faithful SIGTRAP status). JITs — JavaScriptCore in Bun —
// emit `int3` for traps and assertions, so an unhandled `int3` panicked the
// translator before. The handler must run and execution resume just past the
// breakpoint.

#include <signal.h>

static volatile sig_atomic_t ran;

static void handler(int sig) {
    (void) sig;
    ran = 1;
}

int main(void) {
    struct sigaction sa = {0};
    sa.sa_handler = handler;
    if (sigaction(SIGTRAP, &sa, 0) != 0) return 1;

    __asm__ volatile("int3");

    // Reached only if SIGTRAP was delivered to the handler and its return
    // resumed execution after the breakpoint.
    if (!ran) return 2;
    return 0;
}
