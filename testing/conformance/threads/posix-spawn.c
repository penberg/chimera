// RUN: %cc %s -o %t && %runner %t
//
// posix_spawn is glibc's fork+exec primitive, implemented on Linux via
// clone/clone3 with CLONE_VM|CLONE_VFORK (no CLONE_THREAD) followed by execve
// in the child. Chimera cannot run that child as a host thread -- its execve
// would replace the whole shared image and tear down the parent -- so it must
// emulate the spawn as a fork: a copy-on-write child that runs its file actions
// and execve into an independent process while the parent continues. The child
// is handed a stack (glibc stashes its entry there); Chimera must run the guest
// child on that stack, not the parent's.
//
// The parent spawns argv[0] again with a marker, redirecting the child's stdout
// to a pipe via a file action, then checks both the piped output and the exit
// status -- exercising the spawn, the dup2/close file actions, the execve of the
// spawned image, and status propagation.

#include <spawn.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

#define MARKER "spawned-child"
#define OUTPUT "hello-from-child\n"
#define CHILD_CODE 42

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], MARKER) == 0) {
        // The spawned image: prove we came up as our own process and that the
        // file action redirected our stdout, then exit with a known code.
        if (write(1, OUTPUT, sizeof(OUTPUT) - 1) != (ssize_t) (sizeof(OUTPUT) - 1))
            return 1;
        return CHILD_CODE;
    }

    int fds[2];
    if (pipe(fds) != 0) return 2;

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 1); // child stdout -> pipe
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_addclose(&fa, fds[1]);

    char *child_argv[] = {argv[0], (char *) MARKER, NULL};
    pid_t pid;
    int rc = posix_spawn(&pid, argv[0], &fa, NULL, child_argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return 3;

    close(fds[1]); // parent keeps only the read end

    char buf[64] = {0};
    ssize_t n = read(fds[0], buf, sizeof(buf) - 1);
    if (n != (ssize_t) (sizeof(OUTPUT) - 1) || strcmp(buf, OUTPUT) != 0) return 4;

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 5;
    if (!WIFEXITED(status)) return 6;
    if (WEXITSTATUS(status) != CHILD_CODE) return 7;
    return 0;
}
