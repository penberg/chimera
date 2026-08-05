// fork() is clone() without CLONE_VM: the child is a copy-on-write duplicate of
// the whole process and, under Chimera, resumes in translated code, so the
// runtime forwards it (only CLONE_VM thread creation is refused). The child's
// exit status must travel back to the parent through waitpid, identically under
// Chimera and on the bare host, so this test carries no special requirement.
//
// RUN: %cc %s -o %t && %runner %t

#include <sys/wait.h>
#include <unistd.h>

int main(void) {
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        // Child: a little work, then an exit status the parent will read back.
        int acc = 0;
        for (int i = 0; i <= 8; i++) acc += i;
        _exit(acc); // 36
    }

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return 2;
    if (!WIFEXITED(status)) return 3;
    if (WEXITSTATUS(status) != 36) return 4;
    return 0;
}
