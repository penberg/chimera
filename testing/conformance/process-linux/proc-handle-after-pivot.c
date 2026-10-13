// RUN: %cc %s -o %t && %runner %t
//
// A descriptor opened on /proc before a pivot_root keeps naming that procfs
// afterwards, whatever the new root has at /proc. bwrap takes such a handle
// first thing, binds a read-only /proc into its new root, pivots, and then
// writes its uid_map through the old handle; a handle that silently follows
// the path into the new root lands on the read-only bind instead. Here the
// new root's /proc is an empty tmpfs, so a handle that followed the path
// finds no self/status at all.
//
// Namespaces may be unavailable or unprivileged; that is a pass, not a
// finding, as in execve-pivoted-root.

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <sys/mount.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

// Report which setup step failed and why: CI hosts restrict namespaces in
// ways a developer machine does not, and an exit code alone hides the errno.
static void fail(int code, const char *what) {
    fprintf(stderr, "%s: %m\n", what);
    _exit(code);
}

int main(void) {
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        int proc_fd = open("/proc", O_PATH | O_DIRECTORY);
        if (proc_fd < 0) fail(2, "open /proc");
        if (unshare(CLONE_NEWUSER | CLONE_NEWNS) != 0) {
            fprintf(stderr, "skip: unshare: %m\n");
            _exit(0);
        }
        if (mount(NULL, "/", NULL, MS_REC | MS_PRIVATE, NULL) != 0) {
            if (errno != EPERM && errno != EACCES) fail(3, "mount MS_PRIVATE /");
            fprintf(stderr, "skip: mount: %m\n");
            _exit(0);
        }
        if (mount("/", "/mnt", NULL, MS_BIND | MS_REC, NULL) != 0) fail(4, "mount MS_BIND /mnt");
        if (mount("tmpfs", "/mnt/proc", "tmpfs", 0, NULL) != 0) fail(5, "mount tmpfs /mnt/proc");
        if (chdir("/mnt") != 0) fail(6, "chdir");
        if (syscall(SYS_pivot_root, ".", "mnt") != 0) fail(7, "pivot_root");
        if (chdir("/") != 0) fail(8, "chdir");
        if (umount2("/mnt", MNT_DETACH) != 0) fail(9, "umount2");
        int fd = openat(proc_fd, "self/status", O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "openat(proc_fd, self/status): %m\n");
            _exit(11);
        }
        char buf[16];
        _exit(read(fd, buf, sizeof(buf)) > 0 ? 0 : 12);
    }
    int status;
    if (waitpid(pid, &status, 0) != pid) return 13;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 14;
}
