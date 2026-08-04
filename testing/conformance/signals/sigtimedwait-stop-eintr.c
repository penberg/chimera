// RUN: %cc %s -o %t && %runner %t
// UNSUPPORTED: darwin -- Darwin has no sigtimedwait syscall.
//
// A default-disposition stop signal both stops the process and interrupts
// sigtimedwait() with EINTR: the kernel wakes the sleep to perform the stop,
// and once continued the wait reports EINTR rather than resuming — unlike the
// ignore-at-generation signals (SIGCHLD/SIGCONT/SIGURG/SIGWINCH), which the
// wait sleeps through. Wait on {SIGUSR1} while a child sends SIGTSTP, verifies
// the parent is actually stopped (/proc state 'T'), and continues it; the
// wait must return EINTR, not time out with EAGAIN. Exits 0 only if both the
// stop and the EINTR are observed.

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// The process state letter from /proc/<pid>/stat: the field after the
// parenthesized comm, 'T' when stopped by a job-control signal.
static char proc_state(pid_t pid) {
    char path[64], buf[256];
    snprintf(path, sizeof(path), "/proc/%d/stat", (int) pid);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return '?';
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) return '?';
    buf[n] = 0;
    char *p = strrchr(buf, ')');
    if (!p || p[1] != ' ') return '?';
    return p[2];
}

int main(void) {
    // The kernel discards SIGTSTP at delivery inside an orphaned process
    // group (nobody could ever SIGCONT it) — and a CI runner's whole step is
    // typically one orphaned group. Move into a fresh group whose parent (the
    // test harness) is in a different group of the same session, so this test
    // is never the orphaned case and job control works everywhere.
    if (setpgid(0, 0) != 0) return 7;

    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);

    pid_t parent = getpid();
    pid_t child = fork();
    if (child < 0) return 1;
    if (child == 0) {
        struct timespec delay = {0, 50 * 1000 * 1000}; // 50 ms: mid-wait
        nanosleep(&delay, 0);
        kill(parent, SIGTSTP);
        // The stop must actually take effect, not be swallowed on the way to
        // waking the wait. Poll briefly for the 'T' state before continuing.
        int stopped = 0;
        for (int i = 0; i < 40; i++) {
            nanosleep(&delay, 0);
            if (proc_state(parent) == 'T') {
                stopped = 1;
                break;
            }
        }
        kill(parent, SIGCONT);
        _exit(stopped ? 0 : 10);
    }

    struct timespec ts = {5, 0};
    errno = 0;
    int r = sigtimedwait(&set, 0, &ts);
    int saved = errno;

    int status;
    if (waitpid(child, &status, 0) != child) return 2;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        return 6; // the parent never stopped: the swallowed-stop regression
    if (r != -1) return 3;
    if (saved == EAGAIN) return 4; // slept through the stop: the wake regression
    if (saved != EINTR) return 5;

    return 0;
}
