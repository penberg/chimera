// RUN: %cc %s -o %t && %runner %t
//
// A terminal reopened by its path — the way Bun wires stdin: ttyname the
// controlling terminal, open it O_RDONLY|O_NONBLOCK, read keystrokes from
// the new descriptor — must behave like a tty, not like a file: a read at
// a tracked offset is ESPIPE on a tty, so a virtualizing sandbox must fall
// back to plain read(2)/write(2) semantics for non-seekable files, and
// O_NONBLOCK must reach the real open file description (EAGAIN when empty,
// bytes when "typed"). Exits 0 only if the whole round trip works.

#define _XOPEN_SOURCE 700
#define _DEFAULT_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

int main(void) {
    // A pty pair stands in for the terminal.
    int master = posix_openpt(O_RDWR | O_NOCTTY);
    if (master < 0) return 1;
    if (grantpt(master) || unlockpt(master)) return 2;
    const char *slave_path = ptsname(master);

    // Bun's stdin pattern: reopen the tty by its path, nonblocking.
    int tty = open(slave_path, O_RDONLY | O_NONBLOCK | O_NOCTTY);
    if (tty < 0) { fprintf(stderr, "open: %m\n"); return 3; }

    // Nothing typed yet: nonblocking read must say EAGAIN, not ESPIPE.
    char buf[16];
    errno = 0;
    ssize_t n = read(tty, buf, sizeof(buf));
    if (!(n < 0 && errno == EAGAIN)) { fprintf(stderr, "empty read: n=%zd %m\n", n); return 4; }

    // "Type" into the master; the by-path fd must deliver the bytes.
    int wtty = open(slave_path, O_WRONLY | O_NOCTTY);
    if (wtty < 0) { fprintf(stderr, "wopen: %m\n"); return 5; }
    if (write(master, "hi\n", 3) != 3) return 6;
    usleep(100000);
    n = read(tty, buf, sizeof(buf));
    if (n < 3 || memcmp(buf, "hi", 2) != 0) { fprintf(stderr, "read: n=%zd %m\n", n); return 7; }

    // And writes through a by-path tty fd must reach the terminal.
    if (write(wtty, "ok\n", 3) != 3) { fprintf(stderr, "write: %m\n"); return 8; }
    n = read(master, buf, sizeof(buf));
    if (n < 3) { fprintf(stderr, "master read: n=%zd\n", n); return 9; }

    return 0;
}
