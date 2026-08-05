// RUN: %cc %s -o %t && %runner %t
//
// fcntl byte-range record locks on virtualized descriptors. SQLite
// serializes its database files with F_SETLK/F_OFD_SETLK — cargo's
// global-cache database is one — so a descriptor table that answers EINVAL
// breaks every SQLite user with SQLITE_IOERR_LOCK. Real kernel locking is
// observable without write access: an OFD read lock taken through one open
// description of a file is reported as a would-be conflict to a writer
// querying through another, with l_pid = -1 marking the owner as a
// description rather than a process; classic locks of the same process
// never conflict with themselves.

#define _GNU_SOURCE
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

static struct flock range(short type) {
    struct flock fl;
    memset(&fl, 0, sizeof(fl));
    fl.l_type = type;
    fl.l_whence = SEEK_SET;
    fl.l_start = 0;
    fl.l_len = 16;
    return fl;
}

int main(int argc, char **argv) {
    (void) argc;
    int fd1 = open(argv[0], O_RDONLY);
    int fd2 = open(argv[0], O_RDONLY);
    if (fd1 < 0 || fd2 < 0) return 1;

    // Classic POSIX read lock: take it, then query as a writer. The lock
    // belongs to this same process, so no conflict is reported.
    struct flock fl = range(F_RDLCK);
    if (fcntl(fd1, F_SETLK, &fl) != 0) return 2;
    fl = range(F_WRLCK);
    if (fcntl(fd1, F_GETLK, &fl) != 0) return 3;
    if (fl.l_type != F_UNLCK) return 4;
    fl = range(F_UNLCK);
    if (fcntl(fd1, F_SETLK, &fl) != 0) return 5;

    // OFD read lock through fd1: a writer query through fd2 — a separate
    // open description — must see it, owned by a description (l_pid == -1).
    fl = range(F_RDLCK);
    if (fcntl(fd1, F_OFD_SETLK, &fl) != 0) return 6;
    fl = range(F_WRLCK);
    if (fcntl(fd2, F_OFD_GETLK, &fl) != 0) return 7;
    if (fl.l_type != F_RDLCK) return 8;
    if (fl.l_pid != -1) return 9;

    // Release it: the writer query reports the range free again.
    fl = range(F_UNLCK);
    if (fcntl(fd1, F_OFD_SETLK, &fl) != 0) return 10;
    fl = range(F_WRLCK);
    if (fcntl(fd2, F_OFD_GETLK, &fl) != 0) return 11;
    if (fl.l_type != F_UNLCK) return 12;
    return 0;
}
