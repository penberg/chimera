// RUN: %cc %s -o %t && %runner %t read
// RUN: test -z "$(ls -A "$XDG_STATE_HOME/chimera/fs" 2>/dev/null)"
// RUN: %runner %t write %t.scribble
// RUN: test -n "$(ls -A "$XDG_STATE_HOME/chimera/fs" 2>/dev/null)" || test -f %t.scribble
//
// Filesystem hygiene: a run that changes nothing leaves no residue — its
// fresh filesystem's delta is empty and the filesystem is removed on exit —
// while a run that writes keeps its filesystem. The suite runner points
// XDG_STATE_HOME at a per-test directory, so the shell can observe the
// filesystems that survive. Natively no filesystem exists either way; the
// write probe then satisfies the fallback file check instead.

#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "read") == 0) {
        struct stat st;
        return stat("/", &st) == 0 ? 0 : 1;
    }
    if (argc == 3 && strcmp(argv[1], "write") == 0) {
        int fd = open(argv[2], O_CREAT | O_WRONLY, 0644);
        if (fd < 0) return 2;
        if (write(fd, "kept", 4) != 4) return 3;
        return close(fd) == 0 ? 0 : 4;
    }
    return 10;
}
