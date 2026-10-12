// RUN: %cc %s -o %t && %runner %t %t.dir
//
// An entry the guest created lives in the filesystem's upper layer, which the
// runtime serves through a confined directory walk. A confined open adds the
// host's "no symlink anywhere" flag, and the caller's own O_NOFOLLOW must not
// survive beside it: Darwin rejects the pair with EINVAL, so every O_NOFOLLOW
// open of a guest-created entry failed — which is how Rust's remove_dir_all
// opens every directory it removes. Create a directory and a file, then open
// each with O_NOFOLLOW; a symlink opened the same way must still be refused.

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc != 2) return 10;
    char path[4096];
    if (mkdir(argv[1], 0755) != 0) return 1;
    int fd = open(argv[1], O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return 2;
    close(fd);

    snprintf(path, sizeof path, "%s/file", argv[1]);
    fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) return 3;
    close(fd);
    fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) return 4;
    close(fd);

    snprintf(path, sizeof path, "%s/link", argv[1]);
    if (symlink("file", path) != 0) return 5;
    errno = 0;
    if (open(path, O_RDONLY | O_NOFOLLOW) >= 0 || errno != ELOOP) return 6;
    return 0;
}
