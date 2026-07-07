// RUN: %cc %s -o %t && rm -rf %t.delta && CHIMERA_COW=%t.delta %runner %t
//
// The overlay over nothing is indistinguishable from the host: with a fresh,
// empty delta the guest must see the lower tree — read its own binary, stat
// directories, and list the directory it lives in. Every assertion is plain
// Linux semantics, so the test also passes natively (where CHIMERA_COW is
// just an ignored environment variable).

#include <dirent.h>
#include <fcntl.h>
#include <libgen.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
    (void)argc;

    // The binary itself reads through the overlay: ELF magic intact.
    int fd = open(argv[0], O_RDONLY);
    if (fd < 0) return 1;
    char magic[4];
    if (read(fd, magic, 4) != 4) return 2;
    if (memcmp(magic, "\x7f" "ELF", 4) != 0) return 3;
    close(fd);

    struct stat st;
    if (stat("/", &st) != 0 || !S_ISDIR(st.st_mode)) return 4;
    if (stat("/dev/null", &st) != 0 || !S_ISCHR(st.st_mode)) return 5;

    // The merged listing of the binary's own directory contains the binary.
    char path[4096];
    strncpy(path, argv[0], sizeof(path) - 1);
    path[sizeof(path) - 1] = 0;
    char *dir = dirname(path);
    char base[4096];
    strncpy(base, argv[0], sizeof(base) - 1);
    base[sizeof(base) - 1] = 0;
    char *name = basename(base);

    DIR *d = opendir(dir);
    if (!d) return 6;
    int found = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strcmp(e->d_name, name) == 0) found = 1;
    }
    closedir(d);
    return found ? 0 : 7;
}
