// RUN: %cc %s -o %t && rm -f %t.file && %runner %t %t.file
//
// open(2) masks the mode of a creating open to the permission bits, so the
// common idiom open(dst, O_CREAT, src_stat.st_mode) works even though the
// source's st_mode carries file-type bits (S_IFREG | 0644 = 0100644).
// libuv's uv_fs_copyfile opens the destination exactly that way, so a
// runtime that resolves opens through a stricter interface — openat2
// rejects unknown mode bits with EINVAL — must apply the kernel's masking
// itself or node's fs.copyFileSync (and everything above it, fs-extra,
// create-react-app) fails where the kernel succeeds. The created file's
// permission bits must equal the mode with the type bits stripped, less
// the umask, matching native behavior.

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc != 2) return 10;
    umask(022);
    int fd = open(argv[1], O_CREAT | O_TRUNC | O_WRONLY, S_IFREG | 0644);
    if (fd < 0) return 11;
    struct stat st;
    if (fstat(fd, &st) != 0) return 12;
    if ((st.st_mode & 07777) != 0644) return 13;
    if (close(fd) != 0) return 14;
    return 0;
}
