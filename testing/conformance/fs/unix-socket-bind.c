// RUN: %cc %s -o %t && rm -rf %t.ws %t.d && mkdir %t.d && touch %t.d/taken
// RUN: CHIMERA_FS=%t.ws %runner %t serve %t.d
// RUN: %t verify %t.ws %t.d
//
// A Unix-domain socket's name is a path: bind creates the node, connect
// opens it, and both must see the same namespace as every other path call.
// Under the overlay the node lands in the session's upper layer — never on
// the host — a second bind of a live name is EADDRINUSE, unlink removes the
// name, the name can be bound again after, and a name the guest deleted from
// the host's view can be bound over. The overlay's upper paths are far
// longer than sun_path allows, which the runtime must get past without the
// guest noticing. The directory and `taken` exist on the host before the
// guest runs.

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include "portable.h"

static int bind_at(const char *path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof sa.sun_path) return -1;
    strcpy(sa.sun_path, path);
    if (bind(fd, (struct sockaddr *) &sa, sizeof sa) != 0) {
        int e = errno;
        close(fd);
        errno = e;
        return -1;
    }
    return fd;
}

static int connect_to(const char *path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    strcpy(sa.sun_path, path);
    if (connect(fd, (struct sockaddr *) &sa, sizeof sa) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int serve(const char *dir) {
    char sock[4096], sock2[4096], taken[4096];
    snprintf(sock, sizeof sock, "%s/sock", dir);
    snprintf(sock2, sizeof sock2, "%s/sock2", dir);
    snprintf(taken, sizeof taken, "%s/taken", dir);

    int srv = bind_at(sock);
    if (srv < 0) return 2;
    struct stat st;
    if (stat(sock, &st) != 0 || !S_ISSOCK(st.st_mode)) return 3;
    // The name is taken while the node exists.
    if (bind_at(sock) >= 0 || errno != EADDRINUSE) return 4;
    if (listen(srv, 1) != 0) return 5;

    pid_t pid = fork();
    if (pid < 0) return 6;
    if (pid == 0) {
        int c = connect_to(sock);
        if (c < 0) _exit(1);
        _exit(write(c, "ping", 4) == 4 ? 0 : 2);
    }
    int conn = accept(srv, NULL, NULL);
    if (conn < 0) return 7;
    char buf[8] = {0};
    if (read(conn, buf, sizeof buf) != 4 || memcmp(buf, "ping", 4) != 0) return 8;
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) return 9;
    close(conn);
    close(srv);

    // unlink removes the name, and the name is free to bind again.
    if (unlink(sock) != 0) return 10;
    if (stat(sock, &st) == 0 || errno != ENOENT) return 11;
    int again = bind_at(sock);
    if (again < 0) return 12;
    close(again);
    if (unlink(sock) != 0) return 13;

    // A regular file holds the name; once deleted, a socket may take it.
    if (bind_at(taken) >= 0 || errno != EADDRINUSE) return 15;
    if (unlink(taken) != 0) return 16;
    int over = bind_at(taken);
    if (over < 0) return 17;
    if (stat(taken, &st) != 0 || !S_ISSOCK(st.st_mode)) return 18;
    close(over);

    // Left bound for verify to locate.
    return bind_at(sock2) < 0 ? 14 : 0;
}

// Where did the nodes land? Under the overlay they are in the session's
// upper layer, absent from the host, and the host's `taken` is untouched;
// natively (or under --unsafe) they are on the host.
static int verify(const char *ws, const char *dir) {
    char sock2[4096], taken[4096], canon[PATH_MAX], probe[4096];
    snprintf(sock2, sizeof sock2, "%s/sock2", dir);
    snprintf(taken, sizeof taken, "%s/taken", dir);
    if (!resolved_dir_path(canon, sizeof canon, sock2)) return 40;
    snprintf(probe, sizeof probe, "%s/data%s", ws, canon);
    struct stat st;
    if (stat(probe, &st) == 0) {
        if (!S_ISSOCK(st.st_mode)) return 41;
        if (stat(sock2, &st) == 0 || errno != ENOENT) return 42;
        return stat(taken, &st) == 0 && S_ISREG(st.st_mode) ? 0 : 44;
    }
    if (stat(sock2, &st) != 0 || !S_ISSOCK(st.st_mode)) return 43;
    return stat(taken, &st) == 0 && S_ISSOCK(st.st_mode) ? 0 : 45;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "serve") == 0) return serve(argv[2]);
    if (argc == 4 && strcmp(argv[1], "verify") == 0) return verify(argv[2], argv[3]);
    return 50;
}
