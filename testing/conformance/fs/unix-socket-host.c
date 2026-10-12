// RUN: %cc %s -o %t && rm -f %t.sock %t.ready
// RUN: (%t serve %t.sock %t.ready &) && %runner %t connect %t.sock %t.ready
//
// A socket that lives on the host — bound by a native process outside the
// sandbox — is reachable from the guest by its path: connect resolves the
// name through the namespace to the lower layer the node lives in. The
// native server binds, announces readiness through a file, accepts one
// client and answers it; the guest waits for the announcement, connects,
// and reads the answer.

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

static int addr(struct sockaddr_un *sa, const char *path) {
    memset(sa, 0, sizeof *sa);
    sa->sun_family = AF_UNIX;
    if (strlen(path) >= sizeof sa->sun_path) return -1;
    strcpy(sa->sun_path, path);
    return 0;
}

static int serve(const char *sock, const char *ready) {
    // Never outlive the suite's patience, whatever the client does.
    alarm(15);
    unlink(sock);
    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un sa;
    if (srv < 0 || addr(&sa, sock) != 0) return 1;
    if (bind(srv, (struct sockaddr *) &sa, sizeof sa) != 0 || listen(srv, 1) != 0) return 2;
    int flag = open(ready, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (flag < 0) return 3;
    close(flag);
    int conn = accept(srv, NULL, NULL);
    if (conn < 0) return 4;
    int ok = write(conn, "host", 4) == 4;
    close(conn);
    close(srv);
    unlink(sock);
    return ok ? 0 : 5;
}

static int client(const char *sock, const char *ready) {
    struct timespec nap = {0, 50 * 1000 * 1000};
    int waited;
    for (waited = 0; access(ready, F_OK) != 0 && waited < 200; waited++) nanosleep(&nap, NULL);
    if (waited == 200) return 10;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un sa;
    if (fd < 0 || addr(&sa, sock) != 0) return 11;
    if (connect(fd, (struct sockaddr *) &sa, sizeof sa) != 0) return 12;
    char buf[8] = {0};
    if (read(fd, buf, sizeof buf) != 4 || memcmp(buf, "host", 4) != 0) return 13;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) return 20;
    if (strcmp(argv[1], "serve") == 0) return serve(argv[2], argv[3]);
    if (strcmp(argv[1], "connect") == 0) return client(argv[2], argv[3]);
    return 20;
}
