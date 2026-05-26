// execve replaces the image in place: the runtime must load and translate the
// new program, not hand control to it natively. The child branch only runs if
// the post-execve code went through the translator.
//
// RUN: %cc %s -o %t && %runner %t; test $? = 42

#include <string.h>
#include <unistd.h>

extern char **environ;

int main(int argc, char **argv) {
    if (argc == 1) {
        char *newargv[] = {argv[0], "child", 0};
        execve(argv[0], newargv, environ);
        return 1; // execve does not return on success
    }
    if (argc == 2 && strcmp(argv[1], "child") == 0)
        return 42;
    return 2;
}
