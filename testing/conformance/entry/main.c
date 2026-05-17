// RUN: %cc %s -o %t && %runner %t hello; test $? = 42

#include <string.h>

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    if (strcmp(argv[1], "hello") != 0) return 2;
    return 42;
}
