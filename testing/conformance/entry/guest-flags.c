// RUN: %cc %s -o %t && %runner %t --help -n -- -f x; test $? = 42
//
// A guest's arguments must reach it verbatim even when they look like the
// sandbox's own options: a flag in the first position, and a literal `--`.
// The CLI must not consume, reorder, or choke on any of them.

#include <string.h>

int main(int argc, char **argv) {
    if (argc != 6) return 1;
    if (strcmp(argv[1], "--help") != 0) return 2;
    if (strcmp(argv[2], "-n") != 0) return 3;
    if (strcmp(argv[3], "--") != 0) return 4;
    if (strcmp(argv[4], "-f") != 0) return 5;
    if (strcmp(argv[5], "x") != 0) return 6;
    return 42;
}
