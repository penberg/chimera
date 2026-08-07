// RUN: %cc %s -o %t && rm -f %t.err
// RUN: %runner %t 2> %t.err
// RUN: case "%runner" in *--unsafe*) : ;; *chimera*) grep -q "filesystem kept (no changes)" %t.err && grep -q -- "--in" %t.err && test -n "$(ls -A "$XDG_STATE_HOME/chimera/fs")" ;; *) : ;; esac
// RUN: case "%runner" in *--unsafe*) : ;; *chimera*) n=$(ls "$XDG_STATE_HOME/chimera/fs" | wc -l) && %runner --rm %t && test "$(ls "$XDG_STATE_HOME/chimera/fs" | wc -l)" -eq "$n" ;; *) : ;; esac
//
// Nothing is deleted implicitly: a run that changes nothing still keeps its
// filesystem — the badged prompt advertised the id for the whole session,
// and a branch may exist precisely to be somewhere to stand — and the kept
// notice both says the change-set is empty and hands back the `--in` line
// that resumes it. `--rm` is the sole discard: a discarded empty branch
// leaves the state directory exactly as it was. Native and --unsafe runs
// have no filesystem to speak of, so the checks apply only under the
// overlay.

#include <sys/stat.h>

int main(void) {
    struct stat st;
    return stat("/", &st) == 0 ? 0 : 1;
}
