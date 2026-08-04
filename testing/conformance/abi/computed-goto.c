// RUN: %cc -O2 -fPIE %s -pie -o %t && %runner %t
//
// A computed-goto ("labels as values") loop. With GCC on this toolchain, the
// `-O2 -fPIE -pie` shape emits `lea label(%rip)` for the label values and keeps
// those labels in the same straight-line block as the indirect jump that later
// consumes them. The translator must materialize the guest address; if it leaves
// the encoder to relocate the block-internal RIP-relative operand into the code
// cache, the indirect branch loses control flow and hangs. The runner's own
// per-RUN-line timeout turns that broken-runtime hang into a deterministic
// failure, so the RUN line needs no `timeout` utility (absent on macOS).

int main(void) {
    static void *tab[2];
    tab[0] = &&loop;
    tab[1] = &&done;
    unsigned long sum = 0;
    int i = 0;
loop:
    sum += (unsigned) i;
    i++;
    goto *tab[i > 1000];
done:
    return sum == (1000UL * 1001UL / 2UL) ? 0 : 1;
}
