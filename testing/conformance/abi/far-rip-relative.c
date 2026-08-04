// RUN: %cc -O2 %s -static -nostdlib -Wl,-Ttext-segment=0x35f50000000 -o %t && %runner %t; test $? = 42
// UNSUPPORTED: darwin -- x86 inline asm
//
// Guest code can sit tens of terabytes away from the translated-code cache --
// a JIT region placed at a randomized high address, or (as here) an image
// linked at one. A RIP-relative memory operand in such code cannot keep its
// effective address once the instruction moves into the cache: the
// displacement no longer fits in a rel32, and a translator that leans on the
// encoder's fixup fails the whole block. The translator must re-express the
// access through an absolute address instead. This links the text segment at
// 0x35f50000000 so every global access is such an operand, and exercises the
// three shapes that matter: plain loads and stores, an indirect call through a
// RIP-relative memory operand (the `call *fptr(%rip)` below), and a rewritten
// load sitting between a compare and the conditional branch that reads its
// flags, which a rewrite that clobbers the arithmetic flags would break.
//
// Freestanding (`-nostdlib`, own `_start`, raw `exit` syscall) because the crt
// startup objects use 32-bit absolute relocations that cannot express a text
// segment above 4 GiB.

static volatile long counter = 5;

static long flags_survive_rewrite(long a) {
    long r, x;
    __asm__ volatile("cmpq $7, %[a]\n\t"
                     "movq counter(%%rip), %[x]\n\t"
                     "jne 1f\n\t"
                     "movq $1, %[r]\n\t"
                     "jmp 2f\n\t"
                     "1:\n\t"
                     "movq $0, %[r]\n\t"
                     "2:"
                     : [r] "=r"(r), [x] "=r"(x)
                     : [a] "r"(a)
                     : "cc", "memory");
    return r;
}

static long add_counter(long n) {
    counter += n; /* RIP-relative load and store */
    return counter;
}

static long (*volatile fptr)(long) = add_counter;

static long call_through_rip_slot(long n) {
    long r;
    __asm__ volatile("movq %[n], %%rdi\n\t"
                     "callq *fptr(%%rip)\n\t"
                     "movq %%rax, %[r]"
                     : [r] "=r"(r)
                     : [n] "r"(n)
                     : "rax", "rdi", "rsi", "rdx", "rcx", "r8", "r9", "r10",
                       "r11", "cc", "memory");
    return r;
}

static void exit_with(long code) {
    __asm__ volatile("syscall" ::"a"(60L), "D"(code) : "rcx", "r11", "memory");
    __builtin_unreachable();
}

void _start(void) {
    if (add_counter(30) != 35)
        exit_with(1);
    if (call_through_rip_slot(7) != 42)
        exit_with(2);
    if (!flags_survive_rewrite(7) || flags_survive_rewrite(8))
        exit_with(3);
    exit_with(42);
}
