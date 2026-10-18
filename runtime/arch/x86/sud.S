// Entry to and exit from guest code running natively under syscall user
// dispatch.
//
// `chimera_sud_enter` is called from Rust with the thread's ThreadState (also
// the GS base). It loads the guest's register file and jumps to the guest's
// rip. It returns to its caller only through `chimera_sud_exit`, which a
// signal handler reaches by rewriting the interrupted context's rip and rsp
// (see `preempt` in sud.rs). The signal handler records why the guest stopped
// in TS_EXIT_KIND.
//
// Between `chimera_sud_enter_window` and `chimera_sud_enter_end` the thread is
// marked as in the guest while the register file is only partly loaded. A
// handler that interrupts the window abandons the entry rather than capturing
// the half-loaded registers, so the register file in ThreadState still
// describes the guest and the kernel can run it again.

.text

// fn chimera_sud_enter(ctx: *mut ThreadState /* rdi */)
.globl chimera_sud_enter
.type chimera_sud_enter, @function
chimera_sud_enter:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov gs:[{TS_CHIMERA_RSP}], rsp
    mov qword ptr gs:[{TS_EXIT_KIND}], {EXIT_KIND_BLOCK}
.globl chimera_sud_enter_window
chimera_sud_enter_window:
    mov byte ptr gs:[{TS_SUD_IN_GUEST}], 1
    // A signal recorded since the run loop last looked would otherwise wait
    // for the guest's next system call; the catcher sets exit_requested, and
    // from here on a new one finds the thread in the guest and preempts it.
    cmp dword ptr gs:[{TS_EXIT_REQUESTED}], 0
    jne 1f
    mov eax, 0xe7
    xor edx, edx
    xrstor64 [rdi + {TS_FPSTATE}]
    mov rax, [rdi + {TS_GUEST_FS}]
    wrfsbase rax
    mov rbx, [rdi + {TS_RBX}]
    mov rcx, [rdi + {TS_RCX}]
    mov rdx, [rdi + {TS_RDX}]
    mov rsi, [rdi + {TS_RSI}]
    mov rbp, [rdi + {TS_RBP}]
    mov r8,  [rdi + {TS_R8}]
    mov r9,  [rdi + {TS_R9}]
    mov r10, [rdi + {TS_R10}]
    mov r11, [rdi + {TS_R11}]
    mov r12, [rdi + {TS_R12}]
    mov r13, [rdi + {TS_R13}]
    mov r14, [rdi + {TS_R14}]
    mov r15, [rdi + {TS_R15}]
    // Restore rflags while rsp is still Chimera's stack, so the push does not
    // write into the guest's red zone.
    push qword ptr [rdi + {TS_RFLAGS}]
    popfq
    mov rsp, [rdi + {TS_RSP}]
    mov rax, [rdi + {TS_RAX}]
    mov rdi, [rdi + {TS_RDI}]
    // From here every system call outside Chimera's handlers traps.
    mov byte ptr gs:[{TS_SUD_SELECTOR}], 1
    jmp qword ptr gs:[{TS_RIP}]
1:
    mov byte ptr gs:[{TS_SUD_IN_GUEST}], 0
.globl chimera_sud_enter_end
chimera_sud_enter_end:
    jmp chimera_sud_exit
.size chimera_sud_enter, . - chimera_sud_enter

// Return from `chimera_sud_enter`. Entered with rsp at TS_CHIMERA_RSP, either
// from the abandoned entry above or from a signal handler's sigreturn, which
// leaves FS holding whatever it held when the signal arrived.
.globl chimera_sud_exit
.type chimera_sud_exit, @function
chimera_sud_exit:
    mov rax, gs:[{TS_CHIMERA_FS}]
    wrfsbase rax
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret
.size chimera_sud_exit, . - chimera_sud_exit

// The target of every rewritten system call. A rewritten site is `call rax`
// in place of `syscall`, so it lands at the system call number in the page-0
// trampoline and slides into the jump here, with the return address pushed on
// the guest's stack and every register as the guest left it. The hook leaves
// the guest the way a SIGSYS does, but without the signal: it saves the
// register file, rsp still pointing at the return address, and returns from
// `chimera_sud_enter` with TS_EXIT_KIND set to EXIT_KIND_REWRITTEN.
//
// Up to `chimera_sud_hook_leave` the hook only stores registers into
// ThreadState, so a handler that interrupts it captures the guest as usual.
// From there on the register file is complete except for rflags, the FP
// state, and, until `chimera_sud_hook_fs_saved`, the FS base, all still live,
// while rsp is Chimera's; a handler that interrupts that window completes the
// exit itself (see `preempt` in sud.rs).
.globl chimera_sud_hook
.type chimera_sud_hook, @function
chimera_sud_hook:
    mov gs:[{TS_RAX}], rax
    mov gs:[{TS_RBX}], rbx
    mov gs:[{TS_RCX}], rcx
    mov gs:[{TS_RDX}], rdx
    mov gs:[{TS_RSI}], rsi
    mov gs:[{TS_RDI}], rdi
    mov gs:[{TS_RBP}], rbp
    mov gs:[{TS_RSP}], rsp
    mov gs:[{TS_R8}],  r8
    mov gs:[{TS_R9}],  r9
    mov gs:[{TS_R10}], r10
    mov gs:[{TS_R11}], r11
    mov gs:[{TS_R12}], r12
    mov gs:[{TS_R13}], r13
    mov gs:[{TS_R14}], r14
    mov gs:[{TS_R15}], r15
.globl chimera_sud_hook_leave
chimera_sud_hook_leave:
    mov rsp, gs:[{TS_CHIMERA_RSP}]
    rdfsbase r11
    mov gs:[{TS_GUEST_FS}], r11
.globl chimera_sud_hook_fs_saved
chimera_sud_hook_fs_saved:
    mov r11, gs:[{TS_CHIMERA_FS}]
    wrfsbase r11
    mov byte ptr gs:[{TS_SUD_SELECTOR}], 0
    mov byte ptr gs:[{TS_SUD_IN_GUEST}], 0
.globl chimera_sud_hook_left
chimera_sud_hook_left:
    // Nothing since the guest's `call` has written rflags.
    pushfq
    pop qword ptr gs:[{TS_RFLAGS}]
    mov eax, 0xe7
    xor edx, edx
    xsaveopt64 gs:[{TS_FPSTATE}]
    // Run Chimera with the FP environment a signal's return would give it.
    fninit
    ldmxcsr [rip + chimera_sud_default_mxcsr]
    mov qword ptr gs:[{TS_EXIT_KIND}], {EXIT_KIND_REWRITTEN}
    jmp chimera_sud_exit
.size chimera_sud_hook, . - chimera_sud_hook

.pushsection .rodata
.balign 4
chimera_sud_default_mxcsr:
    .long 0x1f80
.popsection
