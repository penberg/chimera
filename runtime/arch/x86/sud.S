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
