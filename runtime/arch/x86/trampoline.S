// Trampolines that cross between Chimera and translated guest code.
//
// Three entry points: `dispatch` is called from Rust with
// `(ctx, host_pc)` and jumps into the code cache; `exit_block` is the
// tail of every per-block exit stub and returns to the Rust caller of
// `dispatch` with `exit_kind = NORMAL`; `exit_syscall`
// does the same, but sets `exit_kind = SYSCALL` so the dispatcher knows to
// invoke the embedder's `SystemCalls` handler before re-entering the cache.
//
// All TS_* placeholders are field offsets into ThreadState, supplied by
// global_asm! via offset_of!. Update the struct and the asm picks up the new
// offsets at compile time — no magic numbers to keep in sync.

.text

// fn dispatch(ctx: *mut ThreadState /* rdi */, host_pc: u64 /* rsi */)
.globl dispatch
.type dispatch, @function
dispatch:
    // Save Chimera callee-saved registers on the Chimera stack.
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    // Stash Chimera state in ThreadState (reached via GS).
    mov gs:[{TS_CHIMERA_RSP}], rsp
    mov gs:[{TS_HOST_PC}], rsi
    // Install the guest's FS base. rax is free at this point.
    mov rax, [rdi + {TS_GUEST_FS}]
    wrfsbase rax
    // Restore guest FP/SIMD state from ts.fpstate. rdi still holds ctx.
    fxrstor64 [rdi + {TS_FPSTATE}]
    // Load guest GPRs from *ctx (rdi). rdi itself is loaded last because we
    // still need it as the base register.
    mov rbx, [rdi + {TS_RBX}]
    mov rcx, [rdi + {TS_RCX}]
    mov rdx, [rdi + {TS_RDX}]
    mov rsi, [rdi + {TS_RSI}]
    mov rbp, [rdi + {TS_RBP}]
    mov rsp, [rdi + {TS_RSP}]   // switch to the guest stack
    mov r8,  [rdi + {TS_R8}]
    mov r9,  [rdi + {TS_R9}]
    mov r10, [rdi + {TS_R10}]
    mov r11, [rdi + {TS_R11}]
    mov r12, [rdi + {TS_R12}]
    mov r13, [rdi + {TS_R13}]
    mov r14, [rdi + {TS_R14}]
    mov r15, [rdi + {TS_R15}]
    // rflags: push then popfq uses the guest stack briefly.
    push qword ptr [rdi + {TS_RFLAGS}]
    popfq
    // rax and rdi loaded last.
    mov rax, [rdi + {TS_RAX}]
    mov rdi, [rdi + {TS_RDI}]
    // Jump to host_pc, which we stashed in TS_HOST_PC.
    jmp qword ptr gs:[{TS_HOST_PC}]
.size dispatch, . - dispatch

// Continuation of every per-block exit stub (normal exit).
//
// Pre-conditions (established by the per-block stub):
//   - TS_RAX holds the guest's rax.
//   - TS_RIP holds the next guest program counter.
//   - The guest rsp still points into the guest stack.
.globl exit_block
.type exit_block, @function
exit_block:
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
    fxsave64 gs:[{TS_FPSTATE}]
    pushfq
    pop rax
    mov gs:[{TS_RFLAGS}], rax
    // Restore Chimera's FS base so Rust TLS works again.
    mov rax, gs:[{TS_CHIMERA_FS}]
    wrfsbase rax
    mov rsp, gs:[{TS_CHIMERA_RSP}]
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret
.size exit_block, . - exit_block

// Same as exit_block, but sets exit_kind = SYSCALL. The translator
// targets this trampoline from the exit stub of every block whose terminator
// is a `syscall` instruction.
//
// rcx, r11, and rflags are deliberately NOT saved here: the per-block syscall
// exit stub has already written their architectural post-syscall values
// (next_ip, the caller's rflags, and the rflags itself) into TS_RCX, TS_R11,
// and TS_RFLAGS. Saving them here would either clobber those values or — for
// pushfq — touch the guest stack, which a real syscall does not.
.globl exit_syscall
.type exit_syscall, @function
exit_syscall:
    mov gs:[{TS_RBX}], rbx
    // TS_RCX materialized by the per-block stub.
    mov gs:[{TS_RDX}], rdx
    mov gs:[{TS_RSI}], rsi
    mov gs:[{TS_RDI}], rdi
    mov gs:[{TS_RBP}], rbp
    mov gs:[{TS_RSP}], rsp
    mov gs:[{TS_R8}],  r8
    mov gs:[{TS_R9}],  r9
    mov gs:[{TS_R10}], r10
    // TS_R11 materialized by the per-block stub.
    mov gs:[{TS_R12}], r12
    mov gs:[{TS_R13}], r13
    mov gs:[{TS_R14}], r14
    mov gs:[{TS_R15}], r15
    fxsave64 gs:[{TS_FPSTATE}]
    // TS_RFLAGS materialized by the per-block stub.
    mov qword ptr gs:[{TS_EXIT_KIND}], {EXIT_KIND_SYSCALL}
    // Restore Chimera's FS base.
    mov rax, gs:[{TS_CHIMERA_FS}]
    wrfsbase rax
    mov rsp, gs:[{TS_CHIMERA_RSP}]
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret
.size exit_syscall, . - exit_syscall
