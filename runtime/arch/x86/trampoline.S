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
    // Restore guest extended FP/SIMD state from ts.fpstate via XRSTOR. FXRSTOR
    // would touch only x87+SSE and leave the upper 128 bits of every YMM (and
    // all ZMM/opmask) register untouched. EDX:EAX is the state-component mask
    // (0xe7 = x87|SSE|AVX|opmask|ZMM_Hi256|Hi16_ZMM), intersected with XCR0 by
    // the CPU. rdi still holds ctx; eax/edx are reloaded from ctx further down.
    mov eax, 0xe7
    xor edx, edx
    xrstor64 [rdi + {TS_FPSTATE}]
    // Load guest GPRs from *ctx (rdi). rdi and rsp are loaded last: rdi is
    // still needed as the base register, and rsp must stay on Chimera's stack
    // until after the rflags restore below.
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
    // Restore rflags via push/popfq while rsp is still Chimera's stack. Doing
    // this after switching to the guest stack would write [guest_rsp-8] and
    // clobber the guest's red zone (the 128 bytes below rsp the System V ABI
    // reserves for leaf functions).
    push qword ptr [rdi + {TS_RFLAGS}]
    popfq
    // Now switch to the guest stack and load the last registers. None of these
    // touch flags, so the rflags just restored survive to the jump.
    mov rsp, [rdi + {TS_RSP}]
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
    // Switch to Chimera's stack before capturing rflags: a pushfq on the guest
    // stack would write [guest_rsp-8] and clobber the guest's red zone. The
    // guest rsp is already saved in TS_RSP, and the moves above leave flags
    // (still the guest's) untouched. Capturing rflags must also precede the
    // XSAVE mask setup, whose `xor edx, edx` clobbers flags.
    mov rsp, gs:[{TS_CHIMERA_RSP}]
    pushfq
    pop rax
    mov gs:[{TS_RFLAGS}], rax
    // Save guest extended FP/SIMD state via XSAVE (see dispatch for why and
    // for the 0xe7 component mask). rax holds no live guest value here (the
    // exit stub already stashed guest rax in gs:[TS_RAX]) and rdx is saved.
    mov eax, 0xe7
    xor edx, edx
    xsave64 gs:[{TS_FPSTATE}]
    // Restore Chimera's FS base so Rust TLS works again.
    mov rax, gs:[{TS_CHIMERA_FS}]
    wrfsbase rax
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
    // Save guest extended FP/SIMD state via XSAVE (see dispatch for the 0xe7
    // component mask). rax is free (the exit stub stashed guest rax) and rdx
    // is already saved; both are clobbered by the mask setup. No flags are
    // captured here, so the flag-clobbering `xor` is harmless.
    mov eax, 0xe7
    xor edx, edx
    xsave64 gs:[{TS_FPSTATE}]
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
