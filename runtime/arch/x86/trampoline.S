// Trampolines that cross between Chimera and translated guest code.
//
// Three entry points: `dispatch` is called from Rust with
// `(ctx, host_pc)` and jumps into the code cache; `exit_block` is the
// tail of every per-block exit stub and returns to the Rust caller of
// `dispatch` with `exit_kind = NORMAL`; `exit_syscall`
// does the same, but sets `exit_kind = SYSCALL` so the dispatcher knows to
// invoke the embedder's `SystemCalls` handler before re-entering the cache.
//
// The file also defines `chimera_fetch_copy`, the fault-guarded copy the
// translator fetches its decode window through. It lives here rather than in
// Rust because its fault tolerance is an ABI contract with the fault handler:
// the handler recognizes a fault by rip range and resumes at a fixed label,
// which requires the faulting instruction and the fixup to sit at exported,
// compiler-opaque addresses.
//
// All TS_* placeholders are field offsets into ThreadState, supplied by
// global_asm! via offset_of!. Update the struct and the asm picks up the new
// offsets at compile time — no magic numbers to keep in sync.

.text

// fn dispatch(ctx: *mut ThreadState /* rdi */, host_pc: u64 /* rsi */)
//
// A signal caught while the run loop was between its pending-signal check
// and this entry sets TS_EXIT_REQUESTED (see `preempt`); the entry bows out
// before touching any state so the loop iterates and delivers. The check
// runs before anything is pushed, so the abort is a bare `ret`. A signal that
// lands anywhere inside this routine after the check is redirected by the
// preemption code instead, by aiming the pending jump at `exit_now`: through
// rsi while it still carries host_pc (before `dispatch_host_pc_stashed`) and
// through TS_HOST_PC once it has been stashed. Both labels, and
// `dispatch_end`, exist only for that classification.
.globl dispatch
.type dispatch, @function
dispatch:
    cmp dword ptr gs:[{TS_EXIT_REQUESTED}], 0
    jne 9f
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
.globl dispatch_host_pc_stashed
dispatch_host_pc_stashed:
    // Install neither the guest's FS base nor its FP/SIMD state here; both are
    // done lazily by the translator's per-block prologue. The Rust code that
    // ran since the last cache exit holds Chimera's FS base (for its TLS) and
    // has clobbered the physical vector registers, so a guest that uses TLS or
    // FP would need them installed first — but most residencies (a bare syscall
    // loop, an integer hot loop) touch neither, and an unconditional `wrfsbase`
    // plus a full 832-byte XRSTOR were the costliest instructions on the
    // syscall path. Defer both: leave FS as Chimera's, leave the vector
    // registers as they are, and clear the two "installed" flags. The first
    // block that reads guest TLS (`fs:`) loads the guest FS base; the first that
    // touches FP restores ts.fpstate; each sets its flag, and the exit
    // trampolines undo only what was installed. eax/edx are reloaded as guest
    // registers further down.
    mov dword ptr [rdi + {TS_FP_IN_REGS}], 0
    mov qword ptr [rdi + {TS_FS_IS_GUEST}], 0
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
9:
    ret
.globl dispatch_end
dispatch_end:
.size dispatch, . - dispatch

// Leave the cache immediately with the guest register file as it stands.
// The preemption code aims an interrupted thread here: every guest register
// is live in the host registers (rax included, which `exit_block` expects
// already stashed) and TS_RIP already names the guest PC to resume at, so
// this is a block exit taken from an arbitrary instruction boundary.
.globl exit_now
.type exit_now, @function
exit_now:
    mov gs:[{TS_RAX}], rax
    jmp exit_block
.size exit_now, . - exit_now

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
    // Save the guest's FP/SIMD state only if some block this residency actually
    // touched it — its prologue restored ts.fpstate and set TS_FP_IN_REGS. When
    // the flag is clear the physical vector registers hold no guest value and
    // ts.fpstate is already canonical, so the save is skipped entirely. rflags
    // is already captured above, so the flag-clobbering cmp/xor are harmless.
    // The save itself is XSAVEOPT (init/modified optimizations over plain XSAVE,
    // paired with the prologue's xrstor64); rax holds no live guest value here
    // (the exit stub stashed guest rax in gs:[TS_RAX]) and rdx is saved.
    cmp dword ptr gs:[{TS_FP_IN_REGS}], 0
    je 1f
    mov eax, 0xe7
    xor edx, edx
    xsaveopt64 gs:[{TS_FPSTATE}]
1:
    // Restore Chimera's FS base for Rust TLS, but only if a block this
    // residency installed the guest's (setting TS_FS_IS_GUEST). Otherwise FS
    // still holds Chimera's base and the wrfsbase is pure cost. The cmp
    // clobbers flags, already captured above.
    cmp qword ptr gs:[{TS_FS_IS_GUEST}], 0
    je 3f
    mov rax, gs:[{TS_CHIMERA_FS}]
    wrfsbase rax
3:
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
    // Save the guest's FP/SIMD state only if some block this residency touched
    // it (see exit_block). A bare getpid syscall never does, so its residency
    // skips the save entirely. No flags are captured here, so the cmp/xor are
    // harmless. rax is free (the exit stub stashed guest rax) and rdx is saved.
    cmp dword ptr gs:[{TS_FP_IN_REGS}], 0
    je 2f
    mov eax, 0xe7
    xor edx, edx
    xsaveopt64 gs:[{TS_FPSTATE}]
2:
    // TS_RFLAGS materialized by the per-block stub.
    mov qword ptr gs:[{TS_EXIT_KIND}], {EXIT_KIND_SYSCALL}
    // Restore Chimera's FS base only if a block installed the guest's (see
    // exit_block). The cmp clobbers flags; none are live here.
    cmp qword ptr gs:[{TS_FS_IS_GUEST}], 0
    je 4f
    mov rax, gs:[{TS_CHIMERA_FS}]
    wrfsbase rax
4:
    mov rsp, gs:[{TS_CHIMERA_RSP}]
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret
.size exit_syscall, . - exit_syscall

// Same as exit_block, but sets exit_kind = TRAP. The translator targets this
// trampoline from the exit stub of every block whose terminator is an `int3`
// breakpoint. Unlike `syscall`, `int3` clobbers no registers, so every guest
// register (including rcx, r11, and rflags) is saved exactly as in exit_block.
.globl exit_trap
.type exit_trap, @function
exit_trap:
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
    mov rsp, gs:[{TS_CHIMERA_RSP}]
    pushfq
    pop rax
    mov gs:[{TS_RFLAGS}], rax
    // Save the guest's FP/SIMD state only if some block this residency touched
    // it (see exit_block); when the flag is clear the vector registers hold no
    // guest value and an unconditional save would clobber the canonical
    // ts.fpstate with them.
    cmp dword ptr gs:[{TS_FP_IN_REGS}], 0
    je 5f
    mov eax, 0xe7
    xor edx, edx
    xsaveopt64 gs:[{TS_FPSTATE}]
5:
    mov qword ptr gs:[{TS_EXIT_KIND}], {EXIT_KIND_TRAP}
    // Restore Chimera's FS base only if a block installed the guest's (see
    // exit_block). The cmp clobbers flags; rflags is already captured above.
    cmp qword ptr gs:[{TS_FS_IS_GUEST}], 0
    je 6f
    mov rax, gs:[{TS_CHIMERA_FS}]
    wrfsbase rax
6:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret
.size exit_trap, . - exit_trap

// fn chimera_fetch_copy(dst: *mut u8 /* rdi */, src: *const u8 /* rsi */,
//                       len: usize /* rdx */) -> usize /* rax */
//
// Copy `len` bytes from an untrusted guest address to `dst`, returning how
// many bytes were copied: `len` when the whole range is readable, the
// byte-exact readable prefix when a fault cuts the copy short. The guarantee
// is the kernel's exception-table pattern: a faulting `rep movsb` leaves
// rcx/rsi/rdi reflecting exact progress, so the fault handler, on seeing a
// rip inside [chimera_fetch_copy_start, chimera_fetch_fixup), rewrites the
// rip to `chimera_fetch_fixup` and returns — execution resumes below the
// copy and `len - rcx` is the count. The recognition is a rip-range compare
// against these exported labels, so the primitive needs no per-thread arming
// and the handler stays within its no-TLS, no-allocation contract.
.globl chimera_fetch_copy
.type chimera_fetch_copy, @function
chimera_fetch_copy:
    mov rcx, rdx
    cld
.globl chimera_fetch_copy_start
chimera_fetch_copy_start:
    rep movsb
.globl chimera_fetch_fixup
chimera_fetch_fixup:
    mov rax, rdx
    sub rax, rcx
    ret
.size chimera_fetch_copy, . - chimera_fetch_copy
// fn guarded_copy(dst: *mut u8 /* rdi */, src: *const u8 /* rsi */,
//                 len: usize /* rdx */) -> i64
//
// Copy `len` bytes from unvalidated guest memory with plain loads — the
// kernel's `copy_from_user` exception-table pattern, in miniature. The fault
// handler recovers a `SIGSEGV`/`SIGBUS` whose rip lies inside this routine by
// redirecting the interrupted thread to `guarded_copy_fault`, which reports
// the failed copy. Returns 1 on a complete copy, 0 on a fault; a faulted copy
// may have written any readable prefix of `dst`.
.globl guarded_copy
.type guarded_copy, @function
guarded_copy:
    mov rcx, rdx
    rep movsb
    mov eax, 1
    ret
.globl guarded_copy_fault
guarded_copy_fault:
    xor eax, eax
    ret
.size guarded_copy, . - guarded_copy
.globl guarded_copy_end
guarded_copy_end:
