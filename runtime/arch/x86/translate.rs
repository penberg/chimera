//! Basic-block translator. Decodes a single guest basic block, copies the
//! straight-line prefix into the code cache (with RIP-relative operands
//! fixed up by `BlockEncoder`), and rewrites the terminator into a
//! "compute next guest PC, then exit to the dispatcher" sequence.

use std::ptr;

use iced_x86::{
    BlockEncoder, BlockEncoderOptions, Code, Decoder, DecoderOptions, FlowControl, Instruction,
    InstructionBlock, MemoryOperand, OpKind, Register,
};

use crate::Error;

const CACHE_SIZE: usize = 16 * 1024 * 1024;
const MAX_BLOCK_GUEST_BYTES: usize = 4096;

/// A bump-allocated RWX region into which `translate()` emits blocks.
pub struct CodeCache {
    base: *mut u8,
    size: usize,
    used: usize,
}

impl CodeCache {
    pub fn new() -> Result<Self, Error> {
        let p = unsafe {
            libc::mmap(
                ptr::null_mut(),
                CACHE_SIZE,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(Error::last_os_error("code cache mmap"));
        }
        Ok(Self {
            base: p as *mut u8,
            size: CACHE_SIZE,
            used: 0,
        })
    }

    fn next_pc(&self) -> u64 {
        (self.base as u64) + self.used as u64
    }

    fn emit(&mut self, bytes: &[u8]) -> Result<(), Error> {
        if self.used + bytes.len() > self.size {
            return Err(Error::CodeCacheExhausted);
        }
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(self.used), bytes.len());
        }
        self.used += bytes.len();
        Ok(())
    }
}

/// `gs:[disp]` with a 32-bit displacement, qword-sized.
fn gs_qword(disp: i64) -> MemoryOperand {
    MemoryOperand::new(
        Register::None,
        Register::None,
        1,
        disp,
        4,
        false,
        Register::GS,
    )
}

/// Translate one basic block starting at `guest_pc`. Returns the host PC at
/// which the translated block begins.
pub fn translate(
    cache: &mut CodeCache,
    guest_pc: u64,
    exit_tramp: u64,
    syscall_tramp: u64,
) -> Result<u64, Error> {
    let host_pc = cache.next_pc();
    let guest_bytes =
        unsafe { std::slice::from_raw_parts(guest_pc as *const u8, MAX_BLOCK_GUEST_BYTES) };
    let mut decoder = Decoder::with_ip(64, guest_bytes, guest_pc, DecoderOptions::NONE);
    let mut instrs = Vec::new();
    let mut instr = Instruction::default();

    loop {
        if !decoder.can_decode() {
            return Err(Error::Translate(format!(
                "decoder ran out of bytes at {:#x}",
                guest_pc
            )));
        }
        decoder.decode_out(&mut instr);
        if matches!(instr.flow_control(), FlowControl::Next) {
            instrs.push(instr);
            continue;
        }
        emit_terminator(&mut instrs, &instr, exit_tramp, syscall_tramp)?;
        break;
    }

    let block = InstructionBlock::new(&instrs, host_pc);
    let result = BlockEncoder::encode(64, block, BlockEncoderOptions::NONE)
        .map_err(|e| Error::Translate(format!("encode block at {:#x}: {}", guest_pc, e)))?;
    cache.emit(&result.code_buffer)?;
    Ok(host_pc)
}

/// Emit the appropriate exit sequence for a terminator instruction.
///
/// Every sequence ends with the next guest PC in `rax`, followed by the
/// common tail (`mov gs:[128], rax; movabs rax, exit_tramp; jmp rax`). The
/// guest's original `rax` value is preserved at `gs:[0]` before being
/// clobbered.
fn emit_terminator(
    instrs: &mut Vec<Instruction>,
    t: &Instruction,
    exit_tramp: u64,
    syscall_tramp: u64,
) -> Result<(), Error> {
    let next_ip = t.next_ip();
    // `syscall` is special: the instruction itself does *not* run. We save
    // the guest state (rax holds the syscall number, the args are in their
    // usual registers — exit_trampoline captures them all) and exit through
    // `syscall_exit_trampoline`, which signals the dispatcher to invoke the
    // embedder's `SystemCalls` handler before resuming. Checked by opcode
    // because iced reports `syscall`'s `flow_control()` as `Call`.
    //
    // x86-64 `syscall` architecturally clobbers `rcx` (with the address of
    // the instruction following `syscall`) and `r11` (with the caller's
    // `rflags`). Since the instruction never actually runs in the cache,
    // the translator has to synthesize these side effects into the guest
    // register slots so the guest sees them on resume. The companion
    // `syscall_exit_trampoline` deliberately leaves `gs:[16]`, `gs:[88]`,
    // and `gs:[136]` untouched, trusting these writes.
    //
    // Real `syscall` does not touch user memory; neither does this
    // emulation. To capture rflags we briefly switch `rsp` to Chimera's
    // own stack (`gs:[144]`), `pushfq`/`pop` there, and switch back to
    // the guest's rsp. The guest never sees a write below its own stack
    // pointer, so a guest with `rsp` parked on a guard page does not
    // spuriously fault on a translated syscall.
    if t.code() == Code::Syscall {
        emit_save_rax(instrs)?;

        // rcx <- next_ip
        emit_load_rax_imm(instrs, next_ip)?;
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_rm64_r64,
            gs_qword(16),
            Register::RAX,
        ))?);

        // Stash guest rsp (the trampoline later overwrites this slot with
        // the same value), switch to Chimera's stack, capture rflags into
        // rax via pushfq/pop, then restore guest rsp.
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_rm64_r64,
            gs_qword(56),
            Register::RSP,
        ))?);
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_r64_rm64,
            Register::RSP,
            gs_qword(144),
        ))?);
        instrs.push(Instruction::with(Code::Pushfq));
        emit_pop_rax(instrs)?;
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_r64_rm64,
            Register::RSP,
            gs_qword(56),
        ))?);

        // r11 <- rflags
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_rm64_r64,
            gs_qword(88),
            Register::RAX,
        ))?);
        // rflags slot <- rflags too, so the trampoline does not need to
        // pushfq on the guest stack to populate it.
        instrs.push(mkinstr(Instruction::with2(
            Code::Mov_rm64_r64,
            gs_qword(136),
            Register::RAX,
        ))?);

        // Reload next_ip for the exit tail, which stores it as the resumed
        // guest rip in `gs:[128]`.
        emit_load_rax_imm(instrs, next_ip)?;
        return emit_exit_tail(instrs, syscall_tramp);
    }
    match t.flow_control() {
        FlowControl::UnconditionalBranch => {
            let target = t.near_branch_target();
            emit_save_rax(instrs)?;
            emit_load_rax_imm(instrs, target)?;
        }
        FlowControl::ConditionalBranch => {
            let taken = t.near_branch_target();
            emit_cond_select(instrs, t.code(), taken, next_ip)?;
        }
        FlowControl::Call => {
            let target = t.near_branch_target();
            emit_save_rax(instrs)?;
            emit_load_rax_imm(instrs, next_ip)?;
            emit_push_rax(instrs)?;
            emit_load_rax_imm(instrs, target)?;
        }
        FlowControl::IndirectBranch => {
            emit_save_rax(instrs)?;
            emit_load_rax_from_op0(instrs, t)?;
        }
        FlowControl::IndirectCall => {
            // Reading the operand can use rax (if op0 is rax) or other regs.
            // Push the return address first; then load the target.
            emit_save_rax(instrs)?;
            emit_load_rax_imm(instrs, next_ip)?;
            emit_push_rax(instrs)?;
            // op0 might refer to rax, but we've already saved it to gs:[0]
            // and clobbered the live rax. Reload from gs:[0] before reading
            // op0 if it uses rax.
            if op0_reads_rax(t) {
                emit_load_rax_from_gs(instrs, 0)?;
            }
            emit_load_rax_from_op0(instrs, t)?;
        }
        FlowControl::Return => {
            emit_save_rax(instrs)?;
            emit_pop_rax(instrs)?;
        }
        other => {
            return Err(Error::Translate(format!(
                "unhandled terminator at {:#x}: flow_control={:?} code={:?}",
                t.ip(),
                other,
                t.code(),
            )));
        }
    }
    emit_exit_tail(instrs, exit_tramp)
}

fn emit_save_rax(instrs: &mut Vec<Instruction>) -> Result<(), Error> {
    instrs.push(mkinstr(Instruction::with2(
        Code::Mov_rm64_r64,
        gs_qword(0),
        Register::RAX,
    ))?);
    Ok(())
}

fn emit_load_rax_imm(instrs: &mut Vec<Instruction>, imm: u64) -> Result<(), Error> {
    instrs.push(mkinstr(Instruction::with2(
        Code::Mov_r64_imm64,
        Register::RAX,
        imm,
    ))?);
    Ok(())
}

fn emit_load_rax_from_gs(instrs: &mut Vec<Instruction>, disp: i64) -> Result<(), Error> {
    instrs.push(mkinstr(Instruction::with2(
        Code::Mov_r64_rm64,
        Register::RAX,
        gs_qword(disp),
    ))?);
    Ok(())
}

fn emit_push_rax(instrs: &mut Vec<Instruction>) -> Result<(), Error> {
    instrs.push(mkinstr(Instruction::with1(Code::Push_r64, Register::RAX))?);
    Ok(())
}

fn emit_pop_rax(instrs: &mut Vec<Instruction>) -> Result<(), Error> {
    instrs.push(mkinstr(Instruction::with1(Code::Pop_r64, Register::RAX))?);
    Ok(())
}

fn emit_load_rax_from_op0(instrs: &mut Vec<Instruction>, t: &Instruction) -> Result<(), Error> {
    match t.op0_kind() {
        OpKind::Register => {
            let src = t.op0_register();
            if src != Register::RAX {
                instrs.push(mkinstr(Instruction::with2(
                    Code::Mov_r64_rm64,
                    Register::RAX,
                    src,
                ))?);
            }
            // If src is rax, rax already holds the value (mov gs:[0], rax
            // didn't modify rax).
        }
        OpKind::Memory => {
            let memop = extract_memory_operand(t);
            instrs.push(mkinstr(Instruction::with2(
                Code::Mov_r64_rm64,
                Register::RAX,
                memop,
            ))?);
        }
        other => {
            return Err(Error::Translate(format!(
                "unexpected op0 kind in indirect branch: {:?}",
                other
            )));
        }
    }
    Ok(())
}

fn op0_reads_rax(t: &Instruction) -> bool {
    match t.op0_kind() {
        OpKind::Register => t.op0_register() == Register::RAX,
        OpKind::Memory => t.memory_base() == Register::RAX || t.memory_index() == Register::RAX,
        _ => false,
    }
}

fn extract_memory_operand(t: &Instruction) -> MemoryOperand {
    MemoryOperand::new(
        t.memory_base(),
        t.memory_index(),
        t.memory_index_scale(),
        t.memory_displacement64() as i64,
        if t.memory_displ_size() == 0 { 0 } else { 8 },
        t.is_broadcast(),
        t.segment_prefix(),
    )
}

/// For a conditional branch, select between `taken` and `fallthrough` with a
/// `cmov`, without ever touching the guest stack. A `push`/`pop` here would
/// write `[rsp-8]` and clobber the guest's red zone — the 128 bytes below
/// `rsp` that the System V ABI reserves for a leaf function's own use.
///
/// Instead, `taken` is stashed in the context's rbx slot (`gs:[8]`) and the
/// `cmov` reads it straight from memory. The guest's rbx *register* is never
/// touched, so it stays live; `exit_block` then saves that live rbx over the
/// slot on the way out, overwriting the scratch value. The flags set by the
/// block's compare survive up to the `cmov` because every instruction emitted
/// here (`mov` to memory, `movabs reg, imm64`) leaves flags untouched.
fn emit_cond_select(
    instrs: &mut Vec<Instruction>,
    jcc_code: Code,
    taken: u64,
    fallthrough: u64,
) -> Result<(), Error> {
    // gs:[8] is the rbx slot; see the `gs_qword(0)` rax slot in `emit_save_rax`.
    const RBX_SLOT: i64 = 8;
    let cmov = jcc_to_cmov(jcc_code)?;
    emit_save_rax(instrs)?;
    // gs:[rbx] <- taken, staged through rax (there is no `mov m64, imm64`).
    emit_load_rax_imm(instrs, taken)?;
    instrs.push(mkinstr(Instruction::with2(
        Code::Mov_rm64_r64,
        gs_qword(RBX_SLOT),
        Register::RAX,
    ))?);
    // rax <- fallthrough, then pull in `taken` from gs:[rbx] if the condition holds.
    emit_load_rax_imm(instrs, fallthrough)?;
    instrs.push(mkinstr(Instruction::with2(
        cmov,
        Register::RAX,
        gs_qword(RBX_SLOT),
    ))?);
    Ok(())
}

fn jcc_to_cmov(jcc: Code) -> Result<Code, Error> {
    Ok(match jcc {
        Code::Je_rel8_64 | Code::Je_rel32_64 => Code::Cmove_r64_rm64,
        Code::Jne_rel8_64 | Code::Jne_rel32_64 => Code::Cmovne_r64_rm64,
        Code::Ja_rel8_64 | Code::Ja_rel32_64 => Code::Cmova_r64_rm64,
        Code::Jae_rel8_64 | Code::Jae_rel32_64 => Code::Cmovae_r64_rm64,
        Code::Jb_rel8_64 | Code::Jb_rel32_64 => Code::Cmovb_r64_rm64,
        Code::Jbe_rel8_64 | Code::Jbe_rel32_64 => Code::Cmovbe_r64_rm64,
        Code::Jg_rel8_64 | Code::Jg_rel32_64 => Code::Cmovg_r64_rm64,
        Code::Jge_rel8_64 | Code::Jge_rel32_64 => Code::Cmovge_r64_rm64,
        Code::Jl_rel8_64 | Code::Jl_rel32_64 => Code::Cmovl_r64_rm64,
        Code::Jle_rel8_64 | Code::Jle_rel32_64 => Code::Cmovle_r64_rm64,
        Code::Jo_rel8_64 | Code::Jo_rel32_64 => Code::Cmovo_r64_rm64,
        Code::Jno_rel8_64 | Code::Jno_rel32_64 => Code::Cmovno_r64_rm64,
        Code::Js_rel8_64 | Code::Js_rel32_64 => Code::Cmovs_r64_rm64,
        Code::Jns_rel8_64 | Code::Jns_rel32_64 => Code::Cmovns_r64_rm64,
        Code::Jp_rel8_64 | Code::Jp_rel32_64 => Code::Cmovp_r64_rm64,
        Code::Jnp_rel8_64 | Code::Jnp_rel32_64 => Code::Cmovnp_r64_rm64,
        other => {
            return Err(Error::Translate(format!(
                "unsupported conditional branch: {:?}",
                other
            )));
        }
    })
}

fn emit_exit_tail(instrs: &mut Vec<Instruction>, exit_tramp: u64) -> Result<(), Error> {
    // mov gs:[128], rax
    instrs.push(mkinstr(Instruction::with2(
        Code::Mov_rm64_r64,
        gs_qword(128),
        Register::RAX,
    ))?);
    // movabs rax, exit_tramp
    emit_load_rax_imm(instrs, exit_tramp)?;
    // jmp rax
    instrs.push(mkinstr(Instruction::with1(Code::Jmp_rm64, Register::RAX))?);
    Ok(())
}

fn mkinstr(r: Result<Instruction, iced_x86::IcedError>) -> Result<Instruction, Error> {
    r.map_err(|e| Error::Translate(format!("build instruction: {}", e)))
}
