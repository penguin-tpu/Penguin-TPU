"""Instruction semantics for the Penguin scalar integer subset."""

from __future__ import annotations

from collections.abc import Callable

from .arch_state import ArchState, StopReason
from .instructions import (
    ALL_INSTRUCTION_SPECS,
    TENSOR_INSTRUCTION_SPECS,
    BType,
    DMAControlType,
    DMAType,
    DelayType,
    EmptyType,
    IType,
    JType,
    MXUAccumulatorType,
    ScaleImmType,
    ScaleMemType,
    MXUMatmulAccType,
    MXUMatmulType,
    RType,
    SType,
    TensorMemType,
    UType,
    VectorImmType,
    WeightTensorType,
    XLUUnaryType,
    VPUBinaryType,
    VPUUnaryType,
    instruction,
)
from .memory import DMA_CHANNEL_COUNT
from .tensor import (
    MATMUL_LATENCY_CYCLES,
    VLOAD_LATENCY_CYCLES,
    VMATPOP_ACC_BF16_LATENCY_CYCLES,
    VMATPOP_ACC_FP8_LATENCY_CYCLES,
    VMATPUSH_ACC_LATENCY_CYCLES,
    VMATPUSH_WEIGHT_LATENCY_CYCLES,
    VPU_NON_PIPELINEABLE_OP_LATENCY_CYCLES,
    VPU_SIMPLE_OP_LATENCY_CYCLES,
    VSTORE_LATENCY_CYCLES,
    XLU_TRANSPOSE_LATENCY_CYCLES,
    compute_bf16_transpose,
    compute_accum_matmul,
    compute_bf16_row_reduce_max,
    compute_bf16_row_reduce_sum,
    compute_bf16_vadd,
    compute_bf16_vredsum,
    compute_bf16_vexp,
    compute_bf16_vmax,
    compute_bf16_vmin,
    compute_bf16_vmov,
    compute_bf16_vrecip,
    compute_bf16_vsub,
    compute_bf16_vmul,
    compute_bf16_vrelu,
    compute_vector_immediate_fill,
)

MASK32 = 0xFFFF_FFFF
SCALAR_LOAD_MNEMONICS = frozenset({"lb", "lh", "lw", "lbu", "lhu"})
SCALAR_STORE_MNEMONICS = frozenset({"sb", "sh", "sw"})


def _u32(value: int) -> int:
    return value & MASK32


def _s32(value: int) -> int:
    value &= MASK32
    return value if value < 0x8000_0000 else value - 0x1_0000_0000


def _imm_shift(params: IType) -> int:
    return params.imm & 0x1F


def _sign_extend(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    mask = (1 << bits) - 1
    value &= mask
    return value | (MASK32 ^ mask) if value & sign_bit else value


def _branch_if(
    state: ArchState, params: BType, predicate: Callable[[int, int], bool]
) -> None:
    lhs = state.read_xreg(params.rs1)
    rhs = state.read_xreg(params.rs2)
    if predicate(lhs, rhs):
        state.set_next_pc(state.pc + params.imm)


def _dma_load_operands(state: ArchState, params: DMAType) -> tuple[int, int, int]:
    return (
        _u32(state.resolve_dma_base_address(params.rs1)),
        _u32(state.read_xreg(params.rd)),
        _u32(state.read_xreg(params.rs2)),
    )


def _dma_store_operands(state: ArchState, params: DMAType) -> tuple[int, int, int]:
    return (
        _u32(state.read_xreg(params.rd)),
        _u32(state.resolve_dma_base_address(params.rs1)),
        _u32(state.read_xreg(params.rs2)),
    )


def _load_mreg_pair(state: ArchState, base: int) -> tuple[object, object] | None:
    if not state.check_mreg_pair_base(base):
        return None
    return (state.load_mreg(base), state.load_mreg(base + 1))


def _store_mreg_pair(state: ArchState, base: int, lo: object, hi: object) -> None:
    if not state.check_mreg_pair_base(base):
        return
    state.store_mreg(base, lo)
    state.store_mreg(base + 1, hi)


def _scalar_load_address(state: ArchState, params: IType) -> int:
    return _u32(state.read_xreg(params.rs1) + params.imm)


@instruction(mnemonic="lui", params_type=UType, latency=1)
def lui(state: ArchState, params: UType) -> None:
    state.write_xreg(params.rd, params.imm << 12)


@instruction(mnemonic="auipc", params_type=UType, latency=1)
def auipc(state: ArchState, params: UType) -> None:
    state.write_xreg(params.rd, state.pc + (params.imm << 12))


@instruction(mnemonic="jal", params_type=JType, latency=1)
def jal(state: ArchState, params: JType) -> None:
    state.write_xreg(params.rd, state.pc + 1)
    state.set_next_pc(state.pc + params.imm)


@instruction(mnemonic="jalr", params_type=IType, latency=1)
def jalr(state: ArchState, params: IType) -> None:
    target = _u32(state.read_xreg(params.rs1) + params.imm)
    state.write_xreg(params.rd, state.pc + 1)
    state.set_next_pc(target)


@instruction(mnemonic="beq", params_type=BType, latency=1)
def beq(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: lhs == rhs)


@instruction(mnemonic="bne", params_type=BType, latency=1)
def bne(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: lhs != rhs)


@instruction(mnemonic="blt", params_type=BType, latency=1)
def blt(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: _s32(lhs) < _s32(rhs))


@instruction(mnemonic="bge", params_type=BType, latency=1)
def bge(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: _s32(lhs) >= _s32(rhs))


@instruction(mnemonic="bltu", params_type=BType, latency=1)
def bltu(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: _u32(lhs) < _u32(rhs))


@instruction(mnemonic="bgeu", params_type=BType, latency=1)
def bgeu(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: _u32(lhs) >= _u32(rhs))


@instruction(mnemonic="lb", params_type=IType, latency=1)
def lb(state: ArchState, params: IType) -> None:
    address = _scalar_load_address(state, params)
    state.write_xreg(params.rd, _sign_extend(state.load_vmem_u8(address), 8))


@instruction(mnemonic="lh", params_type=IType, latency=1)
def lh(state: ArchState, params: IType) -> None:
    address = _scalar_load_address(state, params)
    value = state.load_vmem_u16(address)
    if value is not None:
        state.write_xreg(params.rd, _sign_extend(value, 16))


@instruction(mnemonic="lw", params_type=IType, latency=1)
def lw(state: ArchState, params: IType) -> None:
    address = _scalar_load_address(state, params)
    value = state.load_vmem_u32(address)
    if value is not None:
        state.write_xreg(params.rd, value)


@instruction(mnemonic="lbu", params_type=IType, latency=1)
def lbu(state: ArchState, params: IType) -> None:
    address = _scalar_load_address(state, params)
    state.write_xreg(params.rd, state.load_vmem_u8(address))


@instruction(mnemonic="lhu", params_type=IType, latency=1)
def lhu(state: ArchState, params: IType) -> None:
    address = _scalar_load_address(state, params)
    value = state.load_vmem_u16(address)
    if value is not None:
        state.write_xreg(params.rd, value)


@instruction(mnemonic="seli", params_type=ScaleImmType, latency=1)
def seli(state: ArchState, params: ScaleImmType) -> None:
    state.write_ereg(params.ed, params.imm)


@instruction(mnemonic="seld", params_type=ScaleMemType, latency=1)
def seld(state: ArchState, params: ScaleMemType) -> None:
    address = _u32(state.read_xreg(params.rs1) + params.imm)
    state.write_ereg(params.ed, state.load_vmem_u8(address))


@instruction(mnemonic="sb", params_type=SType, latency=1)
def sb(state: ArchState, params: SType) -> None:
    address = _u32(state.read_xreg(params.rs1) + params.imm)
    state.store_vmem_u8(address, state.read_xreg(params.rs2))


@instruction(mnemonic="sh", params_type=SType, latency=1)
def sh(state: ArchState, params: SType) -> None:
    address = _u32(state.read_xreg(params.rs1) + params.imm)
    state.store_vmem_u16(address, state.read_xreg(params.rs2))


@instruction(mnemonic="sw", params_type=SType, latency=1)
def sw(state: ArchState, params: SType) -> None:
    address = _u32(state.read_xreg(params.rs1) + params.imm)
    state.store_vmem_u32(address, state.read_xreg(params.rs2))


@instruction(mnemonic="addi", params_type=IType, latency=1)
def addi(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) + params.imm)


@instruction(mnemonic="slti", params_type=IType, latency=1)
def slti(state: ArchState, params: IType) -> None:
    state.write_xreg(
        params.rd,
        1 if _s32(state.read_xreg(params.rs1)) < _s32(params.imm) else 0,
    )


@instruction(mnemonic="sltiu", params_type=IType, latency=1)
def sltiu(state: ArchState, params: IType) -> None:
    state.write_xreg(
        params.rd,
        1 if _u32(state.read_xreg(params.rs1)) < _u32(params.imm) else 0,
    )


@instruction(mnemonic="xori", params_type=IType, latency=1)
def xori(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) ^ params.imm)


@instruction(mnemonic="ori", params_type=IType, latency=1)
def ori(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) | params.imm)


@instruction(mnemonic="andi", params_type=IType, latency=1)
def andi(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) & params.imm)


@instruction(mnemonic="slli", params_type=IType, latency=1)
def slli(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) << _imm_shift(params))


@instruction(mnemonic="srli", params_type=IType, latency=1)
def srli(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, _u32(state.read_xreg(params.rs1)) >> _imm_shift(params))


@instruction(mnemonic="srai", params_type=IType, latency=1)
def srai(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, _s32(state.read_xreg(params.rs1)) >> _imm_shift(params))


@instruction(mnemonic="add", params_type=RType, latency=1)
def add(state: ArchState, params: RType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) + state.read_xreg(params.rs2))


@instruction(mnemonic="sub", params_type=RType, latency=1)
def sub(state: ArchState, params: RType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) - state.read_xreg(params.rs2))


@instruction(mnemonic="sll", params_type=RType, latency=1)
def sll(state: ArchState, params: RType) -> None:
    shift = state.read_xreg(params.rs2) & 0x1F
    state.write_xreg(params.rd, state.read_xreg(params.rs1) << shift)


@instruction(mnemonic="slt", params_type=RType, latency=1)
def slt(state: ArchState, params: RType) -> None:
    state.write_xreg(
        params.rd,
        1 if _s32(state.read_xreg(params.rs1)) < _s32(state.read_xreg(params.rs2)) else 0,
    )


@instruction(mnemonic="sltu", params_type=RType, latency=1)
def sltu(state: ArchState, params: RType) -> None:
    state.write_xreg(
        params.rd,
        1 if _u32(state.read_xreg(params.rs1)) < _u32(state.read_xreg(params.rs2)) else 0,
    )


@instruction(mnemonic="xor", params_type=RType, latency=1)
def xor(state: ArchState, params: RType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) ^ state.read_xreg(params.rs2))


@instruction(mnemonic="srl", params_type=RType, latency=1)
def srl(state: ArchState, params: RType) -> None:
    shift = state.read_xreg(params.rs2) & 0x1F
    state.write_xreg(params.rd, _u32(state.read_xreg(params.rs1)) >> shift)


@instruction(mnemonic="sra", params_type=RType, latency=1)
def sra(state: ArchState, params: RType) -> None:
    shift = state.read_xreg(params.rs2) & 0x1F
    state.write_xreg(params.rd, _s32(state.read_xreg(params.rs1)) >> shift)


@instruction(mnemonic="or", params_type=RType, latency=1)
def or_(state: ArchState, params: RType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) | state.read_xreg(params.rs2))


@instruction(mnemonic="and", params_type=RType, latency=1)
def and_(state: ArchState, params: RType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) & state.read_xreg(params.rs2))


@instruction(mnemonic="fence", params_type=EmptyType, latency=1)
def fence(state: ArchState, params: EmptyType) -> None:
    del state, params


@instruction(mnemonic="delay", params_type=DelayType, latency=1)
def delay(state: ArchState, params: DelayType) -> None:
    del state, params


@instruction(mnemonic="ecall", params_type=EmptyType, latency=1)
def ecall(state: ArchState, params: EmptyType) -> None:
    del params
    state.stop(StopReason.ECALL)


@instruction(mnemonic="ebreak", params_type=EmptyType, latency=1)
def ebreak(state: ArchState, params: EmptyType) -> None:
    del params
    state.stop(StopReason.EBREAK)


@instruction(
    mnemonic="vload",
    params_type=TensorMemType,
    latency=VLOAD_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vload(state: ArchState, params: TensorMemType) -> None:
    address = state.resolve_indirect_address(params.rs1, params.imm)
    state.vload(params.mreg, address)


@instruction(
    mnemonic="vstore",
    params_type=TensorMemType,
    latency=VSTORE_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vstore(state: ArchState, params: TensorMemType) -> None:
    address = state.resolve_indirect_address(params.rs1, params.imm)
    state.vstore(params.mreg, address)


@instruction(
    mnemonic="vmatpush.weight.mxu0",
    params_type=WeightTensorType,
    latency=VMATPUSH_WEIGHT_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatpush_weight_mxu0(state: ArchState, params: WeightTensorType) -> None:
    state.push_weight_slot_from_mreg(0, params.slot, params.ms)


@instruction(
    mnemonic="vmatpush.weight.mxu1",
    params_type=WeightTensorType,
    latency=VMATPUSH_WEIGHT_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatpush_weight_mxu1(state: ArchState, params: WeightTensorType) -> None:
    state.push_weight_slot_from_mreg(1, params.slot, params.ms)


@instruction(
    mnemonic="vmatpush.bf16.acc.mxu0",
    params_type=MXUAccumulatorType,
    latency=VMATPUSH_ACC_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatpush_acc_bf16_mxu0(state: ArchState, params: MXUAccumulatorType) -> None:
    state.push_accum_from_mregs(0, params.mreg)


@instruction(
    mnemonic="vmatpush.bf16.acc.mxu1",
    params_type=MXUAccumulatorType,
    latency=VMATPUSH_ACC_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatpush_acc_bf16_mxu1(state: ArchState, params: MXUAccumulatorType) -> None:
    state.push_accum_from_mregs(1, params.mreg)


@instruction(
    mnemonic="vmatpop.bf16.acc.mxu0",
    params_type=MXUAccumulatorType,
    latency=VMATPOP_ACC_BF16_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatpop_bf16_acc_mxu0(state: ArchState, params: MXUAccumulatorType) -> None:
    state.pop_accum_to_mregs(0, params.mreg)


@instruction(
    mnemonic="vmatpop.bf16.acc.mxu1",
    params_type=MXUAccumulatorType,
    latency=VMATPOP_ACC_BF16_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatpop_bf16_acc_mxu1(state: ArchState, params: MXUAccumulatorType) -> None:
    state.pop_accum_to_mregs(1, params.mreg)


@instruction(
    mnemonic="vmatpop.fp8.acc.mxu0",
    params_type=MXUAccumulatorType,
    latency=VMATPOP_ACC_FP8_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatpop_fp8_acc_mxu0(state: ArchState, params: MXUAccumulatorType) -> None:
    state.pop_accum_to_fp8_mreg(0, params.mreg)


@instruction(
    mnemonic="vmatpop.fp8.acc.mxu1",
    params_type=MXUAccumulatorType,
    latency=VMATPOP_ACC_FP8_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatpop_fp8_acc_mxu1(state: ArchState, params: MXUAccumulatorType) -> None:
    state.pop_accum_to_fp8_mreg(1, params.mreg)


def _apply_vpu_simple_latency(state: ArchState) -> None:
    state.instruction_extra_cycles = (
        state.config.vpu_simple_op_latency_cycles - VPU_SIMPLE_OP_LATENCY_CYCLES
    )


def _apply_vpu_non_pipelineable_latency(state: ArchState) -> None:
    state.instruction_extra_cycles = (
        state.config.vpu.non_pipelineable_op_latency_cycles
        - VPU_NON_PIPELINEABLE_OP_LATENCY_CYCLES
    )


def _vpu_binary_result(
    state: ArchState,
    params: VPUBinaryType,
    op: Callable[[object, object], object],
) -> None:
    lhs = state.load_mreg(params.ms1)
    rhs = state.load_mreg(params.ms2)
    _apply_vpu_simple_latency(state)
    state.store_mreg(params.md, op(lhs, rhs))


def _vpu_unary_result(
    state: ArchState,
    params: VPUUnaryType,
    op: Callable[[object], object],
    *,
    non_pipelineable: bool = False,
) -> None:
    src = state.load_mreg(params.ms)
    if non_pipelineable:
        _apply_vpu_non_pipelineable_latency(state)
    else:
        _apply_vpu_simple_latency(state)
    state.store_mreg(params.md, op(src))


def _apply_xlu_transpose_latency(state: ArchState) -> None:
    state.instruction_extra_cycles = (
        state.config.xlu_transpose_latency_cycles - XLU_TRANSPOSE_LATENCY_CYCLES
    )


@instruction(
    mnemonic="vmatmul.mxu0",
    params_type=MXUMatmulType,
    latency=MATMUL_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatmul_mxu0(state: ArchState, params: MXUMatmulType) -> None:
    state.instruction_extra_cycles = (
        state.config.matmul_latency_cycles - MATMUL_LATENCY_CYCLES
    )
    state.store_accum_buffer(
        0,
        compute_accum_matmul(
            state.load_mreg(params.ms),
            state.load_weight_slot(0, params.ws),
            scale_raw=state.read_ereg(0),
            config=state.config,
        ),
    )


@instruction(
    mnemonic="vmatmul.mxu1",
    params_type=MXUMatmulType,
    latency=MATMUL_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatmul_mxu1(state: ArchState, params: MXUMatmulType) -> None:
    state.instruction_extra_cycles = (
        state.config.matmul_latency_cycles - MATMUL_LATENCY_CYCLES
    )
    state.store_accum_buffer(
        1,
        compute_accum_matmul(
            state.load_mreg(params.ms),
            state.load_weight_slot(1, params.ws),
            scale_raw=state.read_ereg(0),
            config=state.config,
        ),
    )


@instruction(
    mnemonic="vmatmul.acc.mxu0",
    params_type=MXUMatmulAccType,
    latency=MATMUL_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatmul_acc_mxu0(state: ArchState, params: MXUMatmulAccType) -> None:
    state.instruction_extra_cycles = (
        state.config.matmul_latency_cycles - MATMUL_LATENCY_CYCLES
    )
    state.store_accum_buffer(
        0,
        compute_accum_matmul(
            state.load_mreg(params.ms),
            state.load_weight_slot(0, params.ws),
            state.load_accum_buffer(0),
            scale_raw=state.read_ereg(0),
            config=state.config,
        ),
    )


@instruction(
    mnemonic="vmatmul.acc.mxu1",
    params_type=MXUMatmulAccType,
    latency=MATMUL_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmatmul_acc_mxu1(state: ArchState, params: MXUMatmulAccType) -> None:
    state.instruction_extra_cycles = (
        state.config.matmul_latency_cycles - MATMUL_LATENCY_CYCLES
    )
    state.store_accum_buffer(
        1,
        compute_accum_matmul(
            state.load_mreg(params.ms),
            state.load_weight_slot(1, params.ws),
            state.load_accum_buffer(1),
            scale_raw=state.read_ereg(0),
            config=state.config,
        ),
    )


@instruction(
    mnemonic="vadd.bf16",
    params_type=VPUBinaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vadd_bf16(state: ArchState, params: VPUBinaryType) -> None:
    _vpu_binary_result(state, params, lambda lhs, rhs: compute_bf16_vadd(lhs, rhs, config=state.config))


@instruction(
    mnemonic="vredsum.bf16",
    params_type=VPUUnaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vredsum_bf16(state: ArchState, params: VPUUnaryType) -> None:
    _vpu_unary_result(state, params, lambda src: compute_bf16_vredsum(src, config=state.config))


@instruction(
    mnemonic="vmul.bf16",
    params_type=VPUBinaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmul_bf16(state: ArchState, params: VPUBinaryType) -> None:
    _vpu_binary_result(state, params, lambda lhs, rhs: compute_bf16_vmul(lhs, rhs, config=state.config))


@instruction(
    mnemonic="vsub.bf16",
    params_type=VPUBinaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vsub_bf16(state: ArchState, params: VPUBinaryType) -> None:
    _vpu_binary_result(state, params, lambda lhs, rhs: compute_bf16_vsub(lhs, rhs, config=state.config))


@instruction(
    mnemonic="vmax.bf16",
    params_type=VPUBinaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmax_bf16(state: ArchState, params: VPUBinaryType) -> None:
    _vpu_binary_result(state, params, lambda lhs, rhs: compute_bf16_vmax(lhs, rhs, config=state.config))


@instruction(
    mnemonic="vmin.bf16",
    params_type=VPUBinaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmin_bf16(state: ArchState, params: VPUBinaryType) -> None:
    _vpu_binary_result(state, params, lambda lhs, rhs: compute_bf16_vmin(lhs, rhs, config=state.config))


@instruction(
    mnemonic="vrelu",
    params_type=VPUUnaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vrelu(state: ArchState, params: VPUUnaryType) -> None:
    _vpu_unary_result(state, params, lambda src: compute_bf16_vrelu(src, config=state.config))


@instruction(
    mnemonic="vmov",
    params_type=VPUUnaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmov(state: ArchState, params: VPUUnaryType) -> None:
    _vpu_unary_result(state, params, lambda src: compute_bf16_vmov(src, config=state.config))


@instruction(
    mnemonic="vexp",
    params_type=VPUUnaryType,
    latency=VPU_NON_PIPELINEABLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vexp(state: ArchState, params: VPUUnaryType) -> None:
    _vpu_unary_result(
        state,
        params,
        lambda src: compute_bf16_vexp(src, config=state.config),
        non_pipelineable=True,
    )


@instruction(
    mnemonic="vrecip.bf16",
    params_type=VPUUnaryType,
    latency=VPU_NON_PIPELINEABLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vrecip_bf16(state: ArchState, params: VPUUnaryType) -> None:
    _vpu_unary_result(
        state,
        params,
        lambda src: compute_bf16_vrecip(src, config=state.config),
        non_pipelineable=True,
    )


@instruction(
    mnemonic="vli.all",
    params_type=VectorImmType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vli_all(state: ArchState, params: VectorImmType) -> None:
    _apply_vpu_simple_latency(state)
    state.store_mreg(params.md, compute_vector_immediate_fill(params.imm, mode="all", config=state.config))


@instruction(
    mnemonic="vli.row",
    params_type=VectorImmType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vli_row(state: ArchState, params: VectorImmType) -> None:
    _apply_vpu_simple_latency(state)
    state.store_mreg(params.md, compute_vector_immediate_fill(params.imm, mode="row", config=state.config))


@instruction(
    mnemonic="vli.col",
    params_type=VectorImmType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vli_col(state: ArchState, params: VectorImmType) -> None:
    _apply_vpu_simple_latency(state)
    state.store_mreg(params.md, compute_vector_immediate_fill(params.imm, mode="col", config=state.config))


@instruction(
    mnemonic="vli.one",
    params_type=VectorImmType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vli_one(state: ArchState, params: VectorImmType) -> None:
    _apply_vpu_simple_latency(state)
    state.store_mreg(params.md, compute_vector_immediate_fill(params.imm, mode="one", config=state.config))


@instruction(
    mnemonic="vtrpose.xlu",
    params_type=XLUUnaryType,
    latency=XLU_TRANSPOSE_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vtrpose_xlu(state: ArchState, params: XLUUnaryType) -> None:
    src = state.load_mreg(params.ms)
    _apply_xlu_transpose_latency(state)
    state.store_mreg(params.md, compute_bf16_transpose(src, config=state.config))


@instruction(
    mnemonic="vreduce.max.xlu",
    params_type=XLUUnaryType,
    latency=XLU_TRANSPOSE_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vreduce_max_xlu(state: ArchState, params: XLUUnaryType) -> None:
    src = state.load_mreg(params.ms)
    _apply_xlu_transpose_latency(state)
    state.store_mreg(params.md, compute_bf16_row_reduce_max(src, config=state.config))


@instruction(
    mnemonic="vreduce.sum.xlu",
    params_type=XLUUnaryType,
    latency=XLU_TRANSPOSE_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vreduce_sum_xlu(state: ArchState, params: XLUUnaryType) -> None:
    src = state.load_mreg(params.ms)
    _apply_xlu_transpose_latency(state)
    state.store_mreg(params.md, compute_bf16_row_reduce_sum(src, config=state.config))


# DMA channel 0
@instruction(mnemonic="dma.load.ch0", params_type=DMAType, latency=1)
def dma_load_ch0(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_load_operands(state, params)
    state.issue_dma_load(0, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch0", params_type=DMAType, latency=1)
def dma_store_ch0(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_store_operands(state, params)
    state.issue_dma_store(0, dram_address, vmem_address, size)


@instruction(mnemonic="dma.config.ch0", params_type=DMAControlType, latency=1)
def dma_config_ch0(state: ArchState, params: DMAControlType) -> None:
    state.write_dma_base(state.read_xreg(params.rs1))


@instruction(mnemonic="dma.wait.ch0", params_type=EmptyType, latency=1)
def dma_wait_ch0(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(0)


# DMA channel 1
@instruction(mnemonic="dma.load.ch1", params_type=DMAType, latency=1)
def dma_load_ch1(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_load_operands(state, params)
    state.issue_dma_load(1, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch1", params_type=DMAType, latency=1)
def dma_store_ch1(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_store_operands(state, params)
    state.issue_dma_store(1, dram_address, vmem_address, size)


@instruction(mnemonic="dma.config.ch1", params_type=DMAControlType, latency=1)
def dma_config_ch1(state: ArchState, params: DMAControlType) -> None:
    state.write_dma_base(state.read_xreg(params.rs1))


@instruction(mnemonic="dma.wait.ch1", params_type=EmptyType, latency=1)
def dma_wait_ch1(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(1)


# DMA channel 2
@instruction(mnemonic="dma.load.ch2", params_type=DMAType, latency=1)
def dma_load_ch2(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_load_operands(state, params)
    state.issue_dma_load(2, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch2", params_type=DMAType, latency=1)
def dma_store_ch2(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_store_operands(state, params)
    state.issue_dma_store(2, dram_address, vmem_address, size)


@instruction(mnemonic="dma.config.ch2", params_type=DMAControlType, latency=1)
def dma_config_ch2(state: ArchState, params: DMAControlType) -> None:
    state.write_dma_base(state.read_xreg(params.rs1))


@instruction(mnemonic="dma.wait.ch2", params_type=EmptyType, latency=1)
def dma_wait_ch2(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(2)


# DMA channel 3
@instruction(mnemonic="dma.load.ch3", params_type=DMAType, latency=1)
def dma_load_ch3(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_load_operands(state, params)
    state.issue_dma_load(3, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch3", params_type=DMAType, latency=1)
def dma_store_ch3(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_store_operands(state, params)
    state.issue_dma_store(3, dram_address, vmem_address, size)


@instruction(mnemonic="dma.config.ch3", params_type=DMAControlType, latency=1)
def dma_config_ch3(state: ArchState, params: DMAControlType) -> None:
    state.write_dma_base(state.read_xreg(params.rs1))


@instruction(mnemonic="dma.wait.ch3", params_type=EmptyType, latency=1)
def dma_wait_ch3(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(3)


# DMA channel 4
@instruction(mnemonic="dma.load.ch4", params_type=DMAType, latency=1)
def dma_load_ch4(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_load_operands(state, params)
    state.issue_dma_load(4, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch4", params_type=DMAType, latency=1)
def dma_store_ch4(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_store_operands(state, params)
    state.issue_dma_store(4, dram_address, vmem_address, size)


@instruction(mnemonic="dma.config.ch4", params_type=DMAControlType, latency=1)
def dma_config_ch4(state: ArchState, params: DMAControlType) -> None:
    state.write_dma_base(state.read_xreg(params.rs1))


@instruction(mnemonic="dma.wait.ch4", params_type=EmptyType, latency=1)
def dma_wait_ch4(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(4)


# DMA channel 5
@instruction(mnemonic="dma.load.ch5", params_type=DMAType, latency=1)
def dma_load_ch5(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_load_operands(state, params)
    state.issue_dma_load(5, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch5", params_type=DMAType, latency=1)
def dma_store_ch5(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_store_operands(state, params)
    state.issue_dma_store(5, dram_address, vmem_address, size)


@instruction(mnemonic="dma.config.ch5", params_type=DMAControlType, latency=1)
def dma_config_ch5(state: ArchState, params: DMAControlType) -> None:
    state.write_dma_base(state.read_xreg(params.rs1))


@instruction(mnemonic="dma.wait.ch5", params_type=EmptyType, latency=1)
def dma_wait_ch5(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(5)


# DMA channel 6
@instruction(mnemonic="dma.load.ch6", params_type=DMAType, latency=1)
def dma_load_ch6(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_load_operands(state, params)
    state.issue_dma_load(6, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch6", params_type=DMAType, latency=1)
def dma_store_ch6(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_store_operands(state, params)
    state.issue_dma_store(6, dram_address, vmem_address, size)


@instruction(mnemonic="dma.config.ch6", params_type=DMAControlType, latency=1)
def dma_config_ch6(state: ArchState, params: DMAControlType) -> None:
    state.write_dma_base(state.read_xreg(params.rs1))


@instruction(mnemonic="dma.wait.ch6", params_type=EmptyType, latency=1)
def dma_wait_ch6(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(6)


# DMA channel 7
@instruction(mnemonic="dma.load.ch7", params_type=DMAType, latency=1)
def dma_load_ch7(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_load_operands(state, params)
    state.issue_dma_load(7, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch7", params_type=DMAType, latency=1)
def dma_store_ch7(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_store_operands(state, params)
    state.issue_dma_store(7, dram_address, vmem_address, size)


@instruction(mnemonic="dma.config.ch7", params_type=DMAControlType, latency=1)
def dma_config_ch7(state: ArchState, params: DMAControlType) -> None:
    state.write_dma_base(state.read_xreg(params.rs1))


@instruction(mnemonic="dma.wait.ch7", params_type=EmptyType, latency=1)
def dma_wait_ch7(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(7)


def _register_instruction_alias(alias: str, target: str) -> None:
    spec = TENSOR_INSTRUCTION_SPECS.get(target)
    if spec is not None:
        TENSOR_INSTRUCTION_SPECS[alias] = spec
    if target in ALL_INSTRUCTION_SPECS:
        ALL_INSTRUCTION_SPECS[alias] = ALL_INSTRUCTION_SPECS[target]


for alias, target in {
    "vadd": "vadd.bf16",
    "vsub": "vsub.bf16",
    "vmul": "vmul.bf16",
    "vmax": "vmax.bf16",
    "vmin": "vmin.bf16",
    "vrecip": "vrecip.bf16",
    "transpose.xlu": "vtrpose.xlu",
    "reduce.max.xlu": "vreduce.max.xlu",
    "reduce.sum.xlu": "vreduce.sum.xlu",
    "vmatpush.mxu0": "vmatpush.weight.mxu0",
    "vmatpush.mxu1": "vmatpush.weight.mxu1",
    "vmatpush.fp8.weight.mxu0": "vmatpush.weight.mxu0",
    "vmatpush.fp8.weight.mxu1": "vmatpush.weight.mxu1",
    "vmatpush.acc.bf16.mxu0": "vmatpush.bf16.acc.mxu0",
    "vmatpush.acc.bf16.mxu1": "vmatpush.bf16.acc.mxu1",
}.items():
    _register_instruction_alias(alias, target)


__all__ = [
    "DMA_CHANNEL_COUNT",
    "SCALAR_LOAD_MNEMONICS",
    "SCALAR_STORE_MNEMONICS",
    "add",
    "addi",
    "and_",
    "andi",
    "auipc",
    "beq",
    "bge",
    "bgeu",
    "blt",
    "bltu",
    "bne",
    "ebreak",
    "ecall",
    "fence",
    "jal",
    "jalr",
    "lb",
    "lbu",
    "lh",
    "lhu",
    "lui",
    "lw",
    "or_",
    "ori",
    "seld",
    "seli",
    "dma_config_ch0",
    "dma_config_ch1",
    "dma_config_ch2",
    "dma_config_ch3",
    "dma_config_ch4",
    "dma_config_ch5",
    "dma_config_ch6",
    "dma_config_ch7",
    "vadd_bf16",
    "vexp",
    "vli_all",
    "vli_col",
    "vli_one",
    "vli_row",
    "vmatmul_acc_mxu0",
    "vmatmul_acc_mxu1",
    "vmatmul_mxu0",
    "vmatmul_mxu1",
    "vmatpop_bf16_acc_mxu0",
    "vmatpop_bf16_acc_mxu1",
    "vmatpop_fp8_acc_mxu0",
    "vmatpop_fp8_acc_mxu1",
    "vmatpush_acc_bf16_mxu0",
    "vmatpush_acc_bf16_mxu1",
    "vmatpush_weight_mxu0",
    "vmatpush_weight_mxu1",
    "vmax_bf16",
    "vmin_bf16",
    "vmov",
    "vmul_bf16",
    "vrecip_bf16",
    "vreduce_max_xlu",
    "vreduce_sum_xlu",
    "vredsum_bf16",
    "vrelu",
    "vsub_bf16",
    "vtrpose_xlu",
    "sb",
    "sh",
    "sll",
    "slli",
    "slt",
    "slti",
    "sltiu",
    "sltu",
    "sra",
    "srai",
    "srl",
    "srli",
    "sub",
    "sw",
    "xor",
    "xori",
    "vload",
    "vstore",
]

sadd = add
saddi = addi
sand = and_
sandi = andi
sauipc = auipc
sbeq = beq
sbge = bge
sbgeu = bgeu
sblt = blt
sbltu = bltu
sbne = bne
sebreak = ebreak
secall = ecall
sfence = fence
sjal = jal
sjalr = jalr
slb = lb
slbu = lbu
slh = lh
slhu = lhu
slui = lui
slw = lw
sor = or_
sori = ori
ssb = sb
ssh = sh
ssll = sll
sslli = slli
sslt = slt
sslti = slti
ssltiu = sltiu
ssltu = sltu
ssra = sra
ssrai = srai
ssrl = srl
ssrli = srli
ssub = sub
ssw = sw

vadd = vadd_bf16
vsub = vsub_bf16
vmul = vmul_bf16
vmax = vmax_bf16
vmin = vmin_bf16
vrecip = vrecip_bf16
vredsum = vredsum_bf16
transpose_xlu = vtrpose_xlu
reduce_max_xlu = vreduce_max_xlu
reduce_sum_xlu = vreduce_sum_xlu
vmatpush_mxu0 = vmatpush_weight_mxu0
vmatpush_mxu1 = vmatpush_weight_mxu1
vmatpush_bf16_acc_mxu0 = vmatpush_acc_bf16_mxu0
vmatpush_bf16_acc_mxu1 = vmatpush_acc_bf16_mxu1
sxor = xor
sxori = xori
