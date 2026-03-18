"""Instruction semantics for the Penguin scalar integer subset."""

from __future__ import annotations

from collections.abc import Callable

from .arch_state import ArchState, StopReason
from .instructions import (
    TENSOR_INSTRUCTION_SPECS,
    BType,
    DMAType,
    EmptyType,
    IType,
    JType,
    ScaleImmType,
    ScaleMemType,
    MXUMatmulAccType,
    MXUMatmulType,
    RType,
    SType,
    TensorMemType,
    UType,
    XLUTransposeType,
    VPUBinaryType,
    VPUUnaryType,
    WeightMemType,
    instruction,
)
from .memory import DMA_CHANNEL_COUNT
from .tensor import (
    MATMUL_LATENCY_CYCLES,
    MXU_PUSH_LATENCY_CYCLES,
    VLOAD_LATENCY_CYCLES,
    VPU_NON_PIPELINEABLE_OP_LATENCY_CYCLES,
    VPU_SIMPLE_OP_LATENCY_CYCLES,
    VSTORE_LATENCY_CYCLES,
    XLU_TRANSPOSE_LATENCY_CYCLES,
    compute_bf16_transpose,
    compute_bf16_matmul,
    compute_bf16_row_reduce_max,
    compute_bf16_row_reduce_sum,
    compute_bf16_vadd,
    compute_bf16_vexp,
    compute_bf16_vmax,
    compute_bf16_vmin,
    compute_bf16_vmov,
    compute_bf16_vrecip,
    compute_bf16_vsub,
    compute_bf16_vmul,
    compute_bf16_vrelu,
)

MASK32 = 0xFFFF_FFFF
SCALAR_LOAD_MNEMONICS = frozenset({"slb", "slh", "slw", "slbu", "slhu"})
SCALAR_STORE_MNEMONICS = frozenset({"ssb", "ssh", "ssw"})


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


def _dma_operands(state: ArchState, params: DMAType) -> tuple[int, int, int]:
    return (
        state.extend_address(_u32(state.read_xreg(params.dram_rs))),
        state.extend_address(_u32(state.read_xreg(params.vmem_rs))),
        _u32(state.read_xreg(params.size_rs)),
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


@instruction(mnemonic="slui", params_type=UType, latency=1)
def slui(state: ArchState, params: UType) -> None:
    state.write_xreg(params.rd, params.imm << 12)


@instruction(mnemonic="sauipc", params_type=UType, latency=1)
def sauipc(state: ArchState, params: UType) -> None:
    state.write_xreg(params.rd, state.pc + (params.imm << 12))


@instruction(mnemonic="sjal", params_type=JType, latency=1)
def sjal(state: ArchState, params: JType) -> None:
    state.write_xreg(params.rd, state.pc + 4)
    state.set_next_pc(state.pc + params.imm)


@instruction(mnemonic="sjalr", params_type=IType, latency=1)
def sjalr(state: ArchState, params: IType) -> None:
    target = (state.read_xreg(params.rs1) + params.imm) & ~1
    state.write_xreg(params.rd, state.pc + 4)
    state.set_next_pc(target)


@instruction(mnemonic="sbeq", params_type=BType, latency=1)
def sbeq(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: lhs == rhs)


@instruction(mnemonic="sbne", params_type=BType, latency=1)
def sbne(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: lhs != rhs)


@instruction(mnemonic="sblt", params_type=BType, latency=1)
def sblt(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: _s32(lhs) < _s32(rhs))


@instruction(mnemonic="sbge", params_type=BType, latency=1)
def sbge(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: _s32(lhs) >= _s32(rhs))


@instruction(mnemonic="sbltu", params_type=BType, latency=1)
def sbltu(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: _u32(lhs) < _u32(rhs))


@instruction(mnemonic="sbgeu", params_type=BType, latency=1)
def sbgeu(state: ArchState, params: BType) -> None:
    _branch_if(state, params, lambda lhs, rhs: _u32(lhs) >= _u32(rhs))


@instruction(mnemonic="slb", params_type=IType, latency=1)
def slb(state: ArchState, params: IType) -> None:
    address = _scalar_load_address(state, params)
    state.write_xreg(params.rd, _sign_extend(state.load_vmem_u8(address), 8))


@instruction(mnemonic="slh", params_type=IType, latency=1)
def slh(state: ArchState, params: IType) -> None:
    address = _scalar_load_address(state, params)
    value = state.load_vmem_u16(address)
    if value is not None:
        state.write_xreg(params.rd, _sign_extend(value, 16))


@instruction(mnemonic="slw", params_type=IType, latency=1)
def slw(state: ArchState, params: IType) -> None:
    address = _scalar_load_address(state, params)
    value = state.load_vmem_u32(address)
    if value is not None:
        state.write_xreg(params.rd, value)


@instruction(mnemonic="slbu", params_type=IType, latency=1)
def slbu(state: ArchState, params: IType) -> None:
    address = _scalar_load_address(state, params)
    state.write_xreg(params.rd, state.load_vmem_u8(address))


@instruction(mnemonic="slhu", params_type=IType, latency=1)
def slhu(state: ArchState, params: IType) -> None:
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


@instruction(mnemonic="ssb", params_type=SType, latency=1)
def ssb(state: ArchState, params: SType) -> None:
    address = _u32(state.read_xreg(params.rs1) + params.imm)
    state.store_vmem_u8(address, state.read_xreg(params.rs2))


@instruction(mnemonic="ssh", params_type=SType, latency=1)
def ssh(state: ArchState, params: SType) -> None:
    address = _u32(state.read_xreg(params.rs1) + params.imm)
    state.store_vmem_u16(address, state.read_xreg(params.rs2))


@instruction(mnemonic="ssw", params_type=SType, latency=1)
def ssw(state: ArchState, params: SType) -> None:
    address = _u32(state.read_xreg(params.rs1) + params.imm)
    state.store_vmem_u32(address, state.read_xreg(params.rs2))


@instruction(mnemonic="saddi", params_type=IType, latency=1)
def saddi(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) + params.imm)


@instruction(mnemonic="sslti", params_type=IType, latency=1)
def sslti(state: ArchState, params: IType) -> None:
    state.write_xreg(
        params.rd,
        1 if _s32(state.read_xreg(params.rs1)) < _s32(params.imm) else 0,
    )


@instruction(mnemonic="ssltiu", params_type=IType, latency=1)
def ssltiu(state: ArchState, params: IType) -> None:
    state.write_xreg(
        params.rd,
        1 if _u32(state.read_xreg(params.rs1)) < _u32(params.imm) else 0,
    )


@instruction(mnemonic="sxori", params_type=IType, latency=1)
def sxori(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) ^ params.imm)


@instruction(mnemonic="sori", params_type=IType, latency=1)
def sori(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) | params.imm)


@instruction(mnemonic="sandi", params_type=IType, latency=1)
def sandi(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) & params.imm)


@instruction(mnemonic="sslli", params_type=IType, latency=1)
def sslli(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) << _imm_shift(params))


@instruction(mnemonic="ssrli", params_type=IType, latency=1)
def ssrli(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, _u32(state.read_xreg(params.rs1)) >> _imm_shift(params))


@instruction(mnemonic="ssrai", params_type=IType, latency=1)
def ssrai(state: ArchState, params: IType) -> None:
    state.write_xreg(params.rd, _s32(state.read_xreg(params.rs1)) >> _imm_shift(params))


@instruction(mnemonic="sadd", params_type=RType, latency=1)
def sadd(state: ArchState, params: RType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) + state.read_xreg(params.rs2))


@instruction(mnemonic="ssub", params_type=RType, latency=1)
def ssub(state: ArchState, params: RType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) - state.read_xreg(params.rs2))


@instruction(mnemonic="ssll", params_type=RType, latency=1)
def ssll(state: ArchState, params: RType) -> None:
    shift = state.read_xreg(params.rs2) & 0x1F
    state.write_xreg(params.rd, state.read_xreg(params.rs1) << shift)


@instruction(mnemonic="sslt", params_type=RType, latency=1)
def sslt(state: ArchState, params: RType) -> None:
    state.write_xreg(
        params.rd,
        1 if _s32(state.read_xreg(params.rs1)) < _s32(state.read_xreg(params.rs2)) else 0,
    )


@instruction(mnemonic="ssltu", params_type=RType, latency=1)
def ssltu(state: ArchState, params: RType) -> None:
    state.write_xreg(
        params.rd,
        1 if _u32(state.read_xreg(params.rs1)) < _u32(state.read_xreg(params.rs2)) else 0,
    )


@instruction(mnemonic="sxor", params_type=RType, latency=1)
def sxor(state: ArchState, params: RType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) ^ state.read_xreg(params.rs2))


@instruction(mnemonic="ssrl", params_type=RType, latency=1)
def ssrl(state: ArchState, params: RType) -> None:
    shift = state.read_xreg(params.rs2) & 0x1F
    state.write_xreg(params.rd, _u32(state.read_xreg(params.rs1)) >> shift)


@instruction(mnemonic="ssra", params_type=RType, latency=1)
def ssra(state: ArchState, params: RType) -> None:
    shift = state.read_xreg(params.rs2) & 0x1F
    state.write_xreg(params.rd, _s32(state.read_xreg(params.rs1)) >> shift)


@instruction(mnemonic="sor", params_type=RType, latency=1)
def sor(state: ArchState, params: RType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) | state.read_xreg(params.rs2))


@instruction(mnemonic="sand", params_type=RType, latency=1)
def sand(state: ArchState, params: RType) -> None:
    state.write_xreg(params.rd, state.read_xreg(params.rs1) & state.read_xreg(params.rs2))


@instruction(mnemonic="sfence", params_type=EmptyType, latency=1)
def sfence(state: ArchState, params: EmptyType) -> None:
    del state, params


@instruction(mnemonic="secall", params_type=EmptyType, latency=1)
def secall(state: ArchState, params: EmptyType) -> None:
    del params
    state.stop(StopReason.ECALL)


@instruction(mnemonic="sebreak", params_type=EmptyType, latency=1)
def sebreak(state: ArchState, params: EmptyType) -> None:
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
    mnemonic="mxu.push.mxu0",
    params_type=WeightMemType,
    latency=MXU_PUSH_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def mxu_push_mxu0(state: ArchState, params: WeightMemType) -> None:
    address = state.resolve_indirect_address(params.rs1, params.imm)
    state.push_weight_slot(0, params.slot, address)


@instruction(
    mnemonic="mxu.push.mxu1",
    params_type=WeightMemType,
    latency=MXU_PUSH_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def mxu_push_mxu1(state: ArchState, params: WeightMemType) -> None:
    address = state.resolve_indirect_address(params.rs1, params.imm)
    state.push_weight_slot(1, params.slot, address)


def _matmul_result(
    state: ArchState,
    *,
    mxu: int,
    dest: int,
    src: int,
    slot: int,
    scale_a: int,
    scale_b: int,
    partial: int | None = None,
) -> None:
    if not state.check_mreg_pair_base(dest):
        return
    activation = state.load_mreg(src)
    weights = state.load_weight_slot(mxu, slot)
    partial_raw = None if partial is None else _load_mreg_pair(state, partial)
    if partial is not None and partial_raw is None:
        return
    state.instruction_extra_cycles = (
        state.config.matmul_latency_cycles - MATMUL_LATENCY_CYCLES
    )
    result_lo, result_hi = compute_bf16_matmul(
        activation,
        weights,
        state.read_ereg(scale_a),
        state.read_ereg(scale_b),
        partial_raw,
        config=state.config,
    )
    _store_mreg_pair(state, dest, result_lo, result_hi)


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
    mnemonic="matmul.mxu0",
    params_type=MXUMatmulType,
    latency=MATMUL_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def matmul_mxu0(state: ArchState, params: MXUMatmulType) -> None:
    _matmul_result(
        state,
        mxu=0,
        dest=params.md,
        src=params.ms,
        slot=params.ws,
        scale_a=params.ea,
        scale_b=params.eb,
    )


@instruction(
    mnemonic="matmul.mxu1",
    params_type=MXUMatmulType,
    latency=MATMUL_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def matmul_mxu1(state: ArchState, params: MXUMatmulType) -> None:
    _matmul_result(
        state,
        mxu=1,
        dest=params.md,
        src=params.ms,
        slot=params.ws,
        scale_a=params.ea,
        scale_b=params.eb,
    )


@instruction(
    mnemonic="matmul.acc.mxu0",
    params_type=MXUMatmulAccType,
    latency=MATMUL_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def matmul_acc_mxu0(state: ArchState, params: MXUMatmulAccType) -> None:
    _matmul_result(
        state,
        mxu=0,
        dest=params.md,
        src=params.ms,
        slot=params.ws,
        partial=params.mp,
        scale_a=params.ea,
        scale_b=params.eb,
    )


@instruction(
    mnemonic="matmul.acc.mxu1",
    params_type=MXUMatmulAccType,
    latency=MATMUL_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def matmul_acc_mxu1(state: ArchState, params: MXUMatmulAccType) -> None:
    _matmul_result(
        state,
        mxu=1,
        dest=params.md,
        src=params.ms,
        slot=params.ws,
        partial=params.mp,
        scale_a=params.ea,
        scale_b=params.eb,
    )


@instruction(
    mnemonic="vadd",
    params_type=VPUBinaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vadd(state: ArchState, params: VPUBinaryType) -> None:
    _vpu_binary_result(state, params, lambda lhs, rhs: compute_bf16_vadd(lhs, rhs, config=state.config))


@instruction(
    mnemonic="vmul",
    params_type=VPUBinaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmul(state: ArchState, params: VPUBinaryType) -> None:
    _vpu_binary_result(state, params, lambda lhs, rhs: compute_bf16_vmul(lhs, rhs, config=state.config))


@instruction(
    mnemonic="vsub",
    params_type=VPUBinaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vsub(state: ArchState, params: VPUBinaryType) -> None:
    _vpu_binary_result(state, params, lambda lhs, rhs: compute_bf16_vsub(lhs, rhs, config=state.config))


@instruction(
    mnemonic="vmax",
    params_type=VPUBinaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmax(state: ArchState, params: VPUBinaryType) -> None:
    _vpu_binary_result(state, params, lambda lhs, rhs: compute_bf16_vmax(lhs, rhs, config=state.config))


@instruction(
    mnemonic="vmin",
    params_type=VPUBinaryType,
    latency=VPU_SIMPLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vmin(state: ArchState, params: VPUBinaryType) -> None:
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
    mnemonic="vrecip",
    params_type=VPUUnaryType,
    latency=VPU_NON_PIPELINEABLE_OP_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def vrecip(state: ArchState, params: VPUUnaryType) -> None:
    _vpu_unary_result(
        state,
        params,
        lambda src: compute_bf16_vrecip(src, config=state.config),
        non_pipelineable=True,
    )


@instruction(
    mnemonic="transpose.xlu",
    params_type=XLUTransposeType,
    latency=XLU_TRANSPOSE_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def transpose_xlu(state: ArchState, params: XLUTransposeType) -> None:
    src = state.load_mreg(params.ms)
    _apply_xlu_transpose_latency(state)
    state.store_mreg(params.md, compute_bf16_transpose(src, config=state.config))


@instruction(
    mnemonic="reduce.max.xlu",
    params_type=XLUTransposeType,
    latency=XLU_TRANSPOSE_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def reduce_max_xlu(state: ArchState, params: XLUTransposeType) -> None:
    src = state.load_mreg(params.ms)
    _apply_xlu_transpose_latency(state)
    state.store_mreg(params.md, compute_bf16_row_reduce_max(src, config=state.config))


@instruction(
    mnemonic="reduce.sum.xlu",
    params_type=XLUTransposeType,
    latency=XLU_TRANSPOSE_LATENCY_CYCLES,
    registry=TENSOR_INSTRUCTION_SPECS,
)
def reduce_sum_xlu(state: ArchState, params: XLUTransposeType) -> None:
    src = state.load_mreg(params.ms)
    _apply_xlu_transpose_latency(state)
    state.store_mreg(params.md, compute_bf16_row_reduce_sum(src, config=state.config))


# DMA channel 0
@instruction(mnemonic="dma.load.ch0", params_type=DMAType, latency=1)
def dma_load_ch0(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_load(0, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch0", params_type=DMAType, latency=1)
def dma_store_ch0(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_store(0, dram_address, vmem_address, size)


@instruction(mnemonic="dma.wait.ch0", params_type=EmptyType, latency=1)
def dma_wait_ch0(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(0)


# DMA channel 1
@instruction(mnemonic="dma.load.ch1", params_type=DMAType, latency=1)
def dma_load_ch1(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_load(1, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch1", params_type=DMAType, latency=1)
def dma_store_ch1(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_store(1, dram_address, vmem_address, size)


@instruction(mnemonic="dma.wait.ch1", params_type=EmptyType, latency=1)
def dma_wait_ch1(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(1)


# DMA channel 2
@instruction(mnemonic="dma.load.ch2", params_type=DMAType, latency=1)
def dma_load_ch2(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_load(2, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch2", params_type=DMAType, latency=1)
def dma_store_ch2(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_store(2, dram_address, vmem_address, size)


@instruction(mnemonic="dma.wait.ch2", params_type=EmptyType, latency=1)
def dma_wait_ch2(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(2)


# DMA channel 3
@instruction(mnemonic="dma.load.ch3", params_type=DMAType, latency=1)
def dma_load_ch3(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_load(3, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch3", params_type=DMAType, latency=1)
def dma_store_ch3(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_store(3, dram_address, vmem_address, size)


@instruction(mnemonic="dma.wait.ch3", params_type=EmptyType, latency=1)
def dma_wait_ch3(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(3)


# DMA channel 4
@instruction(mnemonic="dma.load.ch4", params_type=DMAType, latency=1)
def dma_load_ch4(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_load(4, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch4", params_type=DMAType, latency=1)
def dma_store_ch4(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_store(4, dram_address, vmem_address, size)


@instruction(mnemonic="dma.wait.ch4", params_type=EmptyType, latency=1)
def dma_wait_ch4(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(4)


# DMA channel 5
@instruction(mnemonic="dma.load.ch5", params_type=DMAType, latency=1)
def dma_load_ch5(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_load(5, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch5", params_type=DMAType, latency=1)
def dma_store_ch5(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_store(5, dram_address, vmem_address, size)


@instruction(mnemonic="dma.wait.ch5", params_type=EmptyType, latency=1)
def dma_wait_ch5(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(5)


# DMA channel 6
@instruction(mnemonic="dma.load.ch6", params_type=DMAType, latency=1)
def dma_load_ch6(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_load(6, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch6", params_type=DMAType, latency=1)
def dma_store_ch6(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_store(6, dram_address, vmem_address, size)


@instruction(mnemonic="dma.wait.ch6", params_type=EmptyType, latency=1)
def dma_wait_ch6(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(6)


# DMA channel 7
@instruction(mnemonic="dma.load.ch7", params_type=DMAType, latency=1)
def dma_load_ch7(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_load(7, dram_address, vmem_address, size)


@instruction(mnemonic="dma.store.ch7", params_type=DMAType, latency=1)
def dma_store_ch7(state: ArchState, params: DMAType) -> None:
    dram_address, vmem_address, size = _dma_operands(state, params)
    state.issue_dma_store(7, dram_address, vmem_address, size)


@instruction(mnemonic="dma.wait.ch7", params_type=EmptyType, latency=1)
def dma_wait_ch7(state: ArchState, params: EmptyType) -> None:
    del params
    state.wait_dma_channel(7)


__all__ = [
    "DMA_CHANNEL_COUNT",
    "SCALAR_LOAD_MNEMONICS",
    "SCALAR_STORE_MNEMONICS",
    "slb",
    "slbu",
    "slh",
    "slhu",
    "slw",
    "sadd",
    "saddi",
    "sand",
    "sandi",
    "sauipc",
    "sbeq",
    "sbge",
    "sbgeu",
    "sblt",
    "sbltu",
    "sbne",
    "sebreak",
    "secall",
    "seld",
    "seli",
    "sfence",
    "sjal",
    "sjalr",
    "slui",
    "matmul_acc_mxu0",
    "matmul_acc_mxu1",
    "matmul_mxu0",
    "matmul_mxu1",
    "mxu_push_mxu0",
    "mxu_push_mxu1",
    "ssb",
    "sor",
    "sori",
    "ssh",
    "ssll",
    "sslli",
    "sslt",
    "ssltu",
    "ssra",
    "ssrai",
    "ssrl",
    "ssrli",
    "ssub",
    "sslti",
    "ssltiu",
    "ssw",
    "sxor",
    "sxori",
    "vload",
    "vstore",
]
