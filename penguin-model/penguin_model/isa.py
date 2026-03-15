"""Instruction semantics for the Penguin scalar integer subset."""

from __future__ import annotations

from collections.abc import Callable

from .arch_state import ArchState, StopReason
from .instructions import (
    BType,
    DMAType,
    EmptyType,
    IType,
    JType,
    RType,
    SType,
    UType,
    instruction,
)

MASK32 = 0xFFFF_FFFF
DMA_CHANNEL_COUNT = 8


def _u32(value: int) -> int:
    return value & MASK32


def _s32(value: int) -> int:
    value &= MASK32
    return value if value < 0x8000_0000 else value - 0x1_0000_0000


def _imm_shift(params: IType) -> int:
    return params.imm & 0x1F


def _branch_if(
    state: ArchState, params: BType, predicate: Callable[[int, int], bool]
) -> None:
    lhs = state.read_xreg(params.rs1)
    rhs = state.read_xreg(params.rs2)
    if predicate(lhs, rhs):
        state.set_next_pc(state.pc + params.imm)


def _dma_operands(state: ArchState, params: DMAType) -> tuple[int, int, int]:
    return (
        _u32(state.read_xreg(params.dram_rs)),
        _u32(state.read_xreg(params.vmem_rs)),
        _u32(state.read_xreg(params.size_rs)),
    )


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


@instruction(mnemonic="sld", params_type=IType, latency=1)
def sld(state: ArchState, params: IType) -> None:
    address = _u32(state.read_xreg(params.rs1) + params.imm)
    value = state.load_vmem_u32(address)
    if value is not None:
        state.write_xreg(params.rd, value)


@instruction(mnemonic="sst", params_type=SType, latency=1)
def sst(state: ArchState, params: SType) -> None:
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


def _register_dma_instructions() -> None:
    for channel in range(DMA_CHANNEL_COUNT):
        load_mnemonic = f"dma.load.ch{channel}"
        store_mnemonic = f"dma.store.ch{channel}"
        wait_mnemonic = f"dma.wait.ch{channel}"

        @instruction(mnemonic=load_mnemonic, params_type=DMAType, latency=1)
        def dma_load(state: ArchState, params: DMAType, *, _channel: int = channel) -> None:
            dram_address, vmem_address, size = _dma_operands(state, params)
            state.issue_dma_load(_channel, dram_address, vmem_address, size)

        @instruction(mnemonic=store_mnemonic, params_type=DMAType, latency=1)
        def dma_store(
            state: ArchState, params: DMAType, *, _channel: int = channel
        ) -> None:
            dram_address, vmem_address, size = _dma_operands(state, params)
            state.issue_dma_store(_channel, dram_address, vmem_address, size)

        @instruction(mnemonic=wait_mnemonic, params_type=EmptyType, latency=1)
        def dma_wait(
            state: ArchState, params: EmptyType, *, _channel: int = channel
        ) -> None:
            del params
            state.wait_dma_channel(_channel)

        globals()[f"dma_load_ch{channel}"] = dma_load
        globals()[f"dma_store_ch{channel}"] = dma_store
        globals()[f"dma_wait_ch{channel}"] = dma_wait


_register_dma_instructions()


__all__ = [
    "DMA_CHANNEL_COUNT",
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
    "sfence",
    "sjal",
    "sjalr",
    "sld",
    "slui",
    "sor",
    "sori",
    "ssll",
    "sslli",
    "sslt",
    "ssltu",
    "ssra",
    "ssrai",
    "ssrl",
    "ssrli",
    "sst",
    "ssub",
    "sslti",
    "ssltiu",
    "sxor",
    "sxori",
]
