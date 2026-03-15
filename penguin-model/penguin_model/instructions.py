"""Decoded scalar instruction forms for the Penguin functional model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

from .arch_state import ArchState


@dataclass(frozen=True, slots=True)
class RType:
    """R-type encoding: register-register ALU operations."""

    rd: int
    rs1: int
    rs2: int


@dataclass(frozen=True, slots=True)
class IType:
    """I-type encoding: register-immediate ops, loads, and `sjalr`."""

    rd: int
    rs1: int
    imm: int = 0


@dataclass(frozen=True, slots=True)
class SType:
    """S-type encoding: stores."""

    rs1: int
    rs2: int
    imm: int = 0


@dataclass(frozen=True, slots=True)
class BType:
    """B-type encoding: conditional branches."""

    rs1: int
    rs2: int
    imm: int = 0


@dataclass(frozen=True, slots=True)
class UType:
    """U-type encoding: upper-immediate forms."""

    rd: int
    imm: int


@dataclass(frozen=True, slots=True)
class JType:
    """J-type encoding: unconditional jumps."""

    rd: int
    imm: int = 0


@dataclass(frozen=True, slots=True)
class EmptyType:
    """Zero-operand form used by fence and environment instructions."""


@dataclass(frozen=True, slots=True)
class DMAType:
    """DMA transfer form: DRAM addr reg, VMEM addr reg, size reg."""

    dram_rs: int
    vmem_rs: int
    size_rs: int


InstructionParams: TypeAlias = (
    RType | IType | SType | BType | UType | JType | EmptyType | DMAType
)

InstructionFn: TypeAlias = Callable[[ArchState, InstructionParams], None]


@dataclass(frozen=True, slots=True)
class Instruction:
    """Decoded instruction consisting of a mnemonic and typed operands."""

    mnemonic: str
    params: InstructionParams


@dataclass(frozen=True, slots=True)
class InstructionSpec:
    """Instruction metadata used by the executor."""

    mnemonic: str
    params_type: type[InstructionParams]
    semantics: InstructionFn
    latency: int


INSTRUCTION_SPECS: dict[str, InstructionSpec] = {}


def instruction(
    *, mnemonic: str, params_type: type[InstructionParams], latency: int
) -> Callable[[InstructionFn], InstructionFn]:
    """Register a semantic function as the implementation of one instruction."""

    def decorate(semantics: InstructionFn) -> InstructionFn:
        setattr(semantics, "mnemonic", mnemonic)
        setattr(semantics, "params_type", params_type)
        setattr(semantics, "latency", latency)
        INSTRUCTION_SPECS[mnemonic] = InstructionSpec(
            mnemonic=mnemonic,
            params_type=params_type,
            semantics=semantics,
            latency=latency,
        )
        return semantics

    return decorate


__all__ = [
    "BType",
    "DMAType",
    "EmptyType",
    "IType",
    "INSTRUCTION_SPECS",
    "Instruction",
    "InstructionFn",
    "InstructionParams",
    "InstructionSpec",
    "JType",
    "RType",
    "SType",
    "UType",
    "instruction",
]
