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


@dataclass(frozen=True, slots=True)
class ScaleImmType:
    """Scale-register immediate load form."""

    ed: int
    imm: int


@dataclass(frozen=True, slots=True)
class ScaleMemType:
    """Scale-register load from one VMEM byte."""

    ed: int
    rs1: int
    imm: int = 0


@dataclass(frozen=True, slots=True)
class TensorMemType:
    """Tensor register plus scalar-register-indirect VMEM address."""

    mreg: int
    rs1: int
    imm: int = 0


@dataclass(frozen=True, slots=True)
class WeightMemType:
    """MXU weight-slot selector plus scalar-register-indirect VMEM address."""

    slot: int
    rs1: int
    imm: int = 0


@dataclass(frozen=True, slots=True)
class MXUMatmulType:
    """Fresh matmul launch: dest tensor, activation tensor, weight selector."""

    md: int
    ms: int
    ws: int
    ea: int
    eb: int


@dataclass(frozen=True, slots=True)
class MXUMatmulAccType:
    """Accumulating matmul launch with an explicit partial-sum tensor."""

    md: int
    ms: int
    ws: int
    mp: int
    ea: int
    eb: int


@dataclass(frozen=True, slots=True)
class VPUBinaryType:
    """Whole-register binary VPU form: dest tensor, lhs tensor, rhs tensor."""

    md: int
    ms1: int
    ms2: int


@dataclass(frozen=True, slots=True)
class VPUUnaryType:
    """Whole-register unary VPU form: dest tensor, source tensor."""

    md: int
    ms: int


@dataclass(frozen=True, slots=True)
class XLUTransposeType:
    """Whole-register transpose form: dest tensor, source tensor."""

    md: int
    ms: int


InstructionParams: TypeAlias = (
    RType
    | IType
    | SType
    | BType
    | UType
    | JType
    | EmptyType
    | DMAType
    | ScaleImmType
    | ScaleMemType
    | TensorMemType
    | WeightMemType
    | MXUMatmulType
    | MXUMatmulAccType
    | VPUBinaryType
    | VPUUnaryType
    | XLUTransposeType
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
TENSOR_INSTRUCTION_SPECS: dict[str, InstructionSpec] = {}
ALL_INSTRUCTION_SPECS: dict[str, InstructionSpec] = {}


def instruction(
    *,
    mnemonic: str,
    params_type: type[InstructionParams],
    latency: int,
    registry: dict[str, InstructionSpec] | None = None,
) -> Callable[[InstructionFn], InstructionFn]:
    """Register a semantic function as the implementation of one instruction."""

    def decorate(semantics: InstructionFn) -> InstructionFn:
        target_registry = INSTRUCTION_SPECS if registry is None else registry
        setattr(semantics, "mnemonic", mnemonic)
        setattr(semantics, "params_type", params_type)
        setattr(semantics, "latency", latency)
        spec = InstructionSpec(
            mnemonic=mnemonic,
            params_type=params_type,
            semantics=semantics,
            latency=latency,
        )
        target_registry[mnemonic] = spec
        ALL_INSTRUCTION_SPECS[mnemonic] = spec
        return semantics

    return decorate


__all__ = [
    "BType",
    "DMAType",
    "EmptyType",
    "IType",
    "ALL_INSTRUCTION_SPECS",
    "INSTRUCTION_SPECS",
    "Instruction",
    "InstructionFn",
    "InstructionParams",
    "InstructionSpec",
    "JType",
    "MXUMatmulAccType",
    "MXUMatmulType",
    "RType",
    "ScaleImmType",
    "ScaleMemType",
    "SType",
    "TENSOR_INSTRUCTION_SPECS",
    "TensorMemType",
    "UType",
    "XLUTransposeType",
    "VPUBinaryType",
    "VPUUnaryType",
    "WeightMemType",
    "instruction",
]
