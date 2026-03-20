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
class DelayType:
    """Decode-resident frontend stall form with an unsigned cycle count."""

    cycles: int


@dataclass(frozen=True, slots=True)
class DMAType:
    """DMA transfer form using scalar `R` layout: `rd`, `rs1`, and `rs2`."""

    rd: int
    rs1: int
    rs2: int


@dataclass(frozen=True, slots=True)
class DMAControlType:
    """DMA control form using scalar `I` layout with one source register."""

    rs1: int


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
class WeightTensorType:
    """MXU weight-slot selector plus tensor-register source."""

    slot: int
    ms: int


@dataclass(frozen=True, slots=True)
class MXUAccumulatorType:
    """Tensor-register operand for MXU accumulator movement forms."""

    mreg: int


@dataclass(frozen=True, slots=True)
class MXUMatmulType:
    """MXU launch form: activation tensor and weight-slot selector."""

    ms: int
    ws: int


@dataclass(frozen=True, slots=True)
class MXUMatmulAccType:
    """Accumulating MXU launch form over the unique local accumulation buffer."""

    ms: int
    ws: int


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
class VectorImmType:
    """Vector-immediate form: dest tensor and 16-bit immediate payload."""

    md: int
    imm: int


@dataclass(frozen=True, slots=True)
class XLUUnaryType:
    """Whole-register XLU form: dest tensor, source tensor."""

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
    | DelayType
    | DMAType
    | DMAControlType
    | ScaleImmType
    | ScaleMemType
    | TensorMemType
    | WeightTensorType
    | MXUAccumulatorType
    | MXUMatmulType
    | MXUMatmulAccType
    | VPUBinaryType
    | VPUUnaryType
    | VectorImmType
    | XLUUnaryType
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
    "DMAControlType",
    "DMAType",
    "DelayType",
    "EmptyType",
    "IType",
    "ALL_INSTRUCTION_SPECS",
    "INSTRUCTION_SPECS",
    "Instruction",
    "InstructionFn",
    "InstructionParams",
    "InstructionSpec",
    "JType",
    "MXUAccumulatorType",
    "MXUMatmulAccType",
    "MXUMatmulType",
    "RType",
    "ScaleImmType",
    "ScaleMemType",
    "SType",
    "TENSOR_INSTRUCTION_SPECS",
    "TensorMemType",
    "UType",
    "VectorImmType",
    "WeightTensorType",
    "XLUTransposeType",
    "XLUUnaryType",
    "VPUBinaryType",
    "VPUUnaryType",
    "instruction",
]

XLUTransposeType = XLUUnaryType
