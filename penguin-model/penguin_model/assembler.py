"""Text assembly parser for Penguin scalar programs."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re

from . import isa as _isa  # noqa: F401
from .instructions import (
    ALL_INSTRUCTION_SPECS,
    BType,
    DMAControlType,
    DMAType,
    DelayType,
    EmptyType,
    INSTRUCTION_SPECS,
    IType,
    Instruction,
    JType,
    MXUAccumulatorType,
    MXUMatmulAccType,
    MXUMatmulType,
    RType,
    ScaleImmType,
    ScaleMemType,
    SType,
    TensorMemType,
    UType,
    VectorImmType,
    WeightTensorType,
    XLUUnaryType,
    VPUBinaryType,
    VPUUnaryType,
    WeightMemType,
)
from .memory import DRAM_BASE, IMEM_BASE, VMEM_BASE

_REGISTER_RE = re.compile(r"x(?P<index>[0-9]|[1-2][0-9]|3[0-1])$")
_EREGISTER_RE = re.compile(r"e(?P<index>[0-9]|[1-2][0-9]|3[0-1])$")
_MREGISTER_RE = re.compile(r"m(?P<index>[0-9]|[1-5][0-9]|6[0-3])$")
_WREGISTER_RE = re.compile(r"w(?P<index>[0-1])$")
_LABEL_RE = re.compile(r"(?P<label>[A-Za-z_][A-Za-z0-9_]*)\s*:")
_MEMORY_OPERAND_RE = re.compile(r"(?P<imm>.+)\((?P<rs1>x[0-9]+)\)$")
_MNEMONIC_ALIASES: Mapping[str, str] = {
    "slui": "lui",
    "sauipc": "auipc",
    "sjal": "jal",
    "sjalr": "jalr",
    "sbeq": "beq",
    "sbne": "bne",
    "sblt": "blt",
    "sbge": "bge",
    "sbltu": "bltu",
    "sbgeu": "bgeu",
    "slb": "lb",
    "slbu": "lbu",
    "slh": "lh",
    "slhu": "lhu",
    "slw": "lw",
    "ssb": "sb",
    "ssh": "sh",
    "ssw": "sw",
    "saddi": "addi",
    "sslti": "slti",
    "ssltiu": "sltiu",
    "sxori": "xori",
    "sori": "ori",
    "sandi": "andi",
    "sslli": "slli",
    "ssrli": "srli",
    "ssrai": "srai",
    "sadd": "add",
    "ssub": "sub",
    "ssll": "sll",
    "sslt": "slt",
    "ssltu": "sltu",
    "sxor": "xor",
    "ssrl": "srl",
    "ssra": "sra",
    "sor": "or",
    "sand": "and",
    "sfence": "fence",
    "secall": "ecall",
    "sebreak": "ebreak",
    "sld": "lw",
    "sst": "sw",
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
    "vmatpush.bf16.acc.mxu0": "vmatpush.acc.bf16.mxu0",
    "vmatpush.bf16.acc.mxu1": "vmatpush.acc.bf16.mxu1",
}
_I_TYPE_MEMORY_MNEMONICS = frozenset({"lb", "lh", "lw", "lbu", "lhu", "seld"})

_DEFAULT_SYMBOLS: Mapping[str, int] = {
    "DRAM_BASE": DRAM_BASE,
    "IMEM_BASE": IMEM_BASE,
    "VMEM_BASE": VMEM_BASE,
}


class AssemblySyntaxError(ValueError):
    """Raised when an assembly source file cannot be parsed."""


@dataclass(frozen=True, slots=True)
class AssemblyProgram(Sequence[Instruction]):
    """Parsed assembly program plus its resolved label map."""

    instructions: tuple[Instruction, ...]
    labels: Mapping[str, int]
    base_address: int = 0
    source_name: str | None = None

    def __getitem__(self, index: int) -> Instruction:
        return self.instructions[index]

    def __len__(self) -> int:
        return len(self.instructions)


@dataclass(frozen=True, slots=True)
class _SourceInstruction:
    line_number: int
    mnemonic: str
    operands: tuple[str, ...]
    raw_line: str


def assemble_file(path: str | Path, *, base_address: int = 0) -> AssemblyProgram:
    source_path = Path(path)
    return assemble_text(
        source_path.read_text(),
        source_name=str(source_path),
        base_address=base_address,
    )


def assemble_text(
    source: str,
    *,
    source_name: str = "<memory>",
    base_address: int = 0,
) -> AssemblyProgram:
    labels: dict[str, int] = {}
    source_instructions: list[_SourceInstruction] = []
    pc = base_address

    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        code = raw_line.split("#", 1)[0].strip()
        if not code:
            continue

        while True:
            match = _LABEL_RE.match(code)
            if match is None:
                break
            label = match.group("label")
            if label in labels:
                raise AssemblySyntaxError(
                    f"{source_name}:{line_number}: duplicate label '{label}'"
                )
            labels[label] = pc
            code = code[match.end() :].strip()
            if not code:
                break

        if not code:
            continue

        mnemonic, operands = _split_instruction(code, source_name=source_name, line_number=line_number)
        source_instructions.append(
            _SourceInstruction(
                line_number=line_number,
                mnemonic=mnemonic,
                operands=operands,
                raw_line=raw_line.rstrip(),
            )
        )
        pc += 1

    instructions = tuple(
        _assemble_instruction(
            line,
            pc=base_address + index,
            labels=labels,
            source_name=source_name,
        )
        for index, line in enumerate(source_instructions)
    )
    return AssemblyProgram(
        instructions=instructions,
        labels=dict(labels),
        base_address=base_address,
        source_name=source_name,
    )


def _split_instruction(
    code: str, *, source_name: str, line_number: int
) -> tuple[str, tuple[str, ...]]:
    parts = code.split(None, 1)
    mnemonic = parts[0]
    if len(parts) == 1:
        return mnemonic, ()
    return mnemonic, tuple(_split_operands(parts[1], source_name=source_name, line_number=line_number))


def _split_operands(operand_text: str, *, source_name: str, line_number: int) -> list[str]:
    operands: list[str] = []
    current: list[str] = []
    depth = 0
    for char in operand_text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise AssemblySyntaxError(
                    f"{source_name}:{line_number}: unexpected ')' in operand list"
                )
        elif char == "," and depth == 0:
            operand = "".join(current).strip()
            if not operand:
                raise AssemblySyntaxError(
                    f"{source_name}:{line_number}: empty operand in '{operand_text}'"
                )
            operands.append(operand)
            current = []
            continue
        current.append(char)

    if depth != 0:
        raise AssemblySyntaxError(
            f"{source_name}:{line_number}: unmatched '(' in operand list"
        )

    operand = "".join(current).strip()
    if operand:
        operands.append(operand)
    return operands


def _assemble_instruction(
    line: _SourceInstruction,
    *,
    pc: int,
    labels: Mapping[str, int],
    source_name: str,
) -> Instruction:
    mnemonic = _MNEMONIC_ALIASES.get(line.mnemonic, line.mnemonic)
    operands = line.operands

    if mnemonic == "nop":
        _expect_operand_count(
            mnemonic,
            operands,
            expected=0,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction("addi", IType(rd=0, rs1=0, imm=0))

    if mnemonic == "li":
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            "addi",
            IType(
                rd=_parse_register(operands[0], source_name=source_name, line_number=line.line_number),
                rs1=0,
                imm=_evaluate_expression(
                    operands[1],
                    labels=labels,
                    pc=pc,
                    relative_to_pc=False,
                    source_name=source_name,
                    line_number=line.line_number,
                ),
            ),
        )

    spec = ALL_INSTRUCTION_SPECS.get(mnemonic)
    if spec is None:
        raise AssemblySyntaxError(
            f"{source_name}:{line.line_number}: unknown mnemonic '{mnemonic}'"
        )

    if spec.params_type is RType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=3,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            RType(
                rd=_parse_register(operands[0], source_name=source_name, line_number=line.line_number),
                rs1=_parse_register(operands[1], source_name=source_name, line_number=line.line_number),
                rs2=_parse_register(operands[2], source_name=source_name, line_number=line.line_number),
            ),
        )

    if spec.params_type is IType:
        if mnemonic in _I_TYPE_MEMORY_MNEMONICS:
            _expect_operand_count(
                mnemonic,
                operands,
                expected=2,
                source_name=source_name,
                line_number=line.line_number,
            )
            rs1, imm = _parse_memory_operand(
                operands[1],
                labels=labels,
                pc=pc,
                source_name=source_name,
                line_number=line.line_number,
            )
            return Instruction(
                mnemonic,
                IType(
                    rd=_parse_register(
                        operands[0], source_name=source_name, line_number=line.line_number
                    ),
                    rs1=rs1,
                    imm=imm,
                ),
            )

        _expect_operand_count(
            mnemonic,
            operands,
            expected=3,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            IType(
                rd=_parse_register(operands[0], source_name=source_name, line_number=line.line_number),
                rs1=_parse_register(operands[1], source_name=source_name, line_number=line.line_number),
                imm=_evaluate_expression(
                    operands[2],
                    labels=labels,
                    pc=pc,
                    relative_to_pc=False,
                    source_name=source_name,
                    line_number=line.line_number,
                ),
            ),
        )

    if spec.params_type is ScaleImmType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        imm = _evaluate_expression(
            operands[1],
            labels=labels,
            pc=pc,
            relative_to_pc=False,
            source_name=source_name,
            line_number=line.line_number,
        )
        if imm < 0 or imm > 0xFF:
            raise AssemblySyntaxError(
                f"{source_name}:{line.line_number}: '{mnemonic}' immediate must be in [0, 255]"
            )
        return Instruction(
            mnemonic,
            ScaleImmType(
                ed=_parse_eregister(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                imm=imm,
            ),
        )

    if spec.params_type is ScaleMemType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        rs1, imm = _parse_memory_operand(
            operands[1],
            labels=labels,
            pc=pc,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            ScaleMemType(
                ed=_parse_eregister(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                rs1=rs1,
                imm=imm,
            ),
        )

    if spec.params_type is SType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        rs1, imm = _parse_memory_operand(
            operands[1],
            labels=labels,
            pc=pc,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            SType(
                rs1=rs1,
                rs2=_parse_register(operands[0], source_name=source_name, line_number=line.line_number),
                imm=imm,
            ),
        )

    if spec.params_type is BType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=3,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            BType(
                rs1=_parse_register(operands[0], source_name=source_name, line_number=line.line_number),
                rs2=_parse_register(operands[1], source_name=source_name, line_number=line.line_number),
                imm=_evaluate_expression(
                    operands[2],
                    labels=labels,
                    pc=pc,
                    relative_to_pc=True,
                    source_name=source_name,
                    line_number=line.line_number,
                ),
            ),
        )

    if spec.params_type is UType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            UType(
                rd=_parse_register(operands[0], source_name=source_name, line_number=line.line_number),
                imm=_evaluate_expression(
                    operands[1],
                    labels=labels,
                    pc=pc,
                    relative_to_pc=False,
                    source_name=source_name,
                    line_number=line.line_number,
                ),
            ),
        )

    if spec.params_type is JType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            JType(
                rd=_parse_register(operands[0], source_name=source_name, line_number=line.line_number),
                imm=_evaluate_expression(
                    operands[1],
                    labels=labels,
                    pc=pc,
                    relative_to_pc=True,
                    source_name=source_name,
                    line_number=line.line_number,
                ),
            ),
        )

    if spec.params_type is EmptyType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=0,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(mnemonic, EmptyType())

    if spec.params_type is DelayType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=1,
            source_name=source_name,
            line_number=line.line_number,
        )
        cycles = _evaluate_expression(
            operands[0],
            labels=labels,
            pc=pc,
            relative_to_pc=False,
            source_name=source_name,
            line_number=line.line_number,
        )
        if cycles < 0 or cycles > 0xFFF:
            raise AssemblySyntaxError(
                f"{source_name}:{line.line_number}: 'delay' immediate must be in [0, 4095]"
            )
        return Instruction(mnemonic, DelayType(cycles=cycles))

    if spec.params_type is DMAType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=3,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            DMAType(
                rd=_parse_register(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                rs1=_parse_register(
                    operands[1], source_name=source_name, line_number=line.line_number
                ),
                rs2=_parse_register(
                    operands[2], source_name=source_name, line_number=line.line_number
                ),
            ),
        )

    if spec.params_type is DMAControlType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=1,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            DMAControlType(
                rs1=_parse_register(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
            ),
        )

    if spec.params_type is TensorMemType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        rs1, imm = _parse_memory_operand(
            operands[1],
            labels=labels,
            pc=pc,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            TensorMemType(
                mreg=_parse_mregister(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                rs1=rs1,
                imm=imm,
            ),
        )

    if spec.params_type is WeightMemType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        try:
            rs1, imm = _parse_memory_operand(
                operands[1],
                labels=labels,
                pc=pc,
                source_name=source_name,
                line_number=line.line_number,
            )
        except AssemblySyntaxError:
            rs1 = _parse_register(
                operands[1], source_name=source_name, line_number=line.line_number
            )
            imm = 0
        return Instruction(
            mnemonic,
            WeightMemType(
                slot=_parse_weight_selector(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                rs1=rs1,
                imm=imm,
            ),
        )

    if spec.params_type is WeightTensorType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            WeightTensorType(
                slot=_parse_weight_selector(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                ms=_parse_mregister(
                    operands[1], source_name=source_name, line_number=line.line_number
                ),
            ),
        )

    if spec.params_type is MXUAccumulatorType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=1,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            MXUAccumulatorType(
                mreg=_parse_mregister(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
            ),
        )

    if spec.params_type is MXUMatmulType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            MXUMatmulType(
                ms=_parse_mregister(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                ws=_parse_weight_selector(
                    operands[1], source_name=source_name, line_number=line.line_number
                ),
            ),
        )

    if spec.params_type is MXUMatmulAccType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            MXUMatmulAccType(
                ms=_parse_mregister(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                ws=_parse_weight_selector(
                    operands[1], source_name=source_name, line_number=line.line_number
                ),
            ),
        )

    if spec.params_type is VPUBinaryType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=3,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            VPUBinaryType(
                md=_parse_mregister(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                ms1=_parse_mregister(
                    operands[1], source_name=source_name, line_number=line.line_number
                ),
                ms2=_parse_mregister(
                    operands[2], source_name=source_name, line_number=line.line_number
                ),
            ),
        )

    if spec.params_type is VPUUnaryType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            VPUUnaryType(
                md=_parse_mregister(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                ms=_parse_mregister(
                    operands[1], source_name=source_name, line_number=line.line_number
                ),
            ),
        )

    if spec.params_type is VectorImmType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            VectorImmType(
                md=_parse_mregister(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                imm=_evaluate_expression(
                    operands[1],
                    labels=labels,
                    pc=pc,
                    relative_to_pc=False,
                    source_name=source_name,
                    line_number=line.line_number,
                ),
            ),
        )

    if spec.params_type is XLUUnaryType:
        _expect_operand_count(
            mnemonic,
            operands,
            expected=2,
            source_name=source_name,
            line_number=line.line_number,
        )
        return Instruction(
            mnemonic,
            XLUUnaryType(
                md=_parse_mregister(
                    operands[0], source_name=source_name, line_number=line.line_number
                ),
                ms=_parse_mregister(
                    operands[1], source_name=source_name, line_number=line.line_number
                ),
            ),
        )

    raise AssemblySyntaxError(
        f"{source_name}:{line.line_number}: unsupported operand type for '{mnemonic}'"
    )


def _expect_operand_count(
    mnemonic: str,
    operands: Sequence[str],
    *,
    expected: int,
    source_name: str,
    line_number: int,
) -> None:
    if len(operands) != expected:
        raise AssemblySyntaxError(
            f"{source_name}:{line_number}: '{mnemonic}' expects {expected} operands, "
            f"got {len(operands)}"
        )


def _parse_register(token: str, *, source_name: str, line_number: int) -> int:
    match = _REGISTER_RE.fullmatch(token.strip())
    if match is None:
        raise AssemblySyntaxError(
            f"{source_name}:{line_number}: invalid register '{token}'"
        )
    return int(match.group("index"))


def _parse_eregister(token: str, *, source_name: str, line_number: int) -> int:
    match = _EREGISTER_RE.fullmatch(token.strip())
    if match is None:
        raise AssemblySyntaxError(
            f"{source_name}:{line_number}: invalid scale register '{token}'"
        )
    return int(match.group("index"))


def _parse_mregister(token: str, *, source_name: str, line_number: int) -> int:
    match = _MREGISTER_RE.fullmatch(token.strip())
    if match is None:
        raise AssemblySyntaxError(
            f"{source_name}:{line_number}: invalid tensor register '{token}'"
        )
    return int(match.group("index"))


def _parse_weight_selector(token: str, *, source_name: str, line_number: int) -> int:
    match = _WREGISTER_RE.fullmatch(token.strip())
    if match is None:
        raise AssemblySyntaxError(
            f"{source_name}:{line_number}: invalid weight selector '{token}'"
        )
    return int(match.group("index"))


def _parse_memory_operand(
    operand: str,
    *,
    labels: Mapping[str, int],
    pc: int,
    source_name: str,
    line_number: int,
) -> tuple[int, int]:
    match = _MEMORY_OPERAND_RE.fullmatch(operand.replace(" ", ""))
    if match is None:
        raise AssemblySyntaxError(
            f"{source_name}:{line_number}: expected memory operand imm(xN), got '{operand}'"
        )
    return (
        _parse_register(match.group("rs1"), source_name=source_name, line_number=line_number),
        _evaluate_expression(
            match.group("imm"),
            labels=labels,
            pc=pc,
            relative_to_pc=False,
            source_name=source_name,
            line_number=line_number,
        ),
    )


def _evaluate_expression(
    expression: str,
    *,
    labels: Mapping[str, int],
    pc: int,
    relative_to_pc: bool,
    source_name: str,
    line_number: int,
) -> int:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise AssemblySyntaxError(
            f"{source_name}:{line_number}: invalid expression '{expression}'"
        ) from exc

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    symbols = dict(_DEFAULT_SYMBOLS)
    symbols.update(labels)
    value = _evaluate_ast(tree.body, symbols, source_name=source_name, line_number=line_number)
    if relative_to_pc and any(name in labels for name in names):
        value -= pc
    return value


def _evaluate_ast(
    node: ast.AST,
    symbols: Mapping[str, int],
    *,
    source_name: str,
    line_number: int,
) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)

    if isinstance(node, ast.Name):
        if node.id not in symbols:
            raise AssemblySyntaxError(
                f"{source_name}:{line_number}: unknown symbol '{node.id}'"
            )
        return symbols[node.id]

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _evaluate_ast(
            node.operand,
            symbols,
            source_name=source_name,
            line_number=line_number,
        )
        return operand if isinstance(node.op, ast.UAdd) else -operand

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        lhs = _evaluate_ast(node.left, symbols, source_name=source_name, line_number=line_number)
        rhs = _evaluate_ast(node.right, symbols, source_name=source_name, line_number=line_number)
        return lhs + rhs if isinstance(node.op, ast.Add) else lhs - rhs

    raise AssemblySyntaxError(
        f"{source_name}:{line_number}: unsupported expression syntax"
    )


__all__ = [
    "AssemblyProgram",
    "AssemblySyntaxError",
    "assemble_file",
    "assemble_text",
]
