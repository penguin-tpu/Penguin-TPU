"""Tests for the Penguin text assembly parser."""

from __future__ import annotations

import pytest

from penguin_model import (
    AssemblySyntaxError,
    BType,
    EmptyType,
    IMEM_BASE,
    IType,
    Instruction,
    JType,
    MXUMatmulAccType,
    TensorMemType,
    VPUBinaryType,
    VPUUnaryType,
    VMEM_BASE,
    WeightMemType,
    assemble_text,
)


def test_assembler_resolves_labels_and_pseudo_instructions() -> None:
    program = assemble_text(
        """
start:
    li x1, VMEM_BASE + 0x20
    nop
    sbne x1, x0, start
"""
    )

    assert program.labels == {"start": 0}
    assert list(program) == [
        Instruction("saddi", IType(rd=1, rs1=0, imm=VMEM_BASE + 0x20)),
        Instruction("saddi", IType(rd=0, rs1=0, imm=0)),
        Instruction("sbne", BType(rs1=1, rs2=0, imm=-8)),
    ]


def test_assembler_uses_absolute_label_values_for_non_control_immediates() -> None:
    program = assemble_text(
        """
    li x12, target + 1
    sjal x0, target
target:
    nop
"""
    )

    assert list(program) == [
        Instruction("saddi", IType(rd=12, rs1=0, imm=9)),
        Instruction("sjal", JType(rd=0, imm=4)),
        Instruction("saddi", IType(rd=0, rs1=0, imm=0)),
    ]


def test_assembler_rejects_invalid_register_names() -> None:
    with pytest.raises(AssemblySyntaxError, match="invalid register"):
        assemble_text("saddi y1, x0, 1\n")


def test_assembler_parses_tensor_memory_and_mxu_operands() -> None:
    program = assemble_text(
        """
    vload m3, 32(x1)
    mxu.push.mxu1 w0, 64(x2)
    matmul.add.mxu1 m4, m3, w0, m5
    vstore m4, 96(x6)
"""
    )

    assert list(program) == [
        Instruction("vload", TensorMemType(mreg=3, rs1=1, imm=32)),
        Instruction("mxu.push.mxu1", WeightMemType(slot=0, rs1=2, imm=64)),
        Instruction("matmul.add.mxu1", MXUMatmulAccType(md=4, ms=3, ws=0, mp=5)),
        Instruction("vstore", TensorMemType(mreg=4, rs1=6, imm=96)),
    ]


def test_assembler_parses_vpu_operands() -> None:
    program = assemble_text(
        """
    vadd m7, m8, m9
    vrelu m10, m11
    vmov m12, m13
"""
    )

    assert list(program) == [
        Instruction("vadd", VPUBinaryType(md=7, ms1=8, ms2=9)),
        Instruction("vrelu", VPUUnaryType(md=10, ms=11)),
        Instruction("vmov", VPUUnaryType(md=12, ms=13)),
    ]


def test_assembler_parses_memory_operand_expressions_with_symbols_and_labels() -> None:
    program = assemble_text(
        """
start:
    vload m1, VMEM_BASE + 64(x2)
    vstore m1, target - start + 32(x3)
target:
    sebreak
"""
    )

    assert list(program) == [
        Instruction("vload", TensorMemType(mreg=1, rs1=2, imm=VMEM_BASE + 64)),
        Instruction("vstore", TensorMemType(mreg=1, rs1=3, imm=40)),
        Instruction("sebreak", EmptyType()),
    ]


def test_assembler_applies_nonzero_program_base_to_labels() -> None:
    program = assemble_text(
        """
start:
    li x1, target
    sjal x0, target
target:
    sebreak
""",
        base_address=IMEM_BASE,
    )

    assert program.base_address == IMEM_BASE
    assert program.labels == {
        "start": IMEM_BASE,
        "target": IMEM_BASE + 8,
    }
    assert list(program) == [
        Instruction("saddi", IType(rd=1, rs1=0, imm=IMEM_BASE + 8)),
        Instruction("sjal", JType(rd=0, imm=4)),
        Instruction("sebreak", EmptyType()),
    ]
