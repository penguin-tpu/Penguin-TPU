"""Tests for the Penguin text assembly parser."""

from __future__ import annotations

import pytest

from penguin_model import (
    AssemblySyntaxError,
    BType,
    DelayType,
    EmptyType,
    IMEM_BASE,
    IType,
    Instruction,
    JType,
    MXUAccumulatorType,
    MXUMatmulAccType,
    MXUMatmulType,
    ScaleImmType,
    ScaleMemType,
    TensorMemType,
    WeightTensorType,
    XLUTransposeType,
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
        Instruction("addi", IType(rd=1, rs1=0, imm=VMEM_BASE + 0x20)),
        Instruction("addi", IType(rd=0, rs1=0, imm=0)),
        Instruction("bne", BType(rs1=1, rs2=0, imm=-2)),
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
        Instruction("addi", IType(rd=12, rs1=0, imm=3)),
        Instruction("jal", JType(rd=0, imm=1)),
        Instruction("addi", IType(rd=0, rs1=0, imm=0)),
    ]


def test_assembler_rejects_invalid_register_names() -> None:
    with pytest.raises(AssemblySyntaxError, match="invalid register"):
        assemble_text("addi y1, x0, 1\n")


def test_assembler_parses_tensor_memory_and_mxu_operands() -> None:
    program = assemble_text(
        """
    seli e1, 0
    seld e2, 4(x7)
    vload m3, 32(x1)
    vmatpush.mxu1 w0, m8
    vload.weight.mxu0 w1, x2
    vmatpush.bf16.acc.mxu1 m4
    vmatmul.mxu0 m7, w1
    vmatmul.acc.mxu1 m3, w0
    vmatpop.bf16.acc.mxu1 m4
    vmatpop.fp8.acc.mxu0 m6
    vstore m4, 96(x6)
"""
    )

    assert list(program) == [
        Instruction("seli", ScaleImmType(ed=1, imm=0)),
        Instruction("seld", ScaleMemType(ed=2, rs1=7, imm=4)),
        Instruction("vload", TensorMemType(mreg=3, rs1=1, imm=32)),
        Instruction("vmatpush.weight.mxu1", WeightTensorType(slot=0, ms=8)),
        Instruction("vload.weight.mxu0", WeightMemType(slot=1, rs1=2, imm=0)),
        Instruction("vmatpush.acc.bf16.mxu1", MXUAccumulatorType(mreg=4)),
        Instruction("vmatmul.mxu0", MXUMatmulType(ms=7, ws=1)),
        Instruction("vmatmul.acc.mxu1", MXUMatmulAccType(ms=3, ws=0)),
        Instruction("vmatpop.bf16.acc.mxu1", MXUAccumulatorType(mreg=4)),
        Instruction("vmatpop.fp8.acc.mxu0", MXUAccumulatorType(mreg=6)),
        Instruction("vstore", TensorMemType(mreg=4, rs1=6, imm=96)),
    ]


def test_assembler_parses_vpu_operands() -> None:
    program = assemble_text(
        """
    vadd m7, m8, m9
    vsub m10, m11, m12
    vmax m13, m14, m15
    vmin m16, m17, m18
    vmul m19, m20, m21
    vrelu m22, m23
    vmov m24, m25
    vexp m26, m27
    vrecip m28, m29
"""
    )

    assert list(program) == [
        Instruction("vadd.bf16", VPUBinaryType(md=7, ms1=8, ms2=9)),
        Instruction("vsub.bf16", VPUBinaryType(md=10, ms1=11, ms2=12)),
        Instruction("vmax.bf16", VPUBinaryType(md=13, ms1=14, ms2=15)),
        Instruction("vmin.bf16", VPUBinaryType(md=16, ms1=17, ms2=18)),
        Instruction("vmul.bf16", VPUBinaryType(md=19, ms1=20, ms2=21)),
        Instruction("vrelu", VPUUnaryType(md=22, ms=23)),
        Instruction("vmov", VPUUnaryType(md=24, ms=25)),
        Instruction("vexp", VPUUnaryType(md=26, ms=27)),
        Instruction("vrecip.bf16", VPUUnaryType(md=28, ms=29)),
    ]


def test_assembler_parses_delay_instruction() -> None:
    program = assemble_text("delay 7\n")

    assert list(program) == [Instruction("delay", DelayType(cycles=7))]


def test_assembler_parses_xlu_operands() -> None:
    program = assemble_text(
        """
    transpose.xlu m14, m15
    reduce.max.xlu m16, m17
    reduce.sum.xlu m18, m19
"""
    )

    assert list(program) == [
        Instruction("vtrpose.xlu", XLUTransposeType(md=14, ms=15)),
        Instruction("vreduce.max.xlu", XLUTransposeType(md=16, ms=17)),
        Instruction("vreduce.sum.xlu", XLUTransposeType(md=18, ms=19)),
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
        Instruction("vstore", TensorMemType(mreg=1, rs1=3, imm=34)),
        Instruction("ebreak", EmptyType()),
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
        "target": IMEM_BASE + 2,
    }
    assert list(program) == [
        Instruction("addi", IType(rd=1, rs1=0, imm=IMEM_BASE + 2)),
        Instruction("jal", JType(rd=0, imm=1)),
        Instruction("ebreak", EmptyType()),
    ]
