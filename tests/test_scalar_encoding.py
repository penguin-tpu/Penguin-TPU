from __future__ import annotations

import pytest

from penguin_model import BType, EmptyType, IType, Instruction, JType, RType, SType, UType, VPUBinaryType
from penguin_model.scalar_encoding import encode_scalar_instruction


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        (Instruction("saddi", IType(rd=1, rs1=0, imm=42)), 0x02A00093),
        (Instruction("slui", UType(rd=1, imm=0x12345)), 0x123450B7),
        (Instruction("ssb", SType(rs1=1, rs2=2, imm=4)), 0x00208223),
        (Instruction("ssh", SType(rs1=1, rs2=2, imm=4)), 0x00209223),
        (Instruction("ssw", SType(rs1=1, rs2=2, imm=4)), 0x0020A223),
        (Instruction("slb", IType(rd=3, rs1=1, imm=8)), 0x00808183),
        (Instruction("slh", IType(rd=3, rs1=1, imm=8)), 0x00809183),
        (Instruction("slw", IType(rd=3, rs1=1, imm=8)), 0x0080A183),
        (Instruction("slbu", IType(rd=3, rs1=1, imm=8)), 0x0080C183),
        (Instruction("slhu", IType(rd=3, rs1=1, imm=8)), 0x0080D183),
        (Instruction("sadd", RType(rd=5, rs1=6, rs2=7)), 0x007302B3),
        (Instruction("ssub", RType(rd=5, rs1=6, rs2=7)), 0x407302B3),
        (Instruction("vadd", VPUBinaryType(md=2, ms1=0, ms2=1)), 0x0010010B),
        (Instruction("sbeq", BType(rs1=1, rs2=2, imm=16)), 0x00208863),
        (Instruction("sjal", JType(rd=1, imm=32)), 0x020000EF),
        (Instruction("sfence", EmptyType()), 0x0000000F),
        (Instruction("sebreak", EmptyType()), 0x00100073),
    ],
)
def test_encode_scalar_instruction_known_words(instruction: Instruction, expected: int) -> None:
    assert encode_scalar_instruction(instruction) == expected


def test_encode_scalar_instruction_rejects_out_of_range_immediate() -> None:
    with pytest.raises(ValueError):
        encode_scalar_instruction(Instruction("saddi", IType(rd=1, rs1=0, imm=4096)))


def test_encode_scalar_instruction_rejects_preliminary_vpu_registers_above_m31() -> None:
    with pytest.raises(ValueError):
        encode_scalar_instruction(Instruction("vadd", VPUBinaryType(md=32, ms1=0, ms2=1)))


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        (Instruction("sld", IType(rd=3, rs1=1, imm=8)), 0x0080A183),
        (Instruction("sst", SType(rs1=1, rs2=2, imm=4)), 0x0020A223),
    ],
)
def test_encode_scalar_instruction_accepts_legacy_word_load_store_aliases(
    instruction: Instruction, expected: int
) -> None:
    assert encode_scalar_instruction(instruction) == expected
