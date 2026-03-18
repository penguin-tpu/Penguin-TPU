from __future__ import annotations

from dataclasses import dataclass

import cocotb
from cocotb.triggers import Timer

from penguin_model import BType, EmptyType, IType, Instruction, JType, RType, SType, UType, VPUBinaryType
from penguin_model.scalar_encoding import encode_scalar_instruction


FMT_R = 0
FMT_I = 1
FMT_S = 2
FMT_B = 3
FMT_U = 4
FMT_J = 5
FMT_SYSTEM = 6
FMT_MISC_MEM = 7
FMT_RESERVED = 8

OP_ALU_REG = 0
OP_ALU_IMM = 1
OP_LOAD = 2
OP_STORE = 3
OP_BRANCH = 4
OP_JUMP = 5
OP_UPPER_IMM = 6
OP_SYSTEM = 7

ALU_ADD = 0
ALU_SUB = 1
ALU_SLT = 2
ALU_SLTU = 3
ALU_XOR = 4
ALU_OR = 5
ALU_AND = 6
ALU_SLL = 7
ALU_SRL = 8
ALU_SRA = 9
ALU_COMPARE_EQ = 11
ALU_COMPARE_NE = 12
ALU_COMPARE_LT = 13
ALU_COMPARE_GE = 14
ALU_COMPARE_LTU = 15
ALU_COMPARE_GEU = 16


@dataclass(frozen=True)
class DecodeExpectation:
    fmt: int
    op_class: int
    alu_fn: int
    rd: int
    rs1: int
    rs2: int
    imm32: int
    writes_rd: int = 0
    reads_rs1: int = 0
    reads_rs2: int = 0
    is_branch: int = 0
    is_jump: int = 0
    is_load: int = 0
    is_store: int = 0
    is_fence: int = 0
    is_ecall: int = 0
    is_ebreak: int = 0
    is_vadd: int = 0
    is_reserved_custom: int = 0
    illegal: int = 0


async def check_decode(dut, instruction_word: int, expected: DecodeExpectation) -> None:
    dut.instruction_word.value = instruction_word
    await Timer(1, units="ns")

    assert int(dut.valid.value) == 1
    assert int(dut.illegal.value) == expected.illegal
    assert int(dut.format_class.value) == expected.fmt
    assert int(dut.scalar_op_class.value) == expected.op_class
    assert int(dut.alu_fn.value) == expected.alu_fn
    assert int(dut.rd.value) == expected.rd
    assert int(dut.rs1.value) == expected.rs1
    assert int(dut.rs2.value) == expected.rs2
    assert int(dut.imm32.value.signed_integer) == expected.imm32
    assert int(dut.writes_rd.value) == expected.writes_rd
    assert int(dut.reads_rs1.value) == expected.reads_rs1
    assert int(dut.reads_rs2.value) == expected.reads_rs2
    assert int(dut.is_branch.value) == expected.is_branch
    assert int(dut.is_jump.value) == expected.is_jump
    assert int(dut.is_load.value) == expected.is_load
    assert int(dut.is_store.value) == expected.is_store
    assert int(dut.is_fence.value) == expected.is_fence
    assert int(dut.is_ecall.value) == expected.is_ecall
    assert int(dut.is_ebreak.value) == expected.is_ebreak
    assert int(dut.is_vadd.value) == expected.is_vadd
    assert int(dut.is_reserved_custom.value) == expected.is_reserved_custom


@cocotb.test()
async def decoder_handles_scalar_formats_and_reserved_custom(dut) -> None:
    cases = [
        (
            Instruction("saddi", IType(rd=5, rs1=1, imm=-4)),
            DecodeExpectation(FMT_I, OP_ALU_IMM, ALU_ADD, 5, 1, 28, -4, 1, 1),
        ),
        (
            Instruction("lw", IType(rd=7, rs1=2, imm=12)),
            DecodeExpectation(FMT_I, OP_LOAD, ALU_ADD, 7, 2, 12, 12, 1, 1, 0, 0, 0, 1),
        ),
        (
            Instruction("sw", SType(rs1=3, rs2=4, imm=-8)),
            DecodeExpectation(FMT_S, OP_STORE, ALU_ADD, 24, 3, 4, -8, 0, 1, 1, 0, 0, 0, 1),
        ),
        (
            Instruction("ssub", RType(rd=8, rs1=9, rs2=10)),
            DecodeExpectation(FMT_R, OP_ALU_REG, ALU_SUB, 8, 9, 10, 0, 1, 1, 1),
        ),
        (
            Instruction("sbne", BType(rs1=11, rs2=12, imm=20)),
            DecodeExpectation(FMT_B, OP_BRANCH, ALU_COMPARE_NE, 20, 11, 12, 20, 0, 1, 1, 1),
        ),
        (
            Instruction("slui", UType(rd=13, imm=0x12345)),
            DecodeExpectation(FMT_U, OP_UPPER_IMM, 10, 13, 8, 3, 0x12345000, 1),
        ),
        (
            Instruction("sjal", JType(rd=14, imm=24)),
            DecodeExpectation(FMT_J, OP_JUMP, ALU_ADD, 14, 0, 24, 24, 1, 0, 0, 0, 1),
        ),
        (
            Instruction("sfence", EmptyType()),
            DecodeExpectation(FMT_MISC_MEM, OP_SYSTEM, ALU_ADD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        ),
        (
            Instruction("secall", EmptyType()),
            DecodeExpectation(FMT_SYSTEM, OP_SYSTEM, ALU_ADD, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        ),
        (
            Instruction("sebreak", EmptyType()),
            DecodeExpectation(FMT_SYSTEM, OP_SYSTEM, ALU_ADD, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        ),
        (
            Instruction("vadd", VPUBinaryType(md=2, ms1=0, ms2=1)),
            DecodeExpectation(
                FMT_R,
                OP_ALU_IMM,
                ALU_ADD,
                2,
                0,
                1,
                0,
                reads_rs1=1,
                reads_rs2=1,
                is_vadd=1,
            ),
        ),
    ]

    for instruction, expected in cases:
        await check_decode(dut, encode_scalar_instruction(instruction), expected)

    await check_decode(
        dut,
        0x0000000B,
        DecodeExpectation(FMT_RESERVED, OP_ALU_IMM, ALU_ADD, 0, 0, 0, 0, illegal=1, is_reserved_custom=1),
    )

    await check_decode(
        dut,
        0xFFFFFFFF,
        DecodeExpectation(FMT_RESERVED, OP_ALU_IMM, ALU_ADD, 31, 31, 31, 0, illegal=1, is_reserved_custom=1),
    )
