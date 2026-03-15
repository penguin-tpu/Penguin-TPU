"""Directed scalar program tests for the Penguin functional/perf model."""

from __future__ import annotations

from typing import Any

import pytest

from penguin_model import StopReason
from penguin_model.testbench import DRAM_BASE, VMEM_BASE, ScalarProgramBuilder, run_scalar_program

FAIL_REG = 31
SCRATCH_REG = 30


def _li(builder: ScalarProgramBuilder, rd: int, value: int) -> None:
    builder.i("saddi", rd=rd, rs1=0, imm=value)


def _expect_reg_eq(
    builder: ScalarProgramBuilder, *, reg: int, expected: int, fail_code: int = 1
) -> None:
    _li(builder, SCRATCH_REG, expected)
    builder.branch("sbne", rs1=reg, rs2=SCRATCH_REG, target=f"fail_{fail_code}")


def _expect_reg_eq_label(
    builder: ScalarProgramBuilder, *, reg: int, label: str, fail_code: int = 1, offset: int = 0
) -> None:
    builder.li_label(rd=SCRATCH_REG, target=label, offset=offset)
    builder.branch("sbne", rs1=reg, rs2=SCRATCH_REG, target=f"fail_{fail_code}")


def _add_fail_handlers(builder: ScalarProgramBuilder, fail_codes: range) -> None:
    for fail_code in fail_codes:
        builder.label(f"fail_{fail_code}")
        _li(builder, FAIL_REG, fail_code)
        builder.jal(rd=0, target="done")
        builder.delay_slots()


def _finish_program(builder: ScalarProgramBuilder) -> list:
    builder.jal(rd=0, target="done")
    builder.delay_slots()
    _add_fail_handlers(builder, range(1, 2))
    builder.label("done")
    return builder.build()


def _directed_test_context(name: str, core: Any, perf: Any) -> str:
    return (
        f"{name}: "
        f"stop_reason={core.state.stop_reason}, "
        f"pc=0x{core.state.pc:08x}, "
        f"fail_reg=x{FAIL_REG}=0x{core.state.read_xreg(FAIL_REG):08x}, "
        f"x3=0x{core.state.read_xreg(3):08x}, "
        f"x10=0x{core.state.read_xreg(10):08x}, "
        f"x11=0x{core.state.read_xreg(11):08x}, "
        f"instructions={perf.instructions}, "
        f"cycles={perf.cycles}, "
        f"bytes_read={perf.bytes_read}, "
        f"bytes_written={perf.bytes_written}"
    )


def _assert_program_passed(
    name: str,
    core: Any,
    perf: Any,
    *,
    expected_instructions: int,
    expected_cycles: int,
    expected_bytes_read: int = 0,
    expected_bytes_written: int = 0,
) -> str:
    context = _directed_test_context(name, core, perf)
    assert core.state.stop_reason == StopReason.PROGRAM_END, context
    assert core.state.read_xreg(FAIL_REG) == 0, context
    assert perf.instructions == expected_instructions, context
    assert perf.cycles == expected_cycles, context
    assert perf.bytes_read == expected_bytes_read, context
    assert perf.bytes_written == expected_bytes_written, context
    return context


def _build_u_case(mnemonic: str, imm: int, expected: int) -> list:
    builder = ScalarProgramBuilder()
    builder.u(mnemonic, rd=3, imm=imm)
    _expect_reg_eq(builder, reg=3, expected=expected)
    return _finish_program(builder)


def _build_r_case(mnemonic: str, lhs: int, rhs: int, expected: int) -> list:
    builder = ScalarProgramBuilder()
    _li(builder, 1, lhs)
    _li(builder, 2, rhs)
    builder.r(mnemonic, rd=3, rs1=1, rs2=2)
    _expect_reg_eq(builder, reg=3, expected=expected)
    return _finish_program(builder)


def _build_i_case(mnemonic: str, lhs: int, imm: int, expected: int) -> list:
    builder = ScalarProgramBuilder()
    _li(builder, 1, lhs)
    builder.i(mnemonic, rd=3, rs1=1, imm=imm)
    _expect_reg_eq(builder, reg=3, expected=expected)
    return _finish_program(builder)


def _build_branch_case(mnemonic: str, lhs: int, rhs: int, taken: bool) -> list:
    builder = ScalarProgramBuilder()
    _li(builder, 1, lhs)
    _li(builder, 2, rhs)
    if taken:
        builder.branch(mnemonic, rs1=1, rs2=2, target="pass")
        builder.delay_slots()
        builder.jal(rd=0, target="fail_1")
        builder.delay_slots()
        builder.label("pass")
    else:
        builder.branch(mnemonic, rs1=1, rs2=2, target="fail_1")
        builder.delay_slots()
    return _finish_program(builder)


def _build_jal_case() -> list:
    builder = ScalarProgramBuilder()
    builder.jal(rd=10, target="target")
    builder.delay_slots()
    builder.label("link")
    builder.jal(rd=0, target="fail_1")
    builder.delay_slots()
    builder.label("target")
    _expect_reg_eq_label(builder, reg=10, label="link")
    return _finish_program(builder)


def _build_jalr_case() -> list:
    builder = ScalarProgramBuilder()
    builder.li_label(rd=12, target="target", offset=1)
    builder.i("sjalr", rd=11, rs1=12, imm=0)
    builder.delay_slots()
    builder.label("link")
    builder.jal(rd=0, target="fail_1")
    builder.delay_slots()
    builder.label("target")
    _expect_reg_eq_label(builder, reg=11, label="link")
    return _finish_program(builder)


def _build_load_case() -> list:
    builder = ScalarProgramBuilder()
    _li(builder, 1, VMEM_BASE + 0x40)
    builder.i("sld", rd=3, rs1=1, imm=0)
    _expect_reg_eq(builder, reg=3, expected=3)
    return _finish_program(builder)


def _build_store_case() -> list:
    builder = ScalarProgramBuilder()
    _li(builder, 1, VMEM_BASE + 0x60)
    _li(builder, 2, 8)
    builder.s("sst", rs1=1, rs2=2, imm=0)
    builder.i("sld", rd=3, rs1=1, imm=0)
    _expect_reg_eq(builder, reg=3, expected=8)
    return _finish_program(builder)


def _build_x0_load_case() -> list:
    builder = ScalarProgramBuilder()
    _li(builder, 1, VMEM_BASE + 0x48)
    builder.i("sld", rd=0, rs1=1, imm=0)
    _expect_reg_eq(builder, reg=0, expected=0)
    return _finish_program(builder)


def _build_sfence_case() -> list:
    builder = ScalarProgramBuilder()
    builder.empty("sfence")
    return _finish_program(builder)


@pytest.mark.parametrize(
    ("mnemonic", "imm", "expected"),
    [
        pytest.param("sauipc", 1, 0x0000_1000, id="sauipc"),
        pytest.param("slui", 0x12345, 0x1234_5000, id="slui"),
    ],
)
def test_scalar_u_directed_program_case(mnemonic: str, imm: int, expected: int) -> None:
    program = _build_u_case(mnemonic, imm, expected)
    core, perf = run_scalar_program(program)
    _assert_program_passed(mnemonic, core, perf, expected_instructions=6, expected_cycles=6)


@pytest.mark.parametrize(
    ("mnemonic", "lhs", "rhs", "expected"),
    [
        pytest.param("sadd", 7, 3, 10, id="sadd"),
        pytest.param("ssub", 7, 3, 4, id="ssub"),
        pytest.param("sslt", -16, 3, 1, id="sslt"),
        pytest.param("ssltu", 3, -16, 1, id="ssltu"),
        pytest.param("sxor", 7, 3, 4, id="sxor"),
        pytest.param("sor", 7, 3, 7, id="sor"),
        pytest.param("sand", 7, 3, 3, id="sand"),
        pytest.param("ssll", 3, 3, 24, id="ssll"),
        pytest.param("ssrl", -16, 3, 0x1FFF_FFFE, id="ssrl"),
        pytest.param("ssra", -16, 3, 0xFFFF_FFFE, id="ssra"),
    ],
)
def test_scalar_rr_directed_program_case(
    mnemonic: str, lhs: int, rhs: int, expected: int
) -> None:
    program = _build_r_case(mnemonic, lhs, rhs, expected)
    core, perf = run_scalar_program(program)
    _assert_program_passed(mnemonic, core, perf, expected_instructions=8, expected_cycles=8)


@pytest.mark.parametrize(
    ("mnemonic", "lhs", "imm", "expected"),
    [
        pytest.param("saddi", 7, -2, 5, id="saddi"),
        pytest.param("sslti", -16, -8, 1, id="sslti"),
        pytest.param("ssltiu", 3, -1, 1, id="ssltiu"),
        pytest.param("sxori", 7, 0xF, 8, id="sxori"),
        pytest.param("sori", 3, 0x8, 11, id="sori"),
        pytest.param("sandi", 7, 0x6, 6, id="sandi"),
        pytest.param("sslli", 3, 4, 48, id="sslli"),
        pytest.param("ssrli", -16, 4, 0x0FFF_FFFF, id="ssrli"),
        pytest.param("ssrai", -16, 4, 0xFFFF_FFFF, id="ssrai"),
    ],
)
def test_scalar_imm_directed_program_case(
    mnemonic: str, lhs: int, imm: int, expected: int
) -> None:
    program = _build_i_case(mnemonic, lhs, imm, expected)
    core, perf = run_scalar_program(program)
    _assert_program_passed(mnemonic, core, perf, expected_instructions=7, expected_cycles=7)


@pytest.mark.parametrize(
    ("mnemonic", "lhs", "rhs", "taken"),
    [
        pytest.param("sbeq", 5, 5, True, id="sbeq-taken"),
        pytest.param("sbeq", 5, 6, False, id="sbeq-not-taken"),
        pytest.param("sbne", 5, 6, True, id="sbne-taken"),
        pytest.param("sbne", 5, 5, False, id="sbne-not-taken"),
        pytest.param("sblt", -1, 1, True, id="sblt-taken"),
        pytest.param("sblt", 4, -1, False, id="sblt-not-taken"),
        pytest.param("sbge", 7, -3, True, id="sbge-taken"),
        pytest.param("sbge", -4, 2, False, id="sbge-not-taken"),
        pytest.param("sbltu", 1, 2, True, id="sbltu-taken"),
        pytest.param("sbltu", -1, 2, False, id="sbltu-not-taken"),
        pytest.param("sbgeu", -1, 2, True, id="sbgeu-taken"),
        pytest.param("sbgeu", 1, -1, False, id="sbgeu-not-taken"),
    ],
)
def test_scalar_branch_directed_program_case(
    mnemonic: str, lhs: int, rhs: int, taken: bool
) -> None:
    program = _build_branch_case(mnemonic, lhs, rhs, taken)
    core, perf = run_scalar_program(program)
    _assert_program_passed(mnemonic, core, perf, expected_instructions=8, expected_cycles=8)


@pytest.mark.parametrize(
    ("name", "builder_fn", "expected_instructions"),
    [
        pytest.param("sjal", _build_jal_case, 8, id="sjal"),
        pytest.param("sjalr", _build_jalr_case, 9, id="sjalr"),
    ],
)
def test_scalar_jump_directed_program_case(
    name: str, builder_fn: Any, expected_instructions: int
) -> None:
    program = builder_fn()
    core, perf = run_scalar_program(program)
    context = _assert_program_passed(
        name,
        core,
        perf,
        expected_instructions=expected_instructions,
        expected_cycles=expected_instructions,
    )
    if name == "sjal":
        assert core.state.read_xreg(10) != 0, context
    else:
        assert core.state.read_xreg(11) != 0, context


def test_scalar_load_directed_program_case() -> None:
    program = _build_load_case()
    core, perf = run_scalar_program(program, vmem_words={0x40: 3})
    _assert_program_passed(
        "sld",
        core,
        perf,
        expected_instructions=7,
        expected_cycles=7,
        expected_bytes_read=4,
    )


def test_scalar_store_directed_program_case() -> None:
    program = _build_store_case()
    core, perf = run_scalar_program(
        program,
        dram_words={0x60: 0xDEAD_0003},
    )
    context = _assert_program_passed(
        "sst",
        core,
        perf,
        expected_instructions=9,
        expected_cycles=9,
        expected_bytes_read=4,
        expected_bytes_written=4,
    )
    assert core.state.vmem.load_u32(VMEM_BASE + 0x60) == 8, context
    assert core.state.dram.load_u32(DRAM_BASE + 0x60) == 0xDEAD_0003, context


def test_scalar_x0_load_directed_program_case() -> None:
    program = _build_x0_load_case()
    core, perf = run_scalar_program(program, vmem_words={0x48: 7})
    _assert_program_passed(
        "sld-x0",
        core,
        perf,
        expected_instructions=7,
        expected_cycles=7,
        expected_bytes_read=4,
    )


def test_scalar_sfence_directed_program_case() -> None:
    program = _build_sfence_case()
    core, perf = run_scalar_program(program)
    _assert_program_passed("sfence", core, perf, expected_instructions=4, expected_cycles=4)
