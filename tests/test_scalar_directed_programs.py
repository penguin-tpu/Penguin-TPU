"""Directed scalar program tests for the Penguin functional/perf model."""

from __future__ import annotations

from typing import Any

import pytest

from penguin_model import StopReason
from penguin_model.testbench import fresh_arch_state, run_scalar_program

FAIL_REG = 31


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
    power_on_fail_reg = fresh_arch_state().read_xreg(FAIL_REG)
    assert core.state.stop_reason == StopReason.PROGRAM_END, context
    assert core.state.read_xreg(FAIL_REG) in {0, power_on_fail_reg}, context
    assert perf.instructions == expected_instructions, context
    assert perf.cycles == expected_cycles, context
    assert perf.bytes_read == expected_bytes_read, context
    assert perf.bytes_written == expected_bytes_written, context
    return context


@pytest.mark.parametrize(
    ("name", "program"),
    [
        pytest.param("sauipc", "scalar/directed/u_sauipc.S", id="sauipc"),
        pytest.param("slui", "scalar/directed/u_slui.S", id="slui"),
    ],
)
def test_scalar_u_directed_program_case(name: str, program: str) -> None:
    core, perf = run_scalar_program(program)
    _assert_program_passed(name, core, perf, expected_instructions=6, expected_cycles=6)


@pytest.mark.parametrize(
    ("name", "program"),
    [
        pytest.param("sadd", "scalar/directed/r_sadd.S", id="sadd"),
        pytest.param("ssub", "scalar/directed/r_ssub.S", id="ssub"),
        pytest.param("sslt", "scalar/directed/r_sslt.S", id="sslt"),
        pytest.param("ssltu", "scalar/directed/r_ssltu.S", id="ssltu"),
        pytest.param("sxor", "scalar/directed/r_sxor.S", id="sxor"),
        pytest.param("sor", "scalar/directed/r_sor.S", id="sor"),
        pytest.param("sand", "scalar/directed/r_sand.S", id="sand"),
        pytest.param("ssll", "scalar/directed/r_ssll.S", id="ssll"),
        pytest.param("ssrl", "scalar/directed/r_ssrl.S", id="ssrl"),
        pytest.param("ssra", "scalar/directed/r_ssra.S", id="ssra"),
    ],
)
def test_scalar_rr_directed_program_case(name: str, program: str) -> None:
    core, perf = run_scalar_program(program)
    _assert_program_passed(name, core, perf, expected_instructions=8, expected_cycles=8)


@pytest.mark.parametrize(
    ("name", "program"),
    [
        pytest.param("saddi", "scalar/directed/i_saddi.S", id="saddi"),
        pytest.param("sslti", "scalar/directed/i_sslti.S", id="sslti"),
        pytest.param("ssltiu", "scalar/directed/i_ssltiu.S", id="ssltiu"),
        pytest.param("sxori", "scalar/directed/i_sxori.S", id="sxori"),
        pytest.param("sori", "scalar/directed/i_sori.S", id="sori"),
        pytest.param("sandi", "scalar/directed/i_sandi.S", id="sandi"),
        pytest.param("sslli", "scalar/directed/i_sslli.S", id="sslli"),
        pytest.param("ssrli", "scalar/directed/i_ssrli.S", id="ssrli"),
        pytest.param("ssrai", "scalar/directed/i_ssrai.S", id="ssrai"),
    ],
)
def test_scalar_imm_directed_program_case(name: str, program: str) -> None:
    core, perf = run_scalar_program(program)
    _assert_program_passed(name, core, perf, expected_instructions=7, expected_cycles=7)


@pytest.mark.parametrize(
    ("name", "program"),
    [
        pytest.param("sbeq-taken", "scalar/directed/branch_sbeq_taken.S", id="sbeq-taken"),
        pytest.param(
            "sbeq-not-taken",
            "scalar/directed/branch_sbeq_not_taken.S",
            id="sbeq-not-taken",
        ),
        pytest.param("sbne-taken", "scalar/directed/branch_sbne_taken.S", id="sbne-taken"),
        pytest.param(
            "sbne-not-taken",
            "scalar/directed/branch_sbne_not_taken.S",
            id="sbne-not-taken",
        ),
        pytest.param("sblt-taken", "scalar/directed/branch_sblt_taken.S", id="sblt-taken"),
        pytest.param(
            "sblt-not-taken",
            "scalar/directed/branch_sblt_not_taken.S",
            id="sblt-not-taken",
        ),
        pytest.param("sbge-taken", "scalar/directed/branch_sbge_taken.S", id="sbge-taken"),
        pytest.param(
            "sbge-not-taken",
            "scalar/directed/branch_sbge_not_taken.S",
            id="sbge-not-taken",
        ),
        pytest.param("sbltu-taken", "scalar/directed/branch_sbltu_taken.S", id="sbltu-taken"),
        pytest.param(
            "sbltu-not-taken",
            "scalar/directed/branch_sbltu_not_taken.S",
            id="sbltu-not-taken",
        ),
        pytest.param("sbgeu-taken", "scalar/directed/branch_sbgeu_taken.S", id="sbgeu-taken"),
        pytest.param(
            "sbgeu-not-taken",
            "scalar/directed/branch_sbgeu_not_taken.S",
            id="sbgeu-not-taken",
        ),
    ],
)
def test_scalar_branch_directed_program_case(name: str, program: str) -> None:
    core, perf = run_scalar_program(program)
    _assert_program_passed(name, core, perf, expected_instructions=8, expected_cycles=8)


@pytest.mark.parametrize(
    ("name", "program", "expected_instructions"),
    [
        pytest.param("sjal", "scalar/directed/jump_sjal.S", 8, id="sjal"),
        pytest.param("sjalr", "scalar/directed/jump_sjalr.S", 9, id="sjalr"),
    ],
)
def test_scalar_jump_directed_program_case(
    name: str, program: str, expected_instructions: int
) -> None:
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
    core, perf = run_scalar_program("scalar/directed/load.S", vmem_words={0x40: 3})
    _assert_program_passed(
        "sld",
        core,
        perf,
        expected_instructions=7,
        expected_cycles=7,
        expected_bytes_read=4,
    )


def test_scalar_store_directed_program_case() -> None:
    core, perf = run_scalar_program(
        "scalar/directed/store.S",
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
    assert core.state.vmem.load_u32(0x0800_0060) == 8, context
    assert core.state.dram.load_u32(0x8000_0060) == 0xDEAD_0003, context


def test_scalar_x0_load_directed_program_case() -> None:
    core, perf = run_scalar_program("scalar/directed/x0_load.S", vmem_words={0x48: 7})
    _assert_program_passed(
        "sld-x0",
        core,
        perf,
        expected_instructions=7,
        expected_cycles=7,
        expected_bytes_read=4,
    )


def test_scalar_sfence_directed_program_case() -> None:
    core, perf = run_scalar_program("scalar/directed/sfence.S")
    _assert_program_passed("sfence", core, perf, expected_instructions=4, expected_cycles=4)
