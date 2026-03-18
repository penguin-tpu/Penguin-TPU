"""Imported RISC-V ISA tests executed through the Penguin scalar model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from penguin_model import IMEM_BASE, StopReason, Sim, assemble_text
from penguin_model.testbench import (
    TEST_CORE_CONFIG,
    TEST_PROGRAM_ROOT,
    fresh_arch_state,
    load_scalar_program,
)

RISC_V_ISA_TEST_CONFIG = TEST_CORE_CONFIG.with_memory_sizes(imem_size=8 * 1024)
RISC_V_ISA_PROGRAM_ROOT = TEST_PROGRAM_ROOT / "scalar" / "riscv_isa"
RESULT_REG = 10
TESTNUM_REG = 3

EXPECTED_IMPORTED_RV32UI_STEMS = {
    "add",
    "addi",
    "and",
    "andi",
    "auipc",
    "beq",
    "bge",
    "bgeu",
    "blt",
    "bltu",
    "bne",
    "jal",
    "jalr",
    "lb",
    "lbu",
    "ld_st",
    "lh",
    "lhu",
    "lui",
    "lw",
    "or",
    "ori",
    "sb",
    "sh",
    "simple",
    "sll",
    "slli",
    "slt",
    "slti",
    "sltiu",
    "sltu",
    "sra",
    "srai",
    "srl",
    "srli",
    "st_ld",
    "sub",
    "sw",
    "xor",
    "xori",
}

UPSTREAM_EXCLUDED_RV32UI_STEMS = {"fence_i", "ma_data"}


def _pack_u16(words: list[int]) -> list[int]:
    data: list[int] = []
    for word in words:
        data.extend(int(word & 0xFFFF).to_bytes(2, byteorder="little", signed=False))
    return data


def _pack_u32(words: list[int]) -> list[int]:
    data: list[int] = []
    for word in words:
        data.extend(int(word & 0xFFFF_FFFF).to_bytes(4, byteorder="little", signed=False))
    return data


def _vmem_preload_bytes(stem: str) -> list[int] | None:
    if stem in {"lb", "lbu"}:
        return [0xFF, 0x00, 0xF0, 0x0F]
    if stem in {"lh", "lhu"}:
        return _pack_u16([0x00FF, 0xFF00, 0x0FF0, 0xF00F])
    if stem == "lw":
        return _pack_u32([0x00FF_00FF, 0xFF00_FF00, 0x0FF0_0FF0, 0xF00F_F00F])
    if stem == "sb":
        return [0xEF] * 10
    if stem == "sh":
        return _pack_u16([0xBEEF] * 10)
    if stem == "sw":
        return _pack_u32([0xDEAD_BEEF] * 10)
    if stem in {"ld_st", "st_ld"}:
        return _pack_u32([0xDEAD_BEEF] * 20)
    return None


def _run_riscv_isa_program(program_path: Path) -> tuple[Sim, object]:
    stem = program_path.stem
    state = fresh_arch_state(RISC_V_ISA_TEST_CONFIG)
    preload = _vmem_preload_bytes(stem)
    if preload is not None:
        state.vmem.write(
            state.vmem.base + 0x100,
            torch.tensor(preload, dtype=torch.uint8),
        )
    core = Sim(state=state, config=RISC_V_ISA_TEST_CONFIG)
    perf = core.execute(load_scalar_program(program_path))
    return core, perf


def _riscv_isa_context(name: str, core: Sim, perf: object) -> str:
    return (
        f"{name}: "
        f"stop_reason={core.state.stop_reason}, "
        f"pc=0x{core.state.pc:08x}, "
        f"x{RESULT_REG}=0x{core.state.read_xreg(RESULT_REG):08x}, "
        f"x{TESTNUM_REG}=0x{core.state.read_xreg(TESTNUM_REG):08x}, "
        f"instructions={perf.instructions}, "
        f"cycles={perf.cycles}"
    )


def _assert_riscv_isa_result(
    name: str,
    core: Sim,
    perf: object,
    *,
    expected_result: int,
    expected_testnum: int,
) -> str:
    context = _riscv_isa_context(name, core, perf)
    assert core.state.stop_reason == StopReason.ECALL, context
    assert core.state.read_xreg(RESULT_REG) == expected_result, context
    assert core.state.read_xreg(TESTNUM_REG) == expected_testnum, context
    return context


@pytest.mark.parametrize(
    "program_path",
    sorted(RISC_V_ISA_PROGRAM_ROOT.glob("*.S")),
    ids=lambda path: path.stem,
)
def test_imported_rv32ui_program_passes(program_path: Path) -> None:
    core, perf = _run_riscv_isa_program(program_path)
    context = _assert_riscv_isa_result(
        program_path.stem,
        core,
        perf,
        expected_result=0,
        expected_testnum=0,
    )
    assert perf.instructions > 0, context


def test_imported_rv32ui_program_set_matches_supported_upstream_scope() -> None:
    imported_stems = {path.stem for path in RISC_V_ISA_PROGRAM_ROOT.glob("*.S")}
    assert imported_stems == EXPECTED_IMPORTED_RV32UI_STEMS
    assert imported_stems.isdisjoint(UPSTREAM_EXCLUDED_RV32UI_STEMS)


def test_rv32ui_store_and_load_interaction_updates_memory() -> None:
    program_path = RISC_V_ISA_PROGRAM_ROOT / "st_ld.S"
    core, perf = _run_riscv_isa_program(program_path)
    context = _assert_riscv_isa_result(
        "st_ld",
        core,
        perf,
        expected_result=0,
        expected_testnum=0,
    )
    assert core.state.vmem.load_u8(core.state.vmem.base + 0x103) == 0xEF, context


def test_riscv_isa_reporting_convention_preserves_failure_number() -> None:
    program = assemble_text(
        """
    li x3, 7
    li x10, 1
    secall
""",
        base_address=IMEM_BASE,
    )
    state = fresh_arch_state(RISC_V_ISA_TEST_CONFIG)
    core = Sim(state=state, config=RISC_V_ISA_TEST_CONFIG)
    perf = core.execute(program)

    _assert_riscv_isa_result(
        "reporting-convention",
        core,
        perf,
        expected_result=1,
        expected_testnum=7,
    )
