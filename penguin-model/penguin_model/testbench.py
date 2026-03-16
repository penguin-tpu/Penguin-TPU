"""Helpers for loading and running checked-in scalar test programs."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

from .arch_state import ArchState
from .assembler import AssemblyProgram, assemble_file
from .core_config import DEFAULT_PENGUIN_CORE_CONFIG, PenguinCoreConfig
from .core import PenguinCore

TEST_DRAM_SIZE = 4 * 1024
TEST_VMEM_SIZE = 4 * 1024
TEST_IMEM_SIZE = 1 * 1024
TEST_PROGRAM_ROOT = Path(__file__).resolve().parents[2] / "tests" / "vectors" / "programs"
TEST_CORE_CONFIG = DEFAULT_PENGUIN_CORE_CONFIG.with_memory_sizes(
    dram_size=TEST_DRAM_SIZE,
    vmem_size=TEST_VMEM_SIZE,
    imem_size=TEST_IMEM_SIZE,
)
DRAM_BASE = TEST_CORE_CONFIG.memory_map.dram.base
VMEM_BASE = TEST_CORE_CONFIG.memory_map.vmem.base


def fresh_arch_state(
    config: PenguinCoreConfig = TEST_CORE_CONFIG,
) -> ArchState:
    return ArchState.from_config(config)


def preload_words(memory, words: dict[int, int]) -> None:
    for offset, value in words.items():
        memory.store_u32(memory.base + offset, value)


def scalar_program_path(program: str | PathLike[str]) -> Path:
    candidate = Path(program)
    if candidate.is_absolute():
        return candidate
    return TEST_PROGRAM_ROOT / candidate


def load_scalar_program(program: str | PathLike[str]) -> AssemblyProgram:
    return assemble_file(scalar_program_path(program))


def run_scalar_program(
    program: str | PathLike[str],
    *,
    start_pc: int = 0,
    max_instructions: int | None = None,
    vmem_words: dict[int, int] | None = None,
    dram_words: dict[int, int] | None = None,
    config: PenguinCoreConfig = TEST_CORE_CONFIG,
) -> tuple[PenguinCore, object]:
    state = fresh_arch_state(config)
    if vmem_words is not None:
        preload_words(state.vmem, vmem_words)
    if dram_words is not None:
        preload_words(state.dram, dram_words)

    core = PenguinCore(state=state, config=config)
    perf = core.execute(
        load_scalar_program(program),
        start_pc=start_pc,
        max_instructions=max_instructions,
    )
    return core, perf


__all__ = [
    "DRAM_BASE",
    "TEST_DRAM_SIZE",
    "TEST_IMEM_SIZE",
    "TEST_PROGRAM_ROOT",
    "TEST_VMEM_SIZE",
    "TEST_CORE_CONFIG",
    "VMEM_BASE",
    "fresh_arch_state",
    "load_scalar_program",
    "preload_words",
    "run_scalar_program",
    "scalar_program_path",
]
