"""Scalar workload-style tests for the Penguin functional model."""

from __future__ import annotations

from penguin_model import StopReason
from penguin_model.testbench import VMEM_BASE, run_scalar_program


def test_scalar_workload_reduces_staged_tensor_tile() -> None:
    values = [1, 2, 3, 4, 5, 6, 7, 8]

    core, perf = run_scalar_program(
        "scalar/workloads/dma_stage_and_reduce.S",
        dram_words={0x100 + index * 4: value for index, value in enumerate(values)},
    )

    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert core.state.vmem.load_u32(VMEM_BASE + 0x280) == sum(values)
    assert [core.state.vmem.load_u32(VMEM_BASE + 0x200 + index * 4) for index in range(8)] == values
    assert perf.bytes_read == 64
    assert perf.bytes_written == 36


def test_scalar_workload_generates_row_major_tensor_addresses() -> None:
    core, perf = run_scalar_program("scalar/workloads/address_generation.S")

    expected = [
        VMEM_BASE + 0x400,
        VMEM_BASE + 0x404,
        VMEM_BASE + 0x408,
        VMEM_BASE + 0x420,
        VMEM_BASE + 0x424,
        VMEM_BASE + 0x428,
    ]
    actual = [core.state.vmem.load_u32(VMEM_BASE + 0x500 + index * 4) for index in range(6)]

    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert actual == expected
    assert perf.bytes_written == 24


def test_scalar_workload_copies_tile_and_accumulates_checksum() -> None:
    values = [9, 4, 7, 1, 5, 3]

    core, perf = run_scalar_program(
        "scalar/workloads/copy_and_checksum.S",
        vmem_words={0x600 + index * 4: value for index, value in enumerate(values)},
    )

    copied = [core.state.vmem.load_u32(VMEM_BASE + 0x700 + index * 4) for index in range(len(values))]

    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert copied == values
    assert core.state.vmem.load_u32(VMEM_BASE + 0x780) == sum(values)
    assert perf.bytes_read == len(values) * 4
    assert perf.bytes_written == len(values) * 4 + 4
