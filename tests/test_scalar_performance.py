"""Performance-oriented scalar regression tests."""

from __future__ import annotations

from penguin_model.testbench import VMEM_BASE, run_scalar_program


def test_scalar_benchmark_dma_stage_and_reduce_perf() -> None:
    values = [1, 2, 3, 4, 5, 6, 7, 8]

    core, perf = run_scalar_program(
        "scalar/performance/dma_stage_and_reduce.S",
        dram_words={0x100 + index * 4: value for index, value in enumerate(values)},
    )

    assert core.state.vmem.load_u32(VMEM_BASE + 0x280) == sum(values)
    assert perf.instructions == 67
    assert perf.cycles == 89
    assert perf.bytes_read == 64
    assert perf.bytes_written == 36


def test_scalar_benchmark_address_generation_perf() -> None:
    core, perf = run_scalar_program("scalar/performance/address_generation.S")

    assert core.state.vmem.load_u32(VMEM_BASE + 0x500) == VMEM_BASE + 0x400
    assert perf.instructions == 60
    assert perf.cycles == 63
    assert perf.bytes_read == 0
    assert perf.bytes_written == 24


def test_scalar_benchmark_dma_overlap_hides_transfer_latency() -> None:
    core, perf = run_scalar_program(
        "scalar/performance/dma_overlap.S",
        dram_words={0x100 + index * 4: value for index, value in enumerate(range(1, 9))},
    )

    assert [core.state.vmem.load_u32(VMEM_BASE + 0x100 + index * 4) for index in range(8)] == list(
        range(1, 9)
    )
    assert core.state.read_xreg(20) == 9
    assert perf.instructions == 16
    assert perf.cycles == 28
    assert perf.bytes_read == 32
    assert perf.bytes_written == 32
