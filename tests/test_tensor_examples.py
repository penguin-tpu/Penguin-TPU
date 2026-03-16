"""Integration tests for runnable tensor examples."""

from __future__ import annotations

import torch

from penguin_model import (
    PenguinCoreConfig,
    run_large_linear_example,
    run_large_matmul_example,
    run_linear_example,
    run_matmul_example,
)


def test_matmul_example_matches_pytorch_and_emits_trace(tmp_path) -> None:
    result = run_matmul_example(
        trace_path=tmp_path / "matmul_trace.json",
        config=PenguinCoreConfig(),
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (64, 16)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 38
    assert result.perf.cycles == 386


def test_linear_example_matches_pytorch_and_emits_trace(tmp_path) -> None:
    result = run_linear_example(
        trace_path=tmp_path / "linear_trace.json",
        config=PenguinCoreConfig(),
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (128, 32)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 59
    assert result.perf.cycles == 1_393


def test_large_matmul_example_matches_pytorch_and_emits_trace(tmp_path) -> None:
    result = run_large_matmul_example(
        trace_path=tmp_path / "matmul_large_trace.json",
        config=PenguinCoreConfig(),
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (128, 32)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 223
    assert result.perf.cycles == 14_795
    assert result.perf.bytes_read == 49_152
    assert result.perf.bytes_written == 36_864
    assert result.perf.instructions_by_opcode["sbeq"] > 0
    assert result.perf.instructions_by_opcode["sbne"] > 0
    assert result.perf.instructions_by_opcode["sjal"] > 0


def test_large_linear_example_matches_pytorch_and_emits_trace(tmp_path) -> None:
    result = run_large_linear_example(
        trace_path=tmp_path / "linear_large_trace.json",
        config=PenguinCoreConfig(),
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (192, 48)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 468
    assert result.perf.cycles == 34_407
    assert result.perf.bytes_read == 129_024
    assert result.perf.bytes_written == 82_944
    assert result.perf.instructions_by_opcode["sbeq"] > 0
    assert result.perf.instructions_by_opcode["sbne"] > 0
    assert result.perf.instructions_by_opcode["sjal"] > 0
    assert result.perf.instructions_by_opcode["vadd"] > 0
