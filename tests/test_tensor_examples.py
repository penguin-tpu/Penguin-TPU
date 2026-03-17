"""Integration tests for runnable tensor examples."""

from __future__ import annotations

import torch
from trace_utils import trace_output_path

from penguin_model import (
    PenguinCoreConfig,
    run_large_linear_example,
    run_large_matmul_example,
    run_linear_example,
    run_matmul_example,
)


def test_matmul_example_matches_pytorch_and_emits_trace(tmp_path) -> None:
    result = run_matmul_example(
        trace_path=trace_output_path("tensor_example_matmul.json"),
        config=PenguinCoreConfig(),
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (64, 64)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 41
    assert result.perf.cycles == 1_061


def test_linear_example_matches_pytorch_and_emits_trace(tmp_path) -> None:
    result = run_linear_example(
        trace_path=trace_output_path("tensor_example_linear.json"),
        config=PenguinCoreConfig(),
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (128, 128)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 71
    assert result.perf.cycles == 4_155


def test_large_matmul_example_matches_pytorch_and_emits_trace(tmp_path) -> None:
    result = run_large_matmul_example(
        trace_path=trace_output_path("tensor_example_matmul_large.json"),
        config=PenguinCoreConfig(),
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (128, 128)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 229
    assert result.perf.cycles == 39_141
    assert result.perf.bytes_read == 163_840
    assert result.perf.bytes_written == 131_072
    assert result.perf.instructions_by_opcode["sbeq"] > 0
    assert result.perf.instructions_by_opcode["sbne"] > 0
    assert result.perf.instructions_by_opcode["sjal"] > 0


def test_large_linear_example_matches_pytorch_and_emits_trace(tmp_path) -> None:
    result = run_large_linear_example(
        trace_path=trace_output_path("tensor_example_linear_large.json"),
        config=PenguinCoreConfig(),
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (192, 192)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 497
    assert result.perf.cycles == 92_648
    assert result.perf.bytes_read == 442_368
    assert result.perf.bytes_written == 294_912
    assert result.perf.instructions_by_opcode["sbeq"] > 0
    assert result.perf.instructions_by_opcode["sbne"] > 0
    assert result.perf.instructions_by_opcode["sjal"] > 0
    assert result.perf.instructions_by_opcode["vadd"] > 0
