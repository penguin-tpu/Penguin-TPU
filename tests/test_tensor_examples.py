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
    assert result.perf.instructions == 61
    assert result.perf.cycles == 6_592
    assert result.perf.instructions_by_opcode["delay"] == 5
    assert result.perf.instructions_by_opcode["dma.load.ch0"] == 1
    assert result.perf.instructions_by_opcode["dma.load.ch1"] == 1
    assert result.perf.instructions_by_opcode["dma.store.ch2"] == 1


def test_linear_example_matches_pytorch_and_emits_trace(tmp_path) -> None:
    result = run_linear_example(
        trace_path=trace_output_path("tensor_example_linear.json"),
        config=PenguinCoreConfig(),
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (128, 128)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 127
    assert result.perf.cycles == 9_402
    assert result.perf.instructions_by_opcode["delay"] == 17
    assert result.perf.instructions_by_opcode["dma.load.ch0"] == 1
    assert result.perf.instructions_by_opcode["dma.load.ch5"] == 1
    assert result.perf.instructions_by_opcode["dma.store.ch3"] == 1


def test_large_matmul_example_matches_pytorch_and_emits_trace(tmp_path) -> None:
    result = run_large_matmul_example(
        trace_path=trace_output_path("tensor_example_matmul_large.json"),
        config=PenguinCoreConfig(),
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (128, 128)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions > 200
    assert result.perf.cycles > 30_000
    assert result.perf.bytes_read == 163_840
    assert result.perf.bytes_written == 131_072
    assert result.perf.instructions_by_opcode["bne"] > 0
    assert result.perf.instructions_by_opcode["jal"] > 0


def test_large_linear_example_matches_pytorch_and_emits_trace(tmp_path) -> None:
    result = run_large_linear_example(
        trace_path=trace_output_path("tensor_example_linear_large.json"),
        config=PenguinCoreConfig(),
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (192, 192)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions > 450
    assert result.perf.cycles > 70_000
    assert result.perf.bytes_read == 466_944
    assert result.perf.bytes_written == 319_488
    assert result.perf.instructions_by_opcode["bne"] > 0
    assert result.perf.instructions_by_opcode["jal"] > 0
    assert result.perf.instructions_by_opcode["vadd.bf16"] > 0
    assert result.perf.instructions_by_opcode["dma.load.ch4"] == 1
