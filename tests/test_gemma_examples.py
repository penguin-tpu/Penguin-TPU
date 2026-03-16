"""Integration tests for fixed-shape Gemma-inspired example workloads."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_utils import trace_output_path

from examples.gemma_workloads import (
    run_gemma_attention_example,
    run_gemma_decoder_example,
    run_gemma_mlp_example,
)


def test_gemma_attention_example_matches_reference_and_emits_stage_bundles(tmp_path) -> None:
    result = run_gemma_attention_example(
        trace_path=trace_output_path("gemma_attention_example.json"),
        bundle_root=tmp_path / "gemma_attention_bundles",
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (64, 32)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 343
    assert result.perf.cycles == 3_871
    assert result.perf.instructions_by_opcode["transpose.xlu"] == 2
    assert result.perf.instructions_by_opcode["matmul.mxu0"] == 11
    assert result.perf.instructions_by_opcode["reduce.max.xlu"] == 1
    assert result.perf.instructions_by_opcode["reduce.sum.xlu"] == 1
    assert result.perf.instructions_by_opcode["vexp"] == 1
    assert result.perf.instructions_by_opcode["vrecip"] == 1
    assert sorted(result.stage_bundles) == [
        "context",
        "k_proj",
        "k_transpose",
        "o_proj",
        "q_proj",
        "scores",
        "softmax",
        "v_proj",
    ]
    assert all((path / "manifest.json5").exists() for path in result.stage_bundles.values())


def test_gemma_mlp_example_matches_reference_and_emits_stage_bundles(tmp_path) -> None:
    result = run_gemma_mlp_example(
        trace_path=trace_output_path("gemma_mlp_example.json"),
        bundle_root=tmp_path / "gemma_mlp_bundles",
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (64, 32)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 216
    assert result.perf.cycles == 3_123
    assert result.perf.instructions_by_opcode["vmul"] == 18
    assert result.perf.instructions_by_opcode["vadd"] == 6
    assert result.perf.instructions_by_opcode["vsub"] == 4
    assert result.perf.instructions_by_opcode["vexp"] == 2
    assert result.perf.instructions_by_opcode["vrecip"] == 2
    assert sorted(result.stage_bundles) == [
        "down_proj",
        "gate_mul",
        "gate_proj",
        "up_proj",
    ]
    assert all((path / "manifest.json5").exists() for path in result.stage_bundles.values())


def test_gemma_decoder_example_matches_reference_and_emits_stage_bundles(tmp_path) -> None:
    result = run_gemma_decoder_example(
        trace_path=trace_output_path("gemma_decoder_example.json"),
        bundle_root=tmp_path / "gemma_decoder_bundles",
    )

    assert result.trace_path.exists()
    assert tuple(result.output.shape) == (64, 32)
    assert torch.equal(result.output, result.golden)
    assert result.perf.instructions == 649
    assert result.perf.cycles == 8_608
    assert result.perf.instructions_by_opcode["transpose.xlu"] == 2
    assert result.perf.instructions_by_opcode["vadd"] == 10
    assert result.perf.instructions_by_opcode["reduce.max.xlu"] == 1
    assert result.perf.instructions_by_opcode["reduce.sum.xlu"] == 1
    assert result.perf.instructions_by_opcode["vexp"] == 3
    assert result.perf.instructions_by_opcode["vrecip"] == 3
    assert result.perf.instructions_by_opcode["vmul"] == 19
    assert sorted(result.stage_bundles) == [
        "attention_context",
        "attention_k_proj",
        "attention_k_transpose",
        "attention_o_proj",
        "attention_q_proj",
        "attention_scores",
        "attention_softmax",
        "attention_v_proj",
        "mlp_down_proj",
        "mlp_gate_mul",
        "mlp_gate_proj",
        "mlp_up_proj",
        "residual_after_attention",
        "residual_after_mlp",
    ]
    assert all((path / "manifest.json5").exists() for path in result.stage_bundles.values())
