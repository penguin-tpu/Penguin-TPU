"""Direct compiler export tests for the fixed PyTorch-to-Penguin flow."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from penguin_compiler import (
    deterministic_hidden,
    execute_compiled_model_package,
    export_pytorch_model_package,
    make_fixed_gemma_attention,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_utils import trace_output_path


def test_exported_attention_package_executes_and_matches_expected_output(tmp_path) -> None:
    package = export_pytorch_model_package(
        make_fixed_gemma_attention(),
        deterministic_hidden(),
        tmp_path / "gemma_attention_package",
    )

    assert package.manifest.model_kind == "gemma_attention"
    assert tuple(package.manifest.input_artifact.shape) == (64, 32)
    assert sorted(package.stage_bundles) == [
        "context",
        "k_proj",
        "k_transpose",
        "o_proj",
        "q_proj",
        "scores",
        "softmax",
        "v_proj",
    ]

    result = execute_compiled_model_package(
        package,
        trace_path=trace_output_path("compiler_export_attention.json"),
    )

    assert result.trace_path.exists()
    assert torch.equal(result.output, result.expected_output)
