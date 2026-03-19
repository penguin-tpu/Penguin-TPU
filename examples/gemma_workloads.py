"""Fixed-shape Gemma-inspired workload examples built through `penguin-compiler`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from penguin_compiler import (
    CompiledModelPackage,
    deterministic_hidden,
    execute_compiled_model_package,
    export_pytorch_model_package,
    make_fixed_gemma_attention,
    make_fixed_gemma_decoder,
    make_fixed_gemma_mlp,
)
from penguin_model.core_config import DEFAULT_PENGUIN_CORE_CONFIG, PenguinCoreConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_ROOT = REPO_ROOT / "outputs" / "examples"
DEFAULT_BUNDLE_ROOT = DEFAULT_TRACE_ROOT / "bundles"


@dataclass(frozen=True, slots=True)
class GemmaExampleRunResult:
    name: str
    trace_path: Path
    bundle_root: Path
    perf: object
    output: torch.Tensor
    golden: torch.Tensor
    stage_bundles: Mapping[str, Path]


def run_gemma_attention_example(
    trace_path: str | Path | None = None,
    *,
    bundle_root: str | Path | None = None,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> GemmaExampleRunResult:
    return _run_example(
        name="gemma_attention",
        model=make_fixed_gemma_attention(),
        trace_path=trace_path,
        bundle_root=bundle_root,
        config=config,
    )


def run_gemma_mlp_example(
    trace_path: str | Path | None = None,
    *,
    bundle_root: str | Path | None = None,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> GemmaExampleRunResult:
    return _run_example(
        name="gemma_mlp",
        model=make_fixed_gemma_mlp(),
        trace_path=trace_path,
        bundle_root=bundle_root,
        config=config,
    )


def run_gemma_decoder_example(
    trace_path: str | Path | None = None,
    *,
    bundle_root: str | Path | None = None,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> GemmaExampleRunResult:
    return _run_example(
        name="gemma_decoder",
        model=make_fixed_gemma_decoder(),
        trace_path=trace_path,
        bundle_root=bundle_root,
        config=config,
    )


def _run_example(
    *,
    name: str,
    model,
    trace_path: str | Path | None,
    bundle_root: str | Path | None,
    config: PenguinCoreConfig,
) -> GemmaExampleRunResult:
    hidden = deterministic_hidden()
    resolved_trace_path = _resolve_trace_path(trace_path, f"{name}_trace.json")
    resolved_bundle_root = _resolve_bundle_root(bundle_root, name)
    package = export_pytorch_model_package(model, hidden, resolved_bundle_root, config=config)
    result = execute_compiled_model_package(package, trace_path=resolved_trace_path, config=config)
    _require_exact_match(name, result.output, result.expected_output)
    return GemmaExampleRunResult(
        name=name,
        trace_path=result.trace_path,
        bundle_root=package.root,
        perf=result.perf,
        output=result.output,
        golden=result.expected_output,
        stage_bundles=result.stage_bundles,
    )


def _resolve_trace_path(trace_path: str | Path | None, default_name: str) -> Path:
    resolved = DEFAULT_TRACE_ROOT / default_name if trace_path is None else Path(trace_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_bundle_root(bundle_root: str | Path | None, default_name: str) -> Path:
    resolved = DEFAULT_BUNDLE_ROOT / default_name if bundle_root is None else Path(bundle_root)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _require_exact_match(name: str, output: torch.Tensor, golden: torch.Tensor) -> None:
    if torch.equal(output, golden):
        return
    diff = (output.to(torch.float32) - golden.to(torch.float32)).abs()
    raise RuntimeError(f"{name} output mismatch, max_abs_error={float(diff.max().item())}")


__all__ = [
    "GemmaExampleRunResult",
    "run_gemma_attention_example",
    "run_gemma_decoder_example",
    "run_gemma_mlp_example",
]
