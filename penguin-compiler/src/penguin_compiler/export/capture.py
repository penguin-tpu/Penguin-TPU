"""Capture and validate the fixed PyTorch model flow supported by Penguin."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .fixed_gemma import FixedGemmaAttention, FixedGemmaDecoder, FixedGemmaMLP


@dataclass(frozen=True, slots=True)
class CapturedFixedModel:
    """Captured fixed-model export plus compiler-relevant metadata."""

    model_kind: str
    exported_program: torch.export.ExportedProgram
    example_input: torch.Tensor
    state_dict: dict[str, torch.Tensor]
    golden_output: torch.Tensor


_REQUIRED_OPS = {
    "gemma_attention": {
        "aten.matmul.default",
        "aten.transpose.int",
        "aten.slice.Tensor",
        "aten.mul.Tensor",
        "aten.sub.Tensor",
        "aten.exp.default",
        "aten.sum.dim_IntList",
        "aten.amax.default",
        "aten.reciprocal.default",
    },
    "gemma_mlp": {
        "aten.matmul.default",
        "aten.mul.Tensor",
        "aten.add.Tensor",
        "aten.sub.Tensor",
        "aten.exp.default",
        "aten.reciprocal.default",
    },
    "gemma_decoder": {
        "aten.matmul.default",
        "aten.transpose.int",
        "aten.slice.Tensor",
        "aten.mul.Tensor",
        "aten.add.Tensor",
        "aten.sub.Tensor",
        "aten.exp.default",
        "aten.sum.dim_IntList",
        "aten.amax.default",
        "aten.reciprocal.default",
    },
}


def capture_fixed_model(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    *,
    strict: bool = False,
) -> CapturedFixedModel:
    """Capture one supported fixed PyTorch model through `torch.export`."""

    model_kind = _model_kind(model)
    _validate_example_input(example_input)

    exported = torch.export.export(model.eval(), (example_input,), strict=strict)
    _validate_export(model_kind, exported)

    with torch.no_grad():
        golden = model(example_input).clone()

    return CapturedFixedModel(
        model_kind=model_kind,
        exported_program=exported,
        example_input=example_input.clone(),
        state_dict={name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
        golden_output=golden,
    )


def _model_kind(model: torch.nn.Module) -> str:
    if isinstance(model, FixedGemmaAttention):
        return "gemma_attention"
    if isinstance(model, FixedGemmaMLP):
        return "gemma_mlp"
    if isinstance(model, FixedGemmaDecoder):
        return "gemma_decoder"
    raise ValueError(f"Unsupported model type '{type(model).__name__}' for the fixed Penguin exporter")


def _validate_example_input(example_input: torch.Tensor) -> None:
    if tuple(example_input.shape) != (64, 32):
        raise ValueError(f"Fixed Penguin exporter expects one input of shape (64, 32), got {tuple(example_input.shape)}")
    if example_input.dtype != torch.float32:
        raise ValueError(f"Fixed Penguin exporter expects float32 example input, got {example_input.dtype}")


def _validate_export(model_kind: str, exported: torch.export.ExportedProgram) -> None:
    if tuple(exported.graph_signature.user_inputs) != ("hidden",) and tuple(exported.graph_signature.user_inputs) != ("x",):
        raise ValueError(f"Expected one user input for '{model_kind}', got {exported.graph_signature.user_inputs}")

    targets = []
    for spec in exported.graph_signature.input_specs:
        if spec.kind.name == "PARAMETER":
            targets.append(spec.target)
    expected_targets = _expected_parameter_targets(model_kind)
    if tuple(targets) != expected_targets:
        raise ValueError(
            f"Unexpected parameter targets for '{model_kind}': expected {expected_targets}, got {tuple(targets)}"
        )

    ops = {str(node.target) for node in exported.graph.nodes if node.op == "call_function"}
    missing = sorted(_REQUIRED_OPS[model_kind] - ops)
    if missing:
        raise ValueError(f"Exported graph for '{model_kind}' is missing required operations: {missing}")


def _expected_parameter_targets(model_kind: str) -> tuple[str, ...]:
    if model_kind == "gemma_attention":
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    if model_kind == "gemma_mlp":
        return ("gate_proj", "up_proj", "down_proj")
    if model_kind == "gemma_decoder":
        return ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    raise AssertionError(f"Unhandled model kind '{model_kind}'")
