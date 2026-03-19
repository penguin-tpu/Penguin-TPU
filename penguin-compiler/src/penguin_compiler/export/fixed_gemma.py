"""Fixed-shape PyTorch Gemma-style modules supported by the Penguin compiler."""

from __future__ import annotations

import torch

from ..pack import (
    ATTENTION_KEYS,
    BF16_DTYPE,
    HIDDEN,
    ROWS,
    bf16_binary_op,
    bf16_gelu_decomposition,
    bf16_reference_matmul,
    bf16_softmax_decomposition,
    scaled_attention_scores,
)


def deterministic_hidden() -> torch.Tensor:
    indices = torch.arange(ROWS * HIDDEN, dtype=torch.float32).reshape(ROWS, HIDDEN)
    return ((indices % 11) / 10) + 0.5


def _deterministic_weight(base: float, span: float, *, twist: int) -> torch.Tensor:
    indices = torch.arange(HIDDEN * HIDDEN, dtype=torch.float32).reshape(HIDDEN, HIDDEN)
    pattern = ((indices * (twist + 3) + (indices.transpose(0, 1) * (twist + 1))) % 17) / 16
    return base + span * pattern


def _attention_weights() -> dict[str, torch.Tensor]:
    return {
        "q_proj": _deterministic_weight(0.030, 0.018, twist=0),
        "k_proj": _deterministic_weight(0.024, 0.016, twist=3),
        "v_proj": _deterministic_weight(0.038, 0.022, twist=5),
        "o_proj": _deterministic_weight(0.034, 0.020, twist=7),
    }


def _mlp_weights() -> dict[str, torch.Tensor]:
    return {
        "gate_proj": _deterministic_weight(0.200, 0.120, twist=1),
        "up_proj": _deterministic_weight(0.110, 0.080, twist=4),
        "down_proj": _deterministic_weight(0.060, 0.032, twist=6),
    }


class FixedGemmaAttention(torch.nn.Module):
    """Hardware-visible fixed Gemma attention slice."""

    penguin_model_kind = "gemma_attention"

    def __init__(self, *, weights: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        resolved = _attention_weights() if weights is None else weights
        self.q_proj = torch.nn.Parameter(resolved["q_proj"].clone())
        self.k_proj = torch.nn.Parameter(resolved["k_proj"].clone())
        self.v_proj = torch.nn.Parameter(resolved["v_proj"].clone())
        self.o_proj = torch.nn.Parameter(resolved["o_proj"].clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        q = bf16_reference_matmul(hidden, self.q_proj)
        k = bf16_reference_matmul(hidden, self.k_proj)
        v = bf16_reference_matmul(hidden, self.v_proj)
        key_slice = k[:ATTENTION_KEYS, :].transpose(0, 1).contiguous().to(BF16_DTYPE)
        scores = bf16_reference_matmul(q.to(torch.float32), key_slice.to(torch.float32))
        probabilities = bf16_softmax_decomposition(scaled_attention_scores(scores))
        context = bf16_reference_matmul(
            probabilities.to(torch.float32),
            v[:ATTENTION_KEYS, :].to(torch.float32),
        )
        return bf16_reference_matmul(context.to(torch.float32), self.o_proj)


class FixedGemmaMLP(torch.nn.Module):
    """Hardware-visible fixed Gemma MLP slice."""

    penguin_model_kind = "gemma_mlp"

    def __init__(self, *, weights: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        resolved = _mlp_weights() if weights is None else weights
        self.gate_proj = torch.nn.Parameter(resolved["gate_proj"].clone())
        self.up_proj = torch.nn.Parameter(resolved["up_proj"].clone())
        self.down_proj = torch.nn.Parameter(resolved["down_proj"].clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = bf16_reference_matmul(hidden, self.gate_proj)
        up = bf16_reference_matmul(hidden, self.up_proj)
        activated = bf16_gelu_decomposition(gate)
        gated = bf16_binary_op(activated, up, torch.mul)
        return bf16_reference_matmul(gated.to(torch.float32), self.down_proj)


class FixedGemmaDecoder(torch.nn.Module):
    """Hardware-visible fixed Gemma decoder slice."""

    penguin_model_kind = "gemma_decoder"

    def __init__(self) -> None:
        super().__init__()
        attention = _attention_weights()
        mlp = _mlp_weights()
        self.q_proj = torch.nn.Parameter(attention["q_proj"].clone())
        self.k_proj = torch.nn.Parameter(attention["k_proj"].clone())
        self.v_proj = torch.nn.Parameter(attention["v_proj"].clone())
        self.o_proj = torch.nn.Parameter(attention["o_proj"].clone())
        self.gate_proj = torch.nn.Parameter(mlp["gate_proj"].clone())
        self.up_proj = torch.nn.Parameter(mlp["up_proj"].clone())
        self.down_proj = torch.nn.Parameter(mlp["down_proj"].clone())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        q = bf16_reference_matmul(hidden, self.q_proj)
        k = bf16_reference_matmul(hidden, self.k_proj)
        v = bf16_reference_matmul(hidden, self.v_proj)
        key_slice = k[:ATTENTION_KEYS, :].transpose(0, 1).contiguous().to(BF16_DTYPE)
        scores = bf16_reference_matmul(q.to(torch.float32), key_slice.to(torch.float32))
        probabilities = bf16_softmax_decomposition(scaled_attention_scores(scores))
        context = bf16_reference_matmul(
            probabilities.to(torch.float32),
            v[:ATTENTION_KEYS, :].to(torch.float32),
        )
        attention = bf16_reference_matmul(context.to(torch.float32), self.o_proj)
        hidden_bf16 = hidden.to(BF16_DTYPE)
        post_attention = (hidden_bf16.to(torch.float32) + attention.to(torch.float32)).to(BF16_DTYPE)
        gate = bf16_reference_matmul(post_attention.to(torch.float32), self.gate_proj)
        up = bf16_reference_matmul(post_attention.to(torch.float32), self.up_proj)
        activated = bf16_gelu_decomposition(gate)
        gated = bf16_binary_op(activated, up, torch.mul)
        mlp = bf16_reference_matmul(gated.to(torch.float32), self.down_proj)
        return (post_attention.to(torch.float32) + mlp.to(torch.float32)).to(BF16_DTYPE)


def make_fixed_gemma_attention() -> FixedGemmaAttention:
    return FixedGemmaAttention().eval()


def make_fixed_gemma_mlp() -> FixedGemmaMLP:
    return FixedGemmaMLP().eval()


def make_fixed_gemma_decoder() -> FixedGemmaDecoder:
    return FixedGemmaDecoder().eval()
