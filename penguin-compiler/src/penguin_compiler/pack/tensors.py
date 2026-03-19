"""Compiler-side packing and reference helpers for fixed Gemma-style workloads."""

from __future__ import annotations

import math

import torch

from penguin_model.tensor import (
    BF16_DTYPE,
    FP8_DTYPE,
    bf16_tile_from_bytes,
    bf16_tile_to_bytes,
    bf16_transposed_tile_from_bytes,
    fp8_tile_to_bytes,
    weight_tile_to_bytes,
)

ROWS = 64
HIDDEN = 32
TILE_COLS = 32
ATTENTION_KEYS = TILE_COLS
MREG_BYTES = 4096
WEIGHT_TILE_ROWS = 64
WEIGHT_TILE_COLS_FP8 = 64
FP8_TILE_COLS = 64


def quantize_fp8(values: torch.Tensor) -> torch.Tensor:
    """Round a tensor through the hardware-visible FP8 format."""

    return values.to(FP8_DTYPE).to(torch.float32)


def bf16_round(values: torch.Tensor) -> torch.Tensor:
    """Round a tensor through BF16 and return float32 for further math."""

    return values.to(BF16_DTYPE).to(torch.float32)


def bf16_binary_op(lhs: torch.Tensor, rhs: torch.Tensor, op) -> torch.Tensor:
    """Apply one BF16-rounded binary operator."""

    return bf16_round(op(lhs.to(torch.float32), rhs.to(torch.float32)))


def _bf16_unary_op(src: torch.Tensor, op) -> torch.Tensor:
    return bf16_round(op(src.to(torch.float32)))


def bf16_reference_matmul(activation: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Reference MXU-visible matmul with FP8 inputs and BF16 accumulation."""

    activation_fp8 = quantize_fp8(activation)
    weight_fp8 = quantize_fp8(weights)
    return (activation_fp8 @ weight_fp8).to(BF16_DTYPE)


def _bf16_row_reduce_max(src: torch.Tensor) -> torch.Tensor:
    reduced = torch.amax(src.to(torch.float32), dim=1, keepdim=True).expand_as(src)
    return bf16_round(reduced)


def _bf16_row_reduce_sum(src: torch.Tensor) -> torch.Tensor:
    reduced = torch.sum(src.to(torch.float32), dim=1, keepdim=True).expand_as(src)
    return bf16_round(reduced)


def _scalar_tile(value: float) -> torch.Tensor:
    return torch.full((ROWS, TILE_COLS), value, dtype=torch.float32)


def bf16_gelu_decomposition(src: torch.Tensor) -> torch.Tensor:
    """Current VPU-visible tanh GELU decomposition used by the example program."""

    if src.shape[1] % TILE_COLS != 0:
        raise ValueError(f"GELU source width must be a multiple of {TILE_COLS}, got {tuple(src.shape)}")
    return torch.cat(
        [
            _bf16_gelu_tile(src[:, start : start + TILE_COLS])
            for start in range(0, src.shape[1], TILE_COLS)
        ],
        dim=1,
    )


def _bf16_gelu_tile(src: torch.Tensor) -> torch.Tensor:
    zero = _scalar_tile(0.0)
    one = _scalar_tile(1.0)
    two = _scalar_tile(2.0)
    half = _scalar_tile(0.5)
    alpha = _scalar_tile(math.sqrt(2.0 / math.pi))
    beta = _scalar_tile(0.044715)

    squared = bf16_binary_op(src, src, torch.mul)
    cubed = bf16_binary_op(squared, src, torch.mul)
    beta_cubed = bf16_binary_op(cubed, beta, torch.mul)
    inner = bf16_binary_op(src, beta_cubed, torch.add)
    z = bf16_binary_op(inner, alpha, torch.mul)
    two_z = bf16_binary_op(z, two, torch.mul)
    neg_two_z = bf16_binary_op(zero, two_z, torch.sub)
    exp_neg = _bf16_unary_op(neg_two_z, torch.exp)
    denom = bf16_binary_op(one, exp_neg, torch.add)
    sigma = _bf16_unary_op(denom, torch.reciprocal)
    twice_sigma = bf16_binary_op(sigma, two, torch.mul)
    tanh_z = bf16_binary_op(twice_sigma, one, torch.sub)
    one_plus_tanh = bf16_binary_op(tanh_z, one, torch.add)
    half_x = bf16_binary_op(src, half, torch.mul)
    return bf16_binary_op(half_x, one_plus_tanh, torch.mul)


def bf16_softmax_decomposition(src: torch.Tensor) -> torch.Tensor:
    """Current XLU/VPU-visible softmax decomposition used by the example program."""

    row_max = _bf16_row_reduce_max(src)
    centered = bf16_binary_op(src, row_max, torch.sub)
    exponentiated = _bf16_unary_op(centered, torch.exp)
    row_sum = _bf16_row_reduce_sum(exponentiated)
    reciprocal = _bf16_unary_op(row_sum, torch.reciprocal)
    return bf16_binary_op(exponentiated, reciprocal, torch.mul)


def scaled_attention_scores(scores: torch.Tensor) -> torch.Tensor:
    return bf16_binary_op(scores, _scalar_tile(HIDDEN**-0.5), torch.mul)


def pad_attention_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    if tuple(probabilities.shape) != (ROWS, ATTENTION_KEYS):
        raise ValueError(
            f"Attention probabilities must have shape ({ROWS}, {ATTENTION_KEYS}), "
            f"got {tuple(probabilities.shape)}"
        )
    return probabilities.to(torch.float32)


def pad_attention_values(values: torch.Tensor) -> torch.Tensor:
    if tuple(values.shape) != (ROWS, TILE_COLS):
        raise ValueError(
            f"Attention values must have shape ({ROWS}, {TILE_COLS}), got {tuple(values.shape)}"
        )
    padded = torch.zeros((WEIGHT_TILE_ROWS, TILE_COLS), dtype=torch.float32)
    padded[:ATTENTION_KEYS, :] = values[:ATTENTION_KEYS, :].to(torch.float32)
    return padded


def pack_activation_matrix(matrix: torch.Tensor) -> bytes:
    """Pack one logical activation matrix into one or more FP8 activation tiles."""

    if matrix.shape[0] != ROWS or matrix.shape[1] % TILE_COLS != 0:
        raise ValueError(f"Activation matrix must have shape (64, N*{TILE_COLS}), got {tuple(matrix.shape)}")
    packed = torch.cat(
        [
            fp8_tile_to_bytes(
                torch.cat(
                    [
                        matrix[:, start : start + TILE_COLS].to(torch.float32),
                        torch.zeros((ROWS, FP8_TILE_COLS - TILE_COLS), dtype=torch.float32),
                    ],
                    dim=1,
                )
            )
            for start in range(0, matrix.shape[1], TILE_COLS)
        ]
    )
    return bytes(packed.tolist())


def pack_bf16_matrix(matrix: torch.Tensor) -> bytes:
    """Pack one logical BF16 matrix into whole-register BF16 tiles."""

    if matrix.shape[0] != ROWS or matrix.shape[1] % TILE_COLS != 0:
        raise ValueError(f"BF16 matrix must have shape (64, N*{TILE_COLS}), got {tuple(matrix.shape)}")
    packed = torch.cat(
        [
            bf16_tile_to_bytes(matrix[:, start : start + TILE_COLS])
            for start in range(0, matrix.shape[1], TILE_COLS)
        ]
    )
    return bytes(packed.tolist())


def pack_weight_matrix(matrix: torch.Tensor) -> bytes:
    """Pack one logical weight matrix into one FP8 MXU weight tile."""

    if matrix.shape[0] > WEIGHT_TILE_ROWS or matrix.shape[1] != TILE_COLS:
        raise ValueError(
            f"Weight matrix must have shape (<= {WEIGHT_TILE_ROWS}, {TILE_COLS}), got {tuple(matrix.shape)}"
        )
    padded_rows = matrix.to(torch.float32)
    if padded_rows.shape[0] < WEIGHT_TILE_ROWS:
        padded_rows = torch.cat(
            [
                padded_rows,
                torch.zeros(
                    (WEIGHT_TILE_ROWS - padded_rows.shape[0], padded_rows.shape[1]),
                    dtype=torch.float32,
                ),
            ],
            dim=0,
        )
    padded = torch.cat(
        [
            padded_rows,
            torch.zeros((WEIGHT_TILE_ROWS, WEIGHT_TILE_COLS_FP8 - TILE_COLS), dtype=torch.float32),
        ],
        dim=1,
    )
    return bytes(weight_tile_to_bytes(padded).tolist())


def pack_gelu_constants() -> bytes:
    """Pack the BF16 scalar tiles consumed by the GEMMA MLP gate program."""

    packed = torch.cat(
        [
            bf16_tile_to_bytes(_scalar_tile(0.0)),
            bf16_tile_to_bytes(_scalar_tile(1.0)),
            bf16_tile_to_bytes(_scalar_tile(2.0)),
            bf16_tile_to_bytes(_scalar_tile(0.5)),
            bf16_tile_to_bytes(_scalar_tile(math.sqrt(2.0 / math.pi))),
            bf16_tile_to_bytes(_scalar_tile(0.044715)),
        ]
    )
    return bytes(packed.tolist())


def read_bf16_matrix_symbol(state, symbol, *, cols: int = TILE_COLS) -> torch.Tensor:
    """Read one logical BF16 matrix back from a VMEM symbol."""

    tiles = cols // TILE_COLS
    chunks = [
        bf16_tile_from_bytes(state.vmem.read(symbol.address + index * MREG_BYTES, MREG_BYTES).clone()).clone()
        for index in range(tiles)
    ]
    return torch.cat(chunks, dim=1)


def read_transposed_bf16_matrix_symbol(state, symbol, *, rows: int = TILE_COLS) -> torch.Tensor:
    """Read one logical transposed BF16 matrix back from a VMEM symbol."""

    tiles = rows // TILE_COLS
    chunks = [
        bf16_transposed_tile_from_bytes(
            state.vmem.read(symbol.address + index * MREG_BYTES, MREG_BYTES).clone()
        ).clone()
        for index in range(tiles)
    ]
    return torch.cat(chunks, dim=0)
