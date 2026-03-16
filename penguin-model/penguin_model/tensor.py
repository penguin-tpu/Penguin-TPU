"""Tensor-register and MXU weight-slot helpers for the Penguin model."""

from __future__ import annotations

import torch

from .core_config import DEFAULT_PENGUIN_CORE_CONFIG, PenguinCoreConfig

NUM_MREG = DEFAULT_PENGUIN_CORE_CONFIG.tensor.num_mreg
MREG_ROWS = DEFAULT_PENGUIN_CORE_CONFIG.tensor.mreg_rows
MREG_ROW_BYTES = DEFAULT_PENGUIN_CORE_CONFIG.tensor.mreg_row_bytes
MREG_BYTES = DEFAULT_PENGUIN_CORE_CONFIG.mreg_bytes

MXU_COUNT = DEFAULT_PENGUIN_CORE_CONFIG.tensor.mxu_count
WEIGHT_SLOTS_PER_MXU = DEFAULT_PENGUIN_CORE_CONFIG.tensor.weight_slots_per_mxu
WEIGHT_TILE_ROWS = DEFAULT_PENGUIN_CORE_CONFIG.tensor.weight_tile_rows
WEIGHT_TILE_COLS_FP8 = DEFAULT_PENGUIN_CORE_CONFIG.tensor.weight_tile_cols_fp8
WEIGHT_SLOT_BYTES = DEFAULT_PENGUIN_CORE_CONFIG.weight_slot_bytes

VMEM_TENSOR_ALIGN = DEFAULT_PENGUIN_CORE_CONFIG.tensor.vmem_alignment_bytes

VLOAD_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.vload_latency_cycles
VSTORE_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.vstore_latency_cycles
MXU_PUSH_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.mxu_push_latency_cycles
MATMUL_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.matmul_latency_cycles

FP8_DTYPE = torch.float8_e4m3fn
BF16_DTYPE = torch.bfloat16


def make_tensor_register_file() -> torch.Tensor:
    """Allocate the architectural tensor register file as raw bytes."""

    return make_tensor_register_file_for_config(DEFAULT_PENGUIN_CORE_CONFIG)


def make_tensor_register_file_for_config(config: PenguinCoreConfig) -> torch.Tensor:
    """Allocate the tensor register file for a specific core configuration."""

    return torch.zeros((config.tensor.num_mreg, config.mreg_bytes), dtype=torch.uint8)


def make_weight_slot_file() -> torch.Tensor:
    """Allocate the architected MXU weight slots as raw bytes."""

    return make_weight_slot_file_for_config(DEFAULT_PENGUIN_CORE_CONFIG)


def make_weight_slot_file_for_config(config: PenguinCoreConfig) -> torch.Tensor:
    """Allocate MXU weight slots for a specific core configuration."""

    return torch.zeros(
        (
            config.tensor.mxu_count,
            config.tensor.weight_slots_per_mxu,
            config.weight_slot_bytes,
        ),
        dtype=torch.uint8,
    )


def fp8_tile_from_bytes(
    raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Interpret one tensor-register image as a 64x32 FP8 tile."""

    return (
        raw.reshape(-1)
        .view(FP8_DTYPE)
        .reshape(config.tensor.mreg_rows, config.tensor.mreg_row_bytes)
        .to(torch.float32)
    )


def bf16_tile_from_bytes(
    raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Interpret one tensor-register image as a 64x16 BF16 tile."""

    return (
        raw.reshape(-1)
        .view(BF16_DTYPE)
        .reshape(config.tensor.mreg_rows, config.tensor.mreg_row_bytes // 2)
    )


def weight_tile_from_bytes(
    raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Interpret one MXU weight-slot image as a 32x16 FP8 tile."""

    return (
        raw.reshape(-1)
        .view(FP8_DTYPE)
        .reshape(config.tensor.weight_tile_rows, config.tensor.weight_tile_cols_fp8)
        .to(torch.float32)
    )


def weight_tile_to_bytes(
    tile: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Pack a 32x16 FP8 weight tile into raw bytes."""

    packed = (
        tile.to(torch.float32)
        .reshape(config.tensor.weight_tile_rows, config.tensor.weight_tile_cols_fp8)
        .to(FP8_DTYPE)
    )
    return packed.reshape(-1).view(torch.uint8).clone()


def fp8_tile_to_bytes(
    tile: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Pack a 64x32 FP8 tile into raw bytes."""

    packed = (
        tile.to(torch.float32)
        .reshape(config.tensor.mreg_rows, config.tensor.mreg_row_bytes)
        .to(FP8_DTYPE)
    )
    return packed.reshape(-1).view(torch.uint8).clone()


def bf16_tile_to_bytes(
    tile: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Pack a 64x16 BF16 tile into raw bytes."""

    packed = (
        tile.to(BF16_DTYPE)
        .reshape(config.tensor.mreg_rows, config.tensor.mreg_row_bytes // 2)
    )
    return packed.reshape(-1).view(torch.uint8).clone()


def compute_bf16_matmul(
    activation_raw: torch.Tensor,
    weight_raw: torch.Tensor,
    partial_raw: torch.Tensor | None = None,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Compute one 64x32 @ 32x16 MXU tile with BF16 accumulation."""

    activation = fp8_tile_from_bytes(activation_raw, config=config)
    weight = weight_tile_from_bytes(weight_raw, config=config)
    if partial_raw is None:
        acc = torch.zeros(
            (config.tensor.mreg_rows, config.tensor.weight_tile_cols_fp8),
            dtype=BF16_DTYPE,
        )
    else:
        acc = bf16_tile_from_bytes(partial_raw, config=config).clone()

    for inner in range(config.tensor.weight_tile_rows):
        product = activation[:, inner].unsqueeze(1) * weight[inner, :].unsqueeze(0)
        acc = (acc.to(torch.float32) + product).to(BF16_DTYPE)
    return bf16_tile_to_bytes(acc, config=config)


__all__ = [
    "BF16_DTYPE",
    "FP8_DTYPE",
    "MATMUL_LATENCY_CYCLES",
    "MREG_BYTES",
    "MREG_ROW_BYTES",
    "MREG_ROWS",
    "MXU_COUNT",
    "MXU_PUSH_LATENCY_CYCLES",
    "NUM_MREG",
    "VMEM_TENSOR_ALIGN",
    "VLOAD_LATENCY_CYCLES",
    "VSTORE_LATENCY_CYCLES",
    "WEIGHT_SLOT_BYTES",
    "WEIGHT_SLOTS_PER_MXU",
    "WEIGHT_TILE_COLS_FP8",
    "WEIGHT_TILE_ROWS",
    "bf16_tile_from_bytes",
    "bf16_tile_to_bytes",
    "compute_bf16_matmul",
    "fp8_tile_from_bytes",
    "fp8_tile_to_bytes",
    "make_tensor_register_file",
    "make_tensor_register_file_for_config",
    "make_weight_slot_file",
    "make_weight_slot_file_for_config",
    "weight_tile_from_bytes",
    "weight_tile_to_bytes",
]
