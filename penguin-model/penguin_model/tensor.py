"""Tensor-register, scale-register, and MXU helpers for the Penguin model."""

from __future__ import annotations

import torch

from .core_config import DEFAULT_PENGUIN_CORE_CONFIG, PenguinCoreConfig
from .memory import _mix_seed, _random_u8_tensor

NUM_MREG = DEFAULT_PENGUIN_CORE_CONFIG.tensor.num_mreg
MREG_ROWS = DEFAULT_PENGUIN_CORE_CONFIG.tensor.mreg_rows
MREG_ROW_BYTES = DEFAULT_PENGUIN_CORE_CONFIG.tensor.mreg_row_bytes
MREG_FP8_COLS = DEFAULT_PENGUIN_CORE_CONFIG.mreg_fp8_cols
MREG_BF16_COLS = DEFAULT_PENGUIN_CORE_CONFIG.mreg_bf16_cols
MREG_BYTES = DEFAULT_PENGUIN_CORE_CONFIG.mreg_bytes

MXU_COUNT = DEFAULT_PENGUIN_CORE_CONFIG.tensor.mxu_count
WEIGHT_SLOTS_PER_MXU = DEFAULT_PENGUIN_CORE_CONFIG.tensor.weight_slots_per_mxu
WEIGHT_TILE_ROWS = DEFAULT_PENGUIN_CORE_CONFIG.tensor.weight_tile_rows
WEIGHT_TILE_COLS_FP8 = DEFAULT_PENGUIN_CORE_CONFIG.tensor.weight_tile_cols_fp8
WEIGHT_SLOT_BYTES = DEFAULT_PENGUIN_CORE_CONFIG.weight_slot_bytes
ACCUM_BUFFER_BYTES = DEFAULT_PENGUIN_CORE_CONFIG.accum_buffer_bytes

MATMUL_RESULT_ROWS = DEFAULT_PENGUIN_CORE_CONFIG.matmul_result_rows
MATMUL_RESULT_COLS = DEFAULT_PENGUIN_CORE_CONFIG.matmul_result_cols
MATMUL_RESULT_REGISTERS = DEFAULT_PENGUIN_CORE_CONFIG.matmul_result_registers
VMEM_TENSOR_ALIGN = DEFAULT_PENGUIN_CORE_CONFIG.tensor.vmem_alignment_bytes

VLOAD_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.vload_latency_cycles
VSTORE_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.vstore_latency_cycles
VMATPUSH_WEIGHT_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.vmatpush_weight_latency_cycles
VMATPUSH_ACC_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.vmatpush_acc_latency_cycles
VMATPOP_ACC_BF16_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.vmatpop_acc_bf16_latency_cycles
VMATPOP_ACC_FP8_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.vmatpop_acc_fp8_latency_cycles
MATMUL_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.matmul_latency_cycles
VPU_SIMPLE_OP_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.vpu_simple_op_latency_cycles
VPU_NON_PIPELINEABLE_OP_LATENCY_CYCLES = (
    DEFAULT_PENGUIN_CORE_CONFIG.vpu_non_pipelineable_op_latency_cycles
)
XLU_TRANSPOSE_LATENCY_CYCLES = DEFAULT_PENGUIN_CORE_CONFIG.xlu_transpose_latency_cycles

FP8_DTYPE = torch.float8_e4m3fn
BF16_DTYPE = torch.bfloat16


def make_tensor_register_file() -> torch.Tensor:
    """Allocate the architectural tensor register file as raw bytes."""

    return make_tensor_register_file_for_config(DEFAULT_PENGUIN_CORE_CONFIG)


def make_tensor_register_file_for_config(config: PenguinCoreConfig) -> torch.Tensor:
    """Allocate the tensor register file for a specific core configuration."""

    if not config.initialization.randomize_tensor_registers:
        return torch.zeros((config.tensor.num_mreg, config.mreg_bytes), dtype=torch.uint8)
    return _random_u8_tensor(
        config.tensor.num_mreg * config.mreg_bytes,
        seed=_mix_seed(config.initialization.seed, 0x4D52_4547),
    ).reshape(config.tensor.num_mreg, config.mreg_bytes)


def make_weight_slot_file() -> torch.Tensor:
    """Allocate the architected MXU weight slots as raw bytes."""

    return make_weight_slot_file_for_config(DEFAULT_PENGUIN_CORE_CONFIG)


def make_weight_slot_file_for_config(config: PenguinCoreConfig) -> torch.Tensor:
    """Allocate MXU weight slots for a specific core configuration."""

    shape = (
        config.tensor.mxu_count,
        config.tensor.weight_slots_per_mxu,
        config.weight_slot_bytes,
    )
    if not config.initialization.randomize_weight_slots:
        return torch.zeros(shape, dtype=torch.uint8)
    return _random_u8_tensor(
        config.tensor.mxu_count
        * config.tensor.weight_slots_per_mxu
        * config.weight_slot_bytes,
        seed=_mix_seed(config.initialization.seed, 0x5745_4947),
    ).reshape(shape)


def make_accum_buffer_file_for_config(config: PenguinCoreConfig) -> torch.Tensor:
    """Allocate MXU accumulation buffers for a specific core configuration."""

    shape = (
        config.tensor.mxu_count,
        config.tensor.accum_slots_per_mxu,
        config.accum_buffer_bytes,
    )
    if not config.initialization.randomize_accum_buffers:
        return torch.zeros(shape, dtype=torch.uint8)
    return _random_u8_tensor(
        config.tensor.mxu_count * config.tensor.accum_slots_per_mxu * config.accum_buffer_bytes,
        seed=_mix_seed(config.initialization.seed, 0x4143_4355),
    ).reshape(shape)


def decode_scale_fp8_e8m0(raw: int) -> float:
    """Decode one scale-register payload as an unbiased signed power-of-two scale."""

    value = int(raw) & 0xFF
    if value >= 0x80:
        value -= 0x100
    return float(2.0**value)


def fp8_tile_from_bytes(
    raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Interpret one tensor-register image as a FP8 activation tile."""

    return (
        raw.reshape(-1)
        .view(FP8_DTYPE)
        .reshape(config.tensor.mreg_rows, config.mreg_fp8_cols)
        .to(torch.float32)
    )


def bf16_tile_from_bytes(
    raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Interpret one tensor-register image as one BF16 whole-register tile."""

    return (
        raw.reshape(-1)
        .view(BF16_DTYPE)
        .reshape(config.tensor.mreg_rows, config.mreg_bf16_cols)
    )


def bf16_tile_pair_from_bytes(
    raw_lo: torch.Tensor,
    raw_hi: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Interpret two tensor-register images as one full BF16 MXU result tile."""

    return torch.cat(
        (
            bf16_tile_from_bytes(raw_lo, config=config),
            bf16_tile_from_bytes(raw_hi, config=config),
        ),
        dim=1,
    )


def weight_tile_from_bytes(
    raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Interpret one MXU weight-slot image as one FP8 weight tile."""

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
    """Pack one FP8 weight tile into raw bytes."""

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
    """Pack one FP8 activation tile into raw bytes."""

    packed = (
        tile.to(torch.float32)
        .reshape(config.tensor.mreg_rows, config.mreg_fp8_cols)
        .to(FP8_DTYPE)
    )
    return packed.reshape(-1).view(torch.uint8).clone()


def bf16_tile_to_bytes(
    tile: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Pack one BF16 whole-register tile into raw bytes."""

    packed = tile.to(BF16_DTYPE).reshape(config.tensor.mreg_rows, config.mreg_bf16_cols)
    return packed.reshape(-1).view(torch.uint8).clone()


def bf16_tile_pair_to_bytes(
    tile: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one full BF16 MXU result tile into its two architectural destination registers."""

    packed = tile.to(BF16_DTYPE).reshape(config.matmul_result_rows, config.matmul_result_cols)
    return (
        bf16_tile_to_bytes(packed[:, : config.mreg_bf16_cols], config=config),
        bf16_tile_to_bytes(packed[:, config.mreg_bf16_cols :], config=config),
    )


def accum_tile_from_bytes(
    raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Interpret one MXU accumulation buffer image as a BF16 `64 x 64` tile."""

    return (
        raw.reshape(-1)
        .view(BF16_DTYPE)
        .reshape(config.matmul_result_rows, config.matmul_result_cols)
    )


def accum_tile_to_bytes(
    tile: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Pack one BF16 `64 x 64` tile into one MXU accumulation buffer image."""

    packed = tile.to(BF16_DTYPE).reshape(config.matmul_result_rows, config.matmul_result_cols)
    return packed.reshape(-1).view(torch.uint8).clone()


def bf16_transposed_tile_from_bytes(
    raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Interpret one tensor-register image as a BF16 transposed tile."""

    return (
        raw.reshape(-1)
        .view(BF16_DTYPE)
        .reshape(config.mreg_bf16_cols, config.tensor.mreg_rows)
    )


def bf16_transposed_tile_to_bytes(
    tile: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Pack one BF16 transposed tile into raw bytes."""

    packed = tile.to(BF16_DTYPE).reshape(config.mreg_bf16_cols, config.tensor.mreg_rows)
    return packed.reshape(-1).view(torch.uint8).clone()


def compute_bf16_matmul(
    activation_raw: torch.Tensor,
    weight_raw: torch.Tensor,
    scale_a_raw: int,
    scale_b_raw: int,
    partial_raw: tuple[torch.Tensor, torch.Tensor] | None = None,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one scaled 64x64 @ 64x64 MXU tile with BF16 accumulation."""

    activation = fp8_tile_from_bytes(activation_raw, config=config)
    weight = weight_tile_from_bytes(weight_raw, config=config)
    product = activation @ weight
    product *= decode_scale_fp8_e8m0(scale_a_raw) * decode_scale_fp8_e8m0(scale_b_raw)

    if partial_raw is None:
        acc = torch.zeros(
            (config.matmul_result_rows, config.matmul_result_cols),
            dtype=BF16_DTYPE,
        )
    else:
        acc = bf16_tile_pair_from_bytes(partial_raw[0], partial_raw[1], config=config).clone()

    result = (acc.to(torch.float32) + product.to(torch.float32)).to(BF16_DTYPE)
    return bf16_tile_pair_to_bytes(result, config=config)


def compute_accum_matmul(
    activation_raw: torch.Tensor,
    weight_raw: torch.Tensor,
    accum_raw: torch.Tensor | None = None,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Compute one `FP8 x FP8 -> BF16` MXU tile into accumulation-buffer layout."""

    activation = fp8_tile_from_bytes(activation_raw, config=config)
    weight = weight_tile_from_bytes(weight_raw, config=config)
    product = activation @ weight
    if accum_raw is None:
        acc = torch.zeros(
            (config.matmul_result_rows, config.matmul_result_cols),
            dtype=BF16_DTYPE,
        )
    else:
        acc = accum_tile_from_bytes(accum_raw, config=config).clone()
    result = (acc.to(torch.float32) + product.to(torch.float32)).to(BF16_DTYPE)
    return accum_tile_to_bytes(result, config=config)


def export_accum_to_fp8(
    accum_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Quantize one BF16 accumulation buffer tile into one FP8 tensor register image."""

    accum = accum_tile_from_bytes(accum_raw, config=config).to(torch.float32)
    return fp8_tile_to_bytes(accum, config=config)


def _binary_bf16_tile_op(
    lhs_raw: torch.Tensor,
    rhs_raw: torch.Tensor,
    op,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    lhs = bf16_tile_from_bytes(lhs_raw, config=config).to(torch.float32)
    rhs = bf16_tile_from_bytes(rhs_raw, config=config).to(torch.float32)
    return bf16_tile_to_bytes(op(lhs, rhs).to(BF16_DTYPE), config=config)


def compute_bf16_vadd(
    lhs_raw: torch.Tensor,
    rhs_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Compute one BF16 whole-register elementwise addition."""

    return _binary_bf16_tile_op(lhs_raw, rhs_raw, torch.add, config=config)


def compute_bf16_vmul(
    lhs_raw: torch.Tensor,
    rhs_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Compute one BF16 whole-register elementwise multiply."""

    return _binary_bf16_tile_op(lhs_raw, rhs_raw, torch.mul, config=config)


def compute_bf16_vsub(
    lhs_raw: torch.Tensor,
    rhs_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Compute one BF16 whole-register elementwise subtraction."""

    return _binary_bf16_tile_op(lhs_raw, rhs_raw, torch.sub, config=config)


def compute_bf16_vmax(
    lhs_raw: torch.Tensor,
    rhs_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Compute one BF16 whole-register elementwise maximum."""

    return _binary_bf16_tile_op(lhs_raw, rhs_raw, torch.maximum, config=config)


def compute_bf16_vmin(
    lhs_raw: torch.Tensor,
    rhs_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Compute one BF16 whole-register elementwise minimum."""

    return _binary_bf16_tile_op(lhs_raw, rhs_raw, torch.minimum, config=config)


def compute_bf16_vrelu(
    src_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Compute one BF16 whole-register elementwise ReLU."""

    src = bf16_tile_from_bytes(src_raw, config=config).to(torch.float32)
    return bf16_tile_to_bytes(torch.clamp_min(src, 0.0).to(BF16_DTYPE), config=config)


def compute_bf16_vexp(
    src_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Compute one BF16 whole-register elementwise exponential."""

    src = bf16_tile_from_bytes(src_raw, config=config).to(torch.float32)
    return bf16_tile_to_bytes(torch.exp(src).to(BF16_DTYPE), config=config)


def compute_bf16_vrecip(
    src_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Compute one BF16 whole-register elementwise reciprocal."""

    src = bf16_tile_from_bytes(src_raw, config=config).to(torch.float32)
    return bf16_tile_to_bytes(torch.reciprocal(src).to(BF16_DTYPE), config=config)


def compute_bf16_vmov(
    src_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Copy one BF16 whole-register tile without modification."""

    return bf16_tile_to_bytes(
        bf16_tile_from_bytes(src_raw, config=config).clone(),
        config=config,
    )


def compute_bf16_row_reduce_max(
    src_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Reduce each BF16 row to one maximum and write the result into column zero."""

    src = bf16_tile_from_bytes(src_raw, config=config).to(torch.float32)
    result = torch.zeros_like(src)
    result[:, 0] = torch.amax(src, dim=1)
    return bf16_tile_to_bytes(result.to(BF16_DTYPE), config=config)


def compute_bf16_row_reduce_sum(
    src_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Reduce each BF16 row to one sum and write the result into column zero."""

    src = bf16_tile_from_bytes(src_raw, config=config).to(torch.float32)
    result = torch.zeros_like(src)
    result[:, 0] = torch.sum(src, dim=1)
    return bf16_tile_to_bytes(result.to(BF16_DTYPE), config=config)


def compute_bf16_vredsum(
    src_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Reduce BF16 columns into one row and write the result into row zero."""

    src = bf16_tile_from_bytes(src_raw, config=config).to(torch.float32)
    result = torch.zeros_like(src)
    result[0, :] = torch.sum(src, dim=0)
    return bf16_tile_to_bytes(result.to(BF16_DTYPE), config=config)


def compute_vector_immediate_fill(
    imm16: int,
    *,
    mode: str,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Materialize one BF16 whole-register immediate pattern."""

    value = torch.tensor([imm16 & 0xFFFF], dtype=torch.uint16).view(BF16_DTYPE)[0]
    tile = torch.zeros(
        (config.tensor.mreg_rows, config.mreg_bf16_cols),
        dtype=BF16_DTYPE,
    )
    if mode == "all":
        tile[:, :] = value
    elif mode == "row":
        tile[0, :] = value
    elif mode == "col":
        tile[:, 0] = value
    elif mode == "one":
        tile[0, 0] = value
    else:
        raise ValueError(f"unsupported vector immediate fill mode: {mode}")
    return bf16_tile_to_bytes(tile, config=config)


def compute_bf16_transpose(
    src_raw: torch.Tensor,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
) -> torch.Tensor:
    """Compute one BF16 whole-register transpose."""

    src = bf16_tile_from_bytes(src_raw, config=config)
    transposed = src.transpose(0, 1).contiguous()
    return bf16_transposed_tile_to_bytes(transposed, config=config)


__all__ = [
    "ACCUM_BUFFER_BYTES",
    "BF16_DTYPE",
    "FP8_DTYPE",
    "MATMUL_LATENCY_CYCLES",
    "MATMUL_RESULT_COLS",
    "MATMUL_RESULT_REGISTERS",
    "MATMUL_RESULT_ROWS",
    "MREG_BF16_COLS",
    "MREG_BYTES",
    "MREG_FP8_COLS",
    "MREG_ROW_BYTES",
    "MREG_ROWS",
    "MXU_COUNT",
    "NUM_MREG",
    "VMEM_TENSOR_ALIGN",
    "VLOAD_LATENCY_CYCLES",
    "VMATPOP_ACC_BF16_LATENCY_CYCLES",
    "VMATPOP_ACC_FP8_LATENCY_CYCLES",
    "VMATPUSH_ACC_LATENCY_CYCLES",
    "VMATPUSH_WEIGHT_LATENCY_CYCLES",
    "VPU_NON_PIPELINEABLE_OP_LATENCY_CYCLES",
    "VPU_SIMPLE_OP_LATENCY_CYCLES",
    "VSTORE_LATENCY_CYCLES",
    "WEIGHT_SLOT_BYTES",
    "WEIGHT_SLOTS_PER_MXU",
    "WEIGHT_TILE_COLS_FP8",
    "WEIGHT_TILE_ROWS",
    "XLU_TRANSPOSE_LATENCY_CYCLES",
    "accum_tile_from_bytes",
    "accum_tile_to_bytes",
    "bf16_tile_from_bytes",
    "bf16_tile_pair_from_bytes",
    "bf16_tile_pair_to_bytes",
    "bf16_tile_to_bytes",
    "bf16_transposed_tile_from_bytes",
    "bf16_transposed_tile_to_bytes",
    "compute_bf16_matmul",
    "compute_bf16_row_reduce_max",
    "compute_bf16_row_reduce_sum",
    "compute_bf16_transpose",
    "compute_bf16_vredsum",
    "compute_vector_immediate_fill",
    "compute_bf16_vadd",
    "compute_bf16_vexp",
    "compute_bf16_vmax",
    "compute_bf16_vmin",
    "compute_bf16_vmov",
    "compute_bf16_vmul",
    "compute_bf16_vrecip",
    "compute_bf16_vrelu",
    "compute_bf16_vsub",
    "compute_accum_matmul",
    "decode_scale_fp8_e8m0",
    "export_accum_to_fp8",
    "fp8_tile_from_bytes",
    "fp8_tile_to_bytes",
    "make_accum_buffer_file_for_config",
    "make_tensor_register_file",
    "make_tensor_register_file_for_config",
    "make_weight_slot_file",
    "make_weight_slot_file_for_config",
    "weight_tile_from_bytes",
    "weight_tile_to_bytes",
]
