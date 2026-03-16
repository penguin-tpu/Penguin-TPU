"""Intermediate-state verification for large DMA-backed tensor workloads."""

from __future__ import annotations

import torch

from penguin_model import ArchState, PenguinCore, StopReason, assemble_file
from penguin_model.tensor import (
    BF16_DTYPE,
    FP8_DTYPE,
    MREG_BYTES,
    MREG_ROWS,
    WEIGHT_TILE_COLS_FP8,
    WEIGHT_TILE_ROWS,
    bf16_tile_from_bytes,
    fp8_tile_to_bytes,
    weight_tile_to_bytes,
)

_PROGRAM_ROOT = "tests/vectors/programs/tensor/examples"
_ACT_DRAM_BASE = 0x8100_0000
_WEIGHT_DRAM_BASE = 0x8200_0000
_OUTPUT_DRAM_BASE = 0x8300_0000


def _deterministic_activation(rows: int, cols: int) -> torch.Tensor:
    indices = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return ((indices % 17) - 8) / 8


def _deterministic_weight(rows: int, cols: int) -> torch.Tensor:
    indices = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return (((indices * 3) % 19) - 9) / 7


def _bf16_reference_matmul(activation: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    activation_fp8 = activation.to(dtype=FP8_DTYPE).to(dtype=torch.float32)
    weight_fp8 = weights.to(dtype=FP8_DTYPE).to(dtype=torch.float32)
    acc = torch.zeros((activation_fp8.shape[0], weight_fp8.shape[1]), dtype=BF16_DTYPE)
    for inner in range(weight_fp8.shape[0]):
        product = activation_fp8[:, inner].unsqueeze(1) * weight_fp8[inner, :].unsqueeze(0)
        acc = (acc.to(torch.float32) + product).to(BF16_DTYPE)
    return acc


def _activation_tile_address(m_tile: int, k_tile: int, *, k_tiles: int) -> int:
    return _ACT_DRAM_BASE + (m_tile * k_tiles + k_tile) * MREG_BYTES


def _weight_tile_address(k_tile: int, n_tile: int, *, n_tiles: int) -> int:
    return _WEIGHT_DRAM_BASE + (k_tile * n_tiles + n_tile) * 512


def _output_tile_address(m_tile: int, n_tile: int, *, n_tiles: int) -> int:
    return _OUTPUT_DRAM_BASE + (m_tile * n_tiles + n_tile) * MREG_BYTES


def _preload_large_matmul_state() -> tuple[PenguinCore, torch.Tensor, torch.Tensor]:
    activation = _deterministic_activation(128, 64)
    weights = _deterministic_weight(64, 32)
    state = ArchState.with_memory_sizes()

    for m_tile in range(2):
        for k_tile in range(2):
            state.dram.write(
                _activation_tile_address(m_tile, k_tile, k_tiles=2),
                fp8_tile_to_bytes(
                    activation[
                        m_tile * MREG_ROWS : (m_tile + 1) * MREG_ROWS,
                        k_tile * WEIGHT_TILE_ROWS : (k_tile + 1) * WEIGHT_TILE_ROWS,
                    ]
                ),
            )

    for k_tile in range(2):
        for n_tile in range(2):
            state.dram.write(
                _weight_tile_address(k_tile, n_tile, n_tiles=2),
                weight_tile_to_bytes(
                    weights[
                        k_tile * WEIGHT_TILE_ROWS : (k_tile + 1) * WEIGHT_TILE_ROWS,
                        n_tile * WEIGHT_TILE_COLS_FP8 : (n_tile + 1) * WEIGHT_TILE_COLS_FP8,
                    ]
                ),
            )

    return PenguinCore(state=state), activation, weights


def test_large_matmul_first_partial_tile_matches_reference_and_updates_scalar_state() -> None:
    core, activation, weights = _preload_large_matmul_state()
    program = assemble_file(f"{_PROGRAM_ROOT}/matmul_large.S")

    perf = core.execute(program, max_instructions=23)

    expected = _bf16_reference_matmul(activation[:64, :32], weights[:32, :16])
    actual = bf16_tile_from_bytes(core.state.load_mreg(2))

    assert perf.instructions == 23
    assert core.state.stop_reason == StopReason.STEP_LIMIT
    assert torch.equal(actual, expected)
    assert core.state.read_xreg(9) == 2
    assert core.state.read_xreg(10) == _ACT_DRAM_BASE
    assert core.state.read_xreg(11) == _WEIGHT_DRAM_BASE


def test_large_matmul_first_output_tile_is_correct_in_tensor_and_dram_state() -> None:
    core, activation, weights = _preload_large_matmul_state()
    program = assemble_file(f"{_PROGRAM_ROOT}/matmul_large.S")

    perf = core.execute(program, max_instructions=48)

    expected = _bf16_reference_matmul(activation[:64, :64], weights[:64, :16])
    tensor_tile = bf16_tile_from_bytes(core.state.load_mreg(2))
    dram_tile = bf16_tile_from_bytes(core.state.dram.read(_output_tile_address(0, 0, n_tiles=2), MREG_BYTES))

    assert perf.instructions == 48
    assert core.state.stop_reason == StopReason.STEP_LIMIT
    assert torch.equal(tensor_tile, expected)
    assert torch.equal(dram_tile, expected)
    assert core.state.read_xreg(19) == _OUTPUT_DRAM_BASE
