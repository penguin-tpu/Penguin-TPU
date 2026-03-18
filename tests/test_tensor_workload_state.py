"""Intermediate-state verification for large DMA-backed tensor workloads."""

from __future__ import annotations

import torch

from penguin_model import ArchState, Sim, StopReason, load_mapped_program
from penguin_model.tensor import (
    BF16_DTYPE,
    FP8_DTYPE,
    MREG_BYTES,
    MREG_ROWS,
    WEIGHT_TILE_COLS_FP8,
    WEIGHT_TILE_ROWS,
    bf16_tile_pair_from_bytes,
    compute_bf16_matmul,
    fp8_tile_to_bytes,
    weight_tile_to_bytes,
)

PROGRAM_ROOT = "tests/vectors/programs/tensor/examples"
ACT_DRAM_BASE = 0x8100_0000
WEIGHT_DRAM_BASE = 0x8200_0000
OUTPUT_DRAM_BASE = 0x8300_0000
RESULT_TILE_BYTES = MREG_ROWS * WEIGHT_TILE_COLS_FP8 * 2


def _deterministic_activation(rows: int, cols: int) -> torch.Tensor:
    indices = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return ((indices % 17) - 8) / 8


def _deterministic_weight(rows: int, cols: int) -> torch.Tensor:
    indices = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    return (((indices * 3) % 19) - 9) / 7


def _reference_tile_matmul(activation: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    raw_lo, raw_hi = compute_bf16_matmul(
        fp8_tile_to_bytes(activation),
        weight_tile_to_bytes(weights),
        0,
        0,
    )
    return bf16_tile_pair_from_bytes(raw_lo, raw_hi).to(torch.float32)


def _reference_tiled_accumulation(activation: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    first_lo, first_hi = compute_bf16_matmul(
        fp8_tile_to_bytes(activation[:, :WEIGHT_TILE_ROWS]),
        weight_tile_to_bytes(weights[:WEIGHT_TILE_ROWS, :]),
        0,
        0,
    )
    second_lo, second_hi = compute_bf16_matmul(
        fp8_tile_to_bytes(activation[:, WEIGHT_TILE_ROWS:]),
        weight_tile_to_bytes(weights[WEIGHT_TILE_ROWS:, :]),
        0,
        0,
        partial_raw=(first_lo, first_hi),
    )
    return bf16_tile_pair_from_bytes(second_lo, second_hi).to(torch.float32)


def _activation_tile_address(m_tile: int, k_tile: int, *, k_tiles: int) -> int:
    return ACT_DRAM_BASE + (m_tile * k_tiles + k_tile) * MREG_BYTES


def _weight_tile_address(k_tile: int, n_tile: int, *, n_tiles: int) -> int:
    return WEIGHT_DRAM_BASE + (k_tile * n_tiles + n_tile) * (WEIGHT_TILE_ROWS * WEIGHT_TILE_COLS_FP8)


def _output_tile_address(m_tile: int, n_tile: int, *, n_tiles: int) -> int:
    return OUTPUT_DRAM_BASE + (m_tile * n_tiles + n_tile) * RESULT_TILE_BYTES


def _preload_large_matmul_state() -> tuple[Sim, torch.Tensor, torch.Tensor]:
    activation = _deterministic_activation(128, 128)
    weights = _deterministic_weight(128, 128)
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

    return Sim(state=state), activation, weights


def test_large_matmul_first_partial_tile_matches_reference_and_updates_scalar_state() -> None:
    core, activation, weights = _preload_large_matmul_state()
    program = load_mapped_program(f"{PROGRAM_ROOT}/matmul_large.S")

    perf = core.execute(program, max_instructions=56)

    expected = _reference_tile_matmul(activation[:64, :64], weights[:64, :64]).to(torch.float32)
    actual = bf16_tile_pair_from_bytes(
        core.state.load_mreg(2),
        core.state.load_mreg(3),
    ).to(torch.float32)

    assert perf.instructions == 60
    assert core.state.stop_reason == StopReason.STEP_LIMIT
    assert torch.equal(actual, expected)
    assert core.state.read_xreg(10) == ACT_DRAM_BASE + 0x1000
    assert core.state.read_xreg(11) == WEIGHT_DRAM_BASE + 0x2000


def test_large_matmul_first_output_tile_is_correct_in_tensor_and_dram_state() -> None:
    core, activation, weights = _preload_large_matmul_state()
    program = load_mapped_program(f"{PROGRAM_ROOT}/matmul_large.S")

    perf = core.execute(program, max_instructions=79)

    expected = _reference_tiled_accumulation(activation[:64, :128], weights[:128, :64]).to(torch.float32)
    tensor_tile = bf16_tile_pair_from_bytes(
        core.state.load_mreg(2),
        core.state.load_mreg(3),
    ).to(torch.float32)
    assert perf.instructions == 81
    assert core.state.stop_reason == StopReason.STEP_LIMIT
    assert torch.equal(tensor_tile, expected)
    assert core.state.read_xreg(19) == OUTPUT_DRAM_BASE
