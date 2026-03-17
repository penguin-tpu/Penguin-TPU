"""Specification-driven MXU and scale-register tests for the Penguin model."""

from __future__ import annotations

from dataclasses import replace

import torch

from penguin_model import (
    INSTRUCTION_LATENCY,
    MATMUL_LATENCY_CYCLES,
    MXU_PUSH_LATENCY_CYCLES,
    TENSOR_INSTRUCTION_SPECS,
    VLOAD_LATENCY_CYCLES,
    VSTORE_LATENCY_CYCLES,
    ArchState,
    Instruction,
    MXUMatmulAccType,
    MXUMatmulType,
    PenguinCore,
    ScaleImmType,
    ScaleMemType,
    StopReason,
    TensorMemType,
    WeightMemType,
)
from penguin_model.tensor import (
    BF16_DTYPE,
    FP8_DTYPE,
    MREG_BF16_COLS,
    MREG_BYTES,
    MREG_FP8_COLS,
    MREG_ROWS,
    WEIGHT_SLOT_BYTES,
    WEIGHT_TILE_COLS_FP8,
    WEIGHT_TILE_ROWS,
    bf16_tile_pair_from_bytes,
    bf16_tile_pair_to_bytes,
    fp8_tile_to_bytes,
    weight_tile_to_bytes,
)
from penguin_model.testbench import TEST_CORE_CONFIG, VMEM_BASE

REQUIRED_MXU_MNEMONICS = {
    "mxu.push.mxu0",
    "mxu.push.mxu1",
    "matmul.mxu0",
    "matmul.mxu1",
    "matmul.acc.mxu0",
    "matmul.acc.mxu1",
}


def _fresh_state(config=TEST_CORE_CONFIG) -> ArchState:
    return ArchState.from_config(config)


def _fresh_core(config=TEST_CORE_CONFIG) -> PenguinCore:
    return PenguinCore(config=config)


def _fp8_tile(values: list[list[float]]) -> torch.Tensor:
    tile = torch.zeros((MREG_ROWS, MREG_FP8_COLS), dtype=torch.float32)
    value_tensor = torch.tensor(values, dtype=torch.float32)
    tile[: value_tensor.shape[0], : value_tensor.shape[1]] = value_tensor
    return tile


def _weight_tile(values: list[list[float]]) -> torch.Tensor:
    tile = torch.zeros((WEIGHT_TILE_ROWS, WEIGHT_TILE_COLS_FP8), dtype=torch.float32)
    value_tensor = torch.tensor(values, dtype=torch.float32)
    tile[: value_tensor.shape[0], : value_tensor.shape[1]] = value_tensor
    return tile


def _bf16_result_tile(values: list[list[float]]) -> torch.Tensor:
    tile = torch.zeros((MREG_ROWS, WEIGHT_TILE_COLS_FP8), dtype=torch.float32)
    value_tensor = torch.tensor(values, dtype=torch.float32)
    tile[: value_tensor.shape[0], : value_tensor.shape[1]] = value_tensor
    return tile.to(BF16_DTYPE)


def _store_activation(state: ArchState, index: int, tile: torch.Tensor) -> None:
    state.store_mreg(index, fp8_tile_to_bytes(tile, config=state.config))


def _store_weight(state: ArchState, mxu: int, slot: int, tile: torch.Tensor) -> None:
    state.store_weight_slot(mxu, slot, weight_tile_to_bytes(tile, config=state.config))


def _store_partial_pair(state: ArchState, index: int, tile: torch.Tensor) -> None:
    raw_lo, raw_hi = bf16_tile_pair_to_bytes(tile, config=state.config)
    state.store_mreg(index, raw_lo)
    state.store_mreg(index + 1, raw_hi)


def _read_result_pair(state: ArchState, index: int) -> torch.Tensor:
    return bf16_tile_pair_from_bytes(
        state.load_mreg(index),
        state.load_mreg(index + 1),
        config=state.config,
    ).to(torch.float32)


def _reference_scaled_matmul(
    activation: torch.Tensor,
    weights: torch.Tensor,
    *,
    scale_a_exp: int = 0,
    scale_b_exp: int = 0,
    partial: torch.Tensor | None = None,
) -> torch.Tensor:
    activation_fp8 = activation.to(dtype=FP8_DTYPE).to(dtype=torch.float32)
    weight_fp8 = weights.to(dtype=FP8_DTYPE).to(dtype=torch.float32)
    result = activation_fp8 @ weight_fp8
    result *= float(2.0**scale_a_exp) * float(2.0**scale_b_exp)
    if partial is not None:
        result = result + partial.to(torch.float32)
    return result.to(dtype=BF16_DTYPE).to(dtype=torch.float32)


def test_mxu_instruction_family_registers_with_expected_latency_classes() -> None:
    assert REQUIRED_MXU_MNEMONICS <= set(TENSOR_INSTRUCTION_SPECS)
    assert INSTRUCTION_LATENCY["vload"] == VLOAD_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vstore"] == VSTORE_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["mxu.push.mxu0"] == MXU_PUSH_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["mxu.push.mxu1"] == MXU_PUSH_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["matmul.mxu0"] == MATMUL_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["matmul.mxu1"] == MATMUL_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["matmul.acc.mxu0"] == MATMUL_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["matmul.acc.mxu1"] == MATMUL_LATENCY_CYCLES


def test_scale_register_instructions_write_expected_raw_payloads() -> None:
    state = _fresh_state()
    state.vmem.write(VMEM_BASE + 0x40, torch.tensor([0xFF], dtype=torch.uint8))
    core = PenguinCore(state=state)

    perf = core.execute(
        [
            Instruction("seli", ScaleImmType(ed=1, imm=0x05)),
            Instruction("seld", ScaleMemType(ed=2, rs1=0, imm=VMEM_BASE + 0x40)),
        ]
    )

    assert state.read_ereg(1) == 0x05
    assert state.read_ereg(2) == 0xFF
    assert perf.instructions_by_opcode == {"seli": 1, "seld": 1}
    assert perf.bytes_read == 1
    assert perf.bytes_written == 0


@torch.no_grad()
def test_vload_push_and_vstore_use_whole_tensor_and_weight_images() -> None:
    activation = _fp8_tile([[1.0, 2.0, 3.0], [-4.0, 0.5, 1.5]])
    weights = _weight_tile([[1.0, -1.0], [2.0, 0.25], [0.5, 4.0]])
    result_tile = _bf16_result_tile([[7.0, 8.0], [9.0, 10.0]])
    result_lo, _ = bf16_tile_pair_to_bytes(result_tile, config=TEST_CORE_CONFIG)

    state = _fresh_state()
    act_addr = VMEM_BASE + 0x0000
    weight_addr = VMEM_BASE + 0x2000
    store_addr = VMEM_BASE + 0x4000
    state.vmem.write(act_addr, fp8_tile_to_bytes(activation, config=state.config))
    state.vmem.write(weight_addr, weight_tile_to_bytes(weights, config=state.config))
    state.store_mreg(5, result_lo)
    core = PenguinCore(state=state)

    perf = core.execute(
        [
            Instruction("vload", TensorMemType(mreg=1, rs1=0, imm=act_addr)),
            Instruction("mxu.push.mxu0", WeightMemType(slot=0, rs1=0, imm=weight_addr)),
            Instruction("vstore", TensorMemType(mreg=5, rs1=0, imm=store_addr)),
        ]
    )

    assert torch.equal(state.load_mreg(1), fp8_tile_to_bytes(activation, config=state.config))
    assert torch.equal(
        state.load_weight_slot(0, 0),
        weight_tile_to_bytes(weights, config=state.config),
    )
    assert torch.equal(state.vmem.read(store_addr, MREG_BYTES), result_lo)
    assert perf.instructions == 3
    assert perf.bytes_read == MREG_BYTES + WEIGHT_SLOT_BYTES
    assert perf.bytes_written == MREG_BYTES


@torch.no_grad()
def test_matmul_writes_paired_bf16_result_registers() -> None:
    activation = _fp8_tile([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]])
    weights = _weight_tile([[1.0, -1.0], [0.5, 2.0], [-3.0, 0.25]])
    state = _fresh_state()
    _store_activation(state, 1, activation)
    _store_weight(state, 0, 0, weights)
    core = PenguinCore(state=state)

    perf = core.execute(
        [
            Instruction("seli", ScaleImmType(ed=0, imm=0)),
            Instruction("seli", ScaleImmType(ed=1, imm=0)),
            Instruction("matmul.mxu0", MXUMatmulType(md=2, ms=1, ws=0, ea=0, eb=1)),
        ]
    )

    expected = _reference_scaled_matmul(activation, weights)
    assert torch.equal(_read_result_pair(state, 2), expected)
    assert perf.instructions_by_opcode == {"seli": 2, "matmul.mxu0": 1}
    assert perf.cycles == 2 + TEST_CORE_CONFIG.tensor.matmul_latency_cycles


@torch.no_grad()
def test_matmul_acc_uses_paired_partial_and_scale_operands() -> None:
    activation = _fp8_tile([[1.0, 2.0], [3.0, 4.0]])
    weights = _weight_tile([[0.5, 1.0], [-2.0, 0.25]])
    partial = _bf16_result_tile([[10.0, -3.0], [1.5, 2.5]])
    state = _fresh_state()
    _store_activation(state, 1, activation)
    _store_weight(state, 0, 0, weights)
    _store_partial_pair(state, 8, partial)
    core = PenguinCore(state=state)

    perf = core.execute(
        [
            Instruction("seli", ScaleImmType(ed=0, imm=1)),
            Instruction("seli", ScaleImmType(ed=1, imm=0xFF)),
            Instruction(
                "matmul.acc.mxu0",
                MXUMatmulAccType(md=4, ms=1, ws=0, mp=8, ea=0, eb=1),
            ),
        ]
    )

    expected = _reference_scaled_matmul(
        activation,
        weights,
        scale_a_exp=1,
        scale_b_exp=-1,
        partial=partial,
    )
    assert torch.equal(_read_result_pair(state, 4), expected)
    assert perf.instructions_by_opcode == {"seli": 2, "matmul.acc.mxu0": 1}


@torch.no_grad()
def test_scale_registers_loaded_from_immediate_and_memory_affect_matmul() -> None:
    activation = _fp8_tile([[1.0, 0.0], [0.0, 1.0]])
    weights = _weight_tile([[2.0, 0.0], [0.0, 4.0]])
    state = _fresh_state()
    _store_activation(state, 1, activation)
    _store_weight(state, 0, 0, weights)
    state.vmem.write(VMEM_BASE + 0x80, torch.tensor([0xFF], dtype=torch.uint8))
    core = PenguinCore(state=state)

    perf = core.execute(
        [
            Instruction("seli", ScaleImmType(ed=0, imm=1)),
            Instruction("seld", ScaleMemType(ed=1, rs1=0, imm=VMEM_BASE + 0x80)),
            Instruction("matmul.mxu0", MXUMatmulType(md=6, ms=1, ws=0, ea=0, eb=1)),
        ]
    )

    expected = _reference_scaled_matmul(
        activation,
        weights,
        scale_a_exp=1,
        scale_b_exp=-1,
    )
    assert torch.equal(_read_result_pair(state, 6), expected)
    assert perf.bytes_read == 1


def test_matmul_rejects_illegal_paired_destination_base() -> None:
    activation = _fp8_tile([[1.0]])
    weights = _weight_tile([[1.0]])
    state = _fresh_state()
    _store_activation(state, 1, activation)
    _store_weight(state, 0, 0, weights)
    core = PenguinCore(state=state)

    perf = core.execute(
        [
            Instruction("seli", ScaleImmType(ed=0, imm=0)),
            Instruction("seli", ScaleImmType(ed=1, imm=0)),
            Instruction("matmul.mxu0", MXUMatmulType(md=63, ms=1, ws=0, ea=0, eb=1)),
        ]
    )

    assert state.stop_reason == StopReason.ILLEGAL_TENSOR_REGISTER_PAIR
    assert perf.instructions == 3


def test_matmul_perf_model_uses_configured_latency() -> None:
    config = replace(
        TEST_CORE_CONFIG,
        tensor=replace(TEST_CORE_CONFIG.tensor, matmul_latency_cycles=17),
    )
    state = _fresh_state(config)
    _store_activation(state, 1, _fp8_tile([[1.0]]))
    _store_weight(state, 0, 0, _weight_tile([[1.0]]))
    core = PenguinCore(state=state, config=config)

    perf = core.execute(
        [
            Instruction("seli", ScaleImmType(ed=0, imm=0)),
            Instruction("seli", ScaleImmType(ed=1, imm=0)),
            Instruction("matmul.mxu0", MXUMatmulType(md=2, ms=1, ws=0, ea=0, eb=1)),
        ]
    )

    assert perf.instructions == 3
    assert perf.cycles == 19
