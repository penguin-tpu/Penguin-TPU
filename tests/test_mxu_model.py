"""Specification-driven MXU and scale-register tests for the Penguin model."""

from __future__ import annotations

from dataclasses import replace

import torch

from penguin_model import (
    ACCUM_BUFFER_BYTES,
    INSTRUCTION_LATENCY,
    MATMUL_LATENCY_CYCLES,
    TENSOR_INSTRUCTION_SPECS,
    VLOAD_LATENCY_CYCLES,
    VLOAD_WEIGHT_LATENCY_CYCLES,
    VMATPOP_ACC_BF16_LATENCY_CYCLES,
    VMATPOP_ACC_FP8_LATENCY_CYCLES,
    VMATPUSH_ACC_LATENCY_CYCLES,
    VMATPUSH_WEIGHT_LATENCY_CYCLES,
    VSTORE_LATENCY_CYCLES,
    ArchState,
    DelayType,
    Instruction,
    MXUAccumulatorType,
    MXUMatmulAccType,
    MXUMatmulType,
    Sim,
    ScaleImmType,
    ScaleMemType,
    StopReason,
    TensorMemType,
    WeightMemType,
    WeightTensorType,
)
from penguin_model.tensor import (
    BF16_DTYPE,
    FP8_DTYPE,
    MREG_BYTES,
    MREG_FP8_COLS,
    MREG_ROWS,
    WEIGHT_SLOT_BYTES,
    WEIGHT_TILE_COLS_FP8,
    WEIGHT_TILE_ROWS,
    accum_tile_from_bytes,
    accum_tile_to_bytes,
    bf16_tile_pair_from_bytes,
    bf16_tile_pair_to_bytes,
    export_accum_to_fp8,
    fp8_tile_to_bytes,
    weight_tile_to_bytes,
)
from penguin_model.testbench import TEST_CORE_CONFIG, VMEM_BASE

REQUIRED_MXU_MNEMONICS = {
    "vmatpush.weight.mxu0",
    "vmatpush.weight.mxu1",
    "vload.weight.mxu0",
    "vload.weight.mxu1",
    "vmatpush.acc.fp8.mxu0",
    "vmatpush.acc.fp8.mxu1",
    "vmatpush.acc.bf16.mxu0",
    "vmatpush.acc.bf16.mxu1",
    "vmatpop.bf16.acc.mxu0",
    "vmatpop.bf16.acc.mxu1",
    "vmatpop.fp8.acc.mxu0",
    "vmatpop.fp8.acc.mxu1",
    "vmatmul.mxu0",
    "vmatmul.mxu1",
    "vmatmul.acc.mxu0",
    "vmatmul.acc.mxu1",
}


def _fresh_state(config=TEST_CORE_CONFIG) -> ArchState:
    return ArchState.from_config(config)


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


def _store_accum_tile(state: ArchState, mxu: int, tile: torch.Tensor) -> None:
    state.store_accum_buffer(mxu, accum_tile_to_bytes(tile, config=state.config))


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


def _reference_matmul(
    activation: torch.Tensor,
    weights: torch.Tensor,
    *,
    accum: torch.Tensor | None = None,
) -> torch.Tensor:
    activation_fp8 = activation.to(dtype=FP8_DTYPE).to(dtype=torch.float32)
    weight_fp8 = weights.to(dtype=FP8_DTYPE).to(dtype=torch.float32)
    result = activation_fp8 @ weight_fp8
    if accum is not None:
        result = result + accum.to(torch.float32)
    return result.to(dtype=BF16_DTYPE).to(dtype=torch.float32)


def test_mxu_instruction_family_registers_with_expected_latency_classes() -> None:
    assert REQUIRED_MXU_MNEMONICS <= set(TENSOR_INSTRUCTION_SPECS)
    assert INSTRUCTION_LATENCY["vload"] == VLOAD_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vstore"] == VSTORE_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vmatpush.weight.mxu0"] == VMATPUSH_WEIGHT_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vload.weight.mxu0"] == VLOAD_WEIGHT_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vmatpush.acc.bf16.mxu0"] == VMATPUSH_ACC_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vmatpop.bf16.acc.mxu0"] == VMATPOP_ACC_BF16_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vmatpop.fp8.acc.mxu0"] == VMATPOP_ACC_FP8_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vmatmul.mxu0"] == MATMUL_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vmatmul.acc.mxu0"] == MATMUL_LATENCY_CYCLES


def test_scale_register_instructions_write_expected_raw_payloads() -> None:
    state = _fresh_state()
    state.vmem.write(VMEM_BASE + 0x40, torch.tensor([0xFF], dtype=torch.uint8))
    core = Sim(state=state)

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
def test_vload_weight_push_and_vstore_use_whole_tensor_images() -> None:
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
    state.write_xreg(2, weight_addr)
    core = Sim(state=state)

    perf = core.execute(
        [
            Instruction("vload", TensorMemType(mreg=1, rs1=0, imm=act_addr)),
            Instruction("vload.weight.mxu0", WeightMemType(slot=0, rs1=2, imm=0)),
            Instruction("vstore", TensorMemType(mreg=5, rs1=0, imm=store_addr)),
        ]
    )

    assert torch.equal(state.load_mreg(1), fp8_tile_to_bytes(activation, config=state.config))
    assert torch.equal(state.load_weight_slot(0, 0), weight_tile_to_bytes(weights, config=state.config))
    assert torch.equal(state.vmem.read(store_addr, MREG_BYTES), result_lo)
    assert perf.instructions == 3


@torch.no_grad()
def test_vmatpush_and_vmatpop_move_weight_and_accumulator_images() -> None:
    activation = _fp8_tile([[1.0, 2.0], [3.0, 4.0]])
    accum_tile = _bf16_result_tile([[5.0, -1.0], [0.5, 2.5]])
    state = _fresh_state()
    _store_activation(state, 1, activation)
    _store_partial_pair(state, 8, accum_tile)
    core = Sim(state=state)

    perf = core.execute(
        [
            Instruction("vmatpush.weight.mxu0", WeightTensorType(slot=1, ms=1)),
            Instruction("vmatpush.acc.bf16.mxu0", MXUAccumulatorType(mreg=8)),
            Instruction("delay", DelayType(cycles=VMATPUSH_ACC_LATENCY_CYCLES - 1)),
            Instruction("vmatpop.bf16.acc.mxu0", MXUAccumulatorType(mreg=10)),
            Instruction("delay", DelayType(cycles=VMATPOP_ACC_BF16_LATENCY_CYCLES - 1)),
            Instruction("vmatpop.fp8.acc.mxu0", MXUAccumulatorType(mreg=12)),
        ]
    )

    assert torch.equal(state.load_weight_slot(0, 1), state.load_mreg(1))
    assert torch.equal(
        accum_tile_from_bytes(state.load_accum_buffer(0), config=state.config).to(torch.float32),
        accum_tile.to(torch.float32),
    )
    assert torch.equal(_read_result_pair(state, 10), accum_tile.to(torch.float32))
    assert torch.equal(
        state.load_mreg(12),
        export_accum_to_fp8(state.load_accum_buffer(0), config=state.config),
    )
    assert perf.instructions == 6


@torch.no_grad()
def test_vmatmul_overwrites_accumulator_and_bf16_pop_exports_result() -> None:
    activation = _fp8_tile([[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]])
    weights = _weight_tile([[1.0, -1.0], [0.5, 2.0], [-3.0, 0.25]])
    state = _fresh_state()
    _store_activation(state, 1, activation)
    _store_weight(state, 0, 0, weights)
    core = Sim(state=state)

    perf = core.execute(
        [
            Instruction("vmatmul.mxu0", MXUMatmulType(ms=1, ws=0)),
            Instruction("delay", DelayType(cycles=MATMUL_LATENCY_CYCLES - 1)),
            Instruction("vmatpop.bf16.acc.mxu0", MXUAccumulatorType(mreg=2)),
        ]
    )

    expected = _reference_matmul(activation, weights)
    assert torch.equal(_read_result_pair(state, 2), expected)
    assert perf.instructions_by_opcode == {
        "delay": 1,
        "vmatmul.mxu0": 1,
        "vmatpop.bf16.acc.mxu0": 1,
    }


@torch.no_grad()
def test_vmatmul_acc_accumulates_into_existing_local_accumulator() -> None:
    activation = _fp8_tile([[1.0, 2.0], [3.0, 4.0]])
    weights = _weight_tile([[0.5, 1.0], [-2.0, 0.25]])
    partial = _bf16_result_tile([[10.0, -3.0], [1.5, 2.5]])
    state = _fresh_state()
    _store_activation(state, 1, activation)
    _store_weight(state, 0, 0, weights)
    _store_accum_tile(state, 0, partial)
    core = Sim(state=state)

    perf = core.execute(
        [
            Instruction("vmatmul.acc.mxu0", MXUMatmulAccType(ms=1, ws=0)),
            Instruction("vmatpop.bf16.acc.mxu0", MXUAccumulatorType(mreg=4)),
        ]
    )

    expected = _reference_matmul(activation, weights, accum=partial)
    assert torch.equal(_read_result_pair(state, 4), expected)
    assert perf.instructions_by_opcode == {
        "vmatmul.acc.mxu0": 1,
        "vmatpop.bf16.acc.mxu0": 1,
    }


def test_vmatpush_and_vmatpop_reject_illegal_paired_register_base() -> None:
    state = _fresh_state()
    core = Sim(state=state)

    perf = core.execute([Instruction("vmatpush.acc.bf16.mxu0", MXUAccumulatorType(mreg=63))])

    assert state.stop_reason == StopReason.ILLEGAL_INSTRUCTION
    assert perf.instructions == 0


def test_vmatmul_perf_model_uses_configured_latency() -> None:
    config = replace(
        TEST_CORE_CONFIG,
        tensor=replace(TEST_CORE_CONFIG.tensor, matmul_latency_cycles=17),
    )
    state = _fresh_state(config)
    _store_activation(state, 1, _fp8_tile([[1.0]]))
    _store_weight(state, 0, 0, _weight_tile([[1.0]]))
    core = Sim(state=state, config=config)

    perf = core.execute(
        [
            Instruction("vmatmul.mxu0", MXUMatmulType(ms=1, ws=0)),
            Instruction("vmatpop.bf16.acc.mxu0", MXUAccumulatorType(mreg=2)),
        ]
    )

    assert perf.instructions == 2
    assert perf.cycles >= config.tensor.matmul_latency_cycles + config.vmatpop_acc_bf16_latency_cycles


def test_accum_buffer_size_matches_spec() -> None:
    state = _fresh_state()
    assert state.load_accum_buffer(0).numel() == ACCUM_BUFFER_BYTES
    assert ACCUM_BUFFER_BYTES == TEST_CORE_CONFIG.matmul_result_bytes
