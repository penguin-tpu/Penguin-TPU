"""Specification-driven XLU tests for the Penguin functional model."""

from __future__ import annotations

from dataclasses import replace

import torch

from penguin_model import (
    INSTRUCTION_LATENCY,
    TENSOR_INSTRUCTION_SPECS,
    ArchState,
    Instruction,
    Sim,
    XLUTransposeType,
    XLU_TRANSPOSE_LATENCY_CYCLES,
)
from penguin_model.tensor import (
    bf16_tile_from_bytes,
    bf16_tile_to_bytes,
    bf16_transposed_tile_from_bytes,
)
from penguin_model.testbench import TEST_CORE_CONFIG


def _fresh_state(config=TEST_CORE_CONFIG) -> ArchState:
    return ArchState.from_config(config)


def _fresh_core(config=TEST_CORE_CONFIG) -> Sim:
    return Sim(config=config)


def _tile(values: list[list[float]]) -> torch.Tensor:
    rows = TEST_CORE_CONFIG.tensor.mreg_rows
    cols = TEST_CORE_CONFIG.tensor.mreg_row_bytes // 2
    tile = torch.zeros((rows, cols), dtype=torch.float32)
    value_tensor = torch.tensor(values, dtype=torch.float32)
    tile[: value_tensor.shape[0], : value_tensor.shape[1]] = value_tensor
    return tile


def test_xlu_transpose_registers_with_latency_view() -> None:
    assert "transpose.xlu" in TENSOR_INSTRUCTION_SPECS
    assert "reduce.max.xlu" in TENSOR_INSTRUCTION_SPECS
    assert "reduce.sum.xlu" in TENSOR_INSTRUCTION_SPECS
    assert INSTRUCTION_LATENCY["transpose.xlu"] == XLU_TRANSPOSE_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["reduce.max.xlu"] == XLU_TRANSPOSE_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["reduce.sum.xlu"] == XLU_TRANSPOSE_LATENCY_CYCLES


@torch.no_grad()
def test_xlu_transpose_matches_pytorch_bf16_transpose() -> None:
    src = _tile(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [-1.0, -2.0, -3.0, -4.0],
        ]
    )
    core = _fresh_core()
    core.state.store_mreg(1, bf16_tile_to_bytes(src, config=core.state.config))

    perf = core.execute([Instruction("transpose.xlu", XLUTransposeType(md=2, ms=1))])

    actual = bf16_transposed_tile_from_bytes(core.state.load_mreg(2), config=core.state.config)
    expected = src.to(torch.bfloat16).transpose(0, 1).contiguous()

    assert torch.equal(actual, expected)
    assert perf.instructions == 1
    assert perf.instructions_by_opcode == {"transpose.xlu": 1}
    assert perf.cycles == TEST_CORE_CONFIG.xlu.transpose_latency_cycles + 3


@torch.no_grad()
def test_xlu_row_reductions_match_pytorch_bf16_reference() -> None:
    src = _tile(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, -6.0, 7.0, -8.0],
            [-1.0, -2.0, -3.0, -4.0],
        ]
    )
    core = _fresh_core()
    core.state.store_mreg(1, bf16_tile_to_bytes(src, config=core.state.config))

    perf = core.execute(
        [
            Instruction("reduce.max.xlu", XLUTransposeType(md=2, ms=1)),
            Instruction("reduce.sum.xlu", XLUTransposeType(md=3, ms=1)),
        ]
    )

    reduced_max = bf16_tile_from_bytes(core.state.load_mreg(2), config=core.state.config).to(torch.float32)
    reduced_sum = bf16_tile_from_bytes(core.state.load_mreg(3), config=core.state.config).to(torch.float32)
    expected_max = torch.amax(src.to(torch.bfloat16).to(torch.float32), dim=1, keepdim=True).expand_as(src).to(torch.bfloat16).to(torch.float32)
    expected_sum = torch.sum(src.to(torch.bfloat16).to(torch.float32), dim=1, keepdim=True).expand_as(src).to(torch.bfloat16).to(torch.float32)

    assert torch.equal(reduced_max, expected_max)
    assert torch.equal(reduced_sum, expected_sum)
    assert perf.instructions_by_opcode == {"reduce.max.xlu": 1, "reduce.sum.xlu": 1}
    assert perf.cycles == TEST_CORE_CONFIG.xlu.transpose_latency_cycles + 4


def test_xlu_perf_model_uses_configured_transpose_latency() -> None:
    config = replace(
        TEST_CORE_CONFIG,
        xlu=replace(TEST_CORE_CONFIG.xlu, transpose_latency_cycles=7),
    )
    state = _fresh_state(config)
    state.store_mreg(1, bf16_tile_to_bytes(_tile([[1.0, 2.0], [3.0, 4.0]]), config=config))
    core = Sim(state=state, config=config)

    perf = core.execute([Instruction("transpose.xlu", XLUTransposeType(md=2, ms=1))])

    assert perf.instructions == 1
    assert perf.cycles == config.xlu.transpose_latency_cycles + 3
