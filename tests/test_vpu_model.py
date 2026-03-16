"""Specification-driven VPU tests for the Penguin functional model."""

from __future__ import annotations

from dataclasses import replace

import torch

from penguin_model import (
    INSTRUCTION_LATENCY,
    TENSOR_INSTRUCTION_SPECS,
    ArchState,
    Instruction,
    PenguinCore,
    VPUBinaryType,
    VPUUnaryType,
    VPU_SIMPLE_OP_LATENCY_CYCLES,
)
from penguin_model.tensor import bf16_tile_from_bytes, bf16_tile_to_bytes
from penguin_model.testbench import TEST_CORE_CONFIG

REQUIRED_VPU_MNEMONICS = {"vadd", "vmul", "vmax", "vmin", "vrelu", "vmov"}


def _fresh_state(config=TEST_CORE_CONFIG) -> ArchState:
    return ArchState.from_config(config)


def _fresh_core(config=TEST_CORE_CONFIG) -> PenguinCore:
    return PenguinCore(config=config)


def _tile(values: list[list[float]]) -> torch.Tensor:
    rows = TEST_CORE_CONFIG.tensor.mreg_rows
    cols = TEST_CORE_CONFIG.tensor.mreg_row_bytes // 2
    tile = torch.zeros((rows, cols), dtype=torch.float32)
    value_tensor = torch.tensor(values, dtype=torch.float32)
    tile[: value_tensor.shape[0], : value_tensor.shape[1]] = value_tensor
    return tile


def _write_bf16_tile(state: ArchState, index: int, tile: torch.Tensor) -> None:
    state.store_mreg(index, bf16_tile_to_bytes(tile, config=state.config))


def _read_bf16_tile(state: ArchState, index: int) -> torch.Tensor:
    return bf16_tile_from_bytes(state.load_mreg(index), config=state.config).to(torch.float32)


def test_vpu_instruction_family_registers_with_simple_latency_class() -> None:
    assert REQUIRED_VPU_MNEMONICS <= set(TENSOR_INSTRUCTION_SPECS)
    assert INSTRUCTION_LATENCY["vadd"] == VPU_SIMPLE_OP_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vmul"] == VPU_SIMPLE_OP_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vmax"] == VPU_SIMPLE_OP_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vmin"] == VPU_SIMPLE_OP_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vrelu"] == VPU_SIMPLE_OP_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vmov"] == VPU_SIMPLE_OP_LATENCY_CYCLES


@torch.no_grad()
def test_vpu_binary_ops_apply_bf16_elementwise_semantics() -> None:
    lhs = _tile([[1.0, -2.0, 3.5], [4.0, -5.0, 6.0]])
    rhs = _tile([[0.5, 8.0, -1.5], [-4.0, -1.0, 2.0]])
    core = _fresh_core()
    _write_bf16_tile(core.state, 1, lhs)
    _write_bf16_tile(core.state, 2, rhs)

    program = [
        Instruction("vadd", VPUBinaryType(md=10, ms1=1, ms2=2)),
        Instruction("vmul", VPUBinaryType(md=11, ms1=1, ms2=2)),
        Instruction("vmax", VPUBinaryType(md=12, ms1=1, ms2=2)),
        Instruction("vmin", VPUBinaryType(md=13, ms1=1, ms2=2)),
    ]
    perf = core.execute(program)

    assert torch.equal(
        _read_bf16_tile(core.state, 10),
        torch.add(lhs, rhs).to(torch.bfloat16).to(torch.float32),
    )
    assert torch.equal(
        _read_bf16_tile(core.state, 11),
        torch.mul(lhs, rhs).to(torch.bfloat16).to(torch.float32),
    )
    assert torch.equal(
        _read_bf16_tile(core.state, 12),
        torch.maximum(lhs, rhs).to(torch.bfloat16).to(torch.float32),
    )
    assert torch.equal(
        _read_bf16_tile(core.state, 13),
        torch.minimum(lhs, rhs).to(torch.bfloat16).to(torch.float32),
    )
    assert perf.instructions_by_opcode == {
        "vadd": 1,
        "vmul": 1,
        "vmax": 1,
        "vmin": 1,
    }
    assert perf.cycles == 4 * TEST_CORE_CONFIG.vpu.simple_op_latency_cycles


@torch.no_grad()
def test_vrelu_and_vmov_operate_on_whole_bf16_registers() -> None:
    src = _tile([[-1.0, 0.0, 1.5], [2.0, -3.0, 4.25]])
    core = _fresh_core()
    _write_bf16_tile(core.state, 5, src)
    _write_bf16_tile(core.state, 6, torch.full_like(src, -7.0))

    perf = core.execute(
        [
            Instruction("vrelu", VPUUnaryType(md=6, ms=5)),
            Instruction("vmov", VPUUnaryType(md=7, ms=6)),
        ]
    )

    relu_expected = torch.clamp_min(src, 0.0).to(torch.bfloat16).to(torch.float32)
    assert torch.equal(_read_bf16_tile(core.state, 6), relu_expected)
    assert torch.equal(_read_bf16_tile(core.state, 7), relu_expected)
    assert perf.instructions_by_opcode == {"vrelu": 1, "vmov": 1}
    assert perf.cycles == 2 * TEST_CORE_CONFIG.vpu.simple_op_latency_cycles


@torch.no_grad()
def test_vpu_in_place_destinations_use_prewrite_source_values() -> None:
    lhs = _tile([[1.0, -2.5], [3.25, 4.5]])
    rhs = _tile([[0.5, 8.0], [-1.0, 2.0]])
    src = _tile([[-7.0, 0.0], [1.5, -3.0]])
    core = _fresh_core()
    _write_bf16_tile(core.state, 1, lhs)
    _write_bf16_tile(core.state, 2, rhs)
    _write_bf16_tile(core.state, 3, src)

    perf = core.execute(
        [
            Instruction("vadd", VPUBinaryType(md=1, ms1=1, ms2=2)),
            Instruction("vrelu", VPUUnaryType(md=3, ms=3)),
        ]
    )

    assert torch.equal(
        _read_bf16_tile(core.state, 1),
        torch.add(lhs, rhs).to(torch.bfloat16).to(torch.float32),
    )
    assert torch.equal(_read_bf16_tile(core.state, 2), rhs.to(torch.bfloat16).to(torch.float32))
    assert torch.equal(
        _read_bf16_tile(core.state, 3),
        torch.clamp_min(src, 0.0).to(torch.bfloat16).to(torch.float32),
    )
    assert perf.instructions_by_opcode == {"vadd": 1, "vrelu": 1}


def test_vpu_perf_model_uses_configured_simple_latency() -> None:
    config = replace(
        TEST_CORE_CONFIG,
        vpu=replace(TEST_CORE_CONFIG.vpu, simple_op_latency_cycles=5),
    )
    state = _fresh_state(config)
    _write_bf16_tile(state, 1, _tile([[1.0, 2.0], [3.0, 4.0]]))
    _write_bf16_tile(state, 2, _tile([[5.0, 6.0], [7.0, 8.0]]))
    core = PenguinCore(state=state, config=config)

    perf = core.execute(
        [
            Instruction("vadd", VPUBinaryType(md=3, ms1=1, ms2=2)),
            Instruction("vrelu", VPUUnaryType(md=4, ms=3)),
        ]
    )

    assert perf.instructions == 2
    assert perf.cycles == 10
    assert perf.instructions_by_opcode == {"vadd": 1, "vrelu": 1}
