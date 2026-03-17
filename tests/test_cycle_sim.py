"""Per-cycle simulator tests for the tick-driven Penguin model."""

from __future__ import annotations

import torch

from penguin_model import DMAType, EmptyType, IType, Instruction, Sim, VPUBinaryType
from penguin_model.tensor import bf16_tile_from_bytes, bf16_tile_to_bytes
from penguin_model.testbench import DRAM_BASE, TEST_CORE_CONFIG, VMEM_BASE, fresh_arch_state


def _tile(values: list[list[float]]) -> torch.Tensor:
    rows = TEST_CORE_CONFIG.tensor.mreg_rows
    cols = TEST_CORE_CONFIG.tensor.mreg_row_bytes // 2
    tile = torch.zeros((rows, cols), dtype=torch.float32)
    value_tensor = torch.tensor(values, dtype=torch.float32)
    tile[: value_tensor.shape[0], : value_tensor.shape[1]] = value_tensor
    return tile


def test_tick_advances_scalar_execution_one_cycle_at_a_time() -> None:
    state = fresh_arch_state()
    core = Sim(state=state, config=state.config)
    initial_x2 = state.read_xreg(2)
    program = [
        Instruction("saddi", IType(rd=1, rs1=0, imm=7)),
        Instruction("saddi", IType(rd=2, rs1=1, imm=1)),
    ]

    core.load_program(program)

    assert core.tick() is True
    assert state.perf.cycles == 1
    assert state.perf.instructions == 0
    assert state.read_xreg(1) != 7
    assert state.read_xreg(2) == initial_x2
    assert state.pc == 4
    assert state.stop_reason is None

    assert core.tick() is True
    assert state.perf.cycles == 2
    assert state.perf.instructions == 0
    assert state.read_xreg(1) != 7
    assert state.read_xreg(2) == initial_x2
    assert state.pc == 8
    assert state.stop_reason is None

    assert core.tick() is True
    assert state.perf.cycles == 3
    assert state.perf.instructions == 0
    assert state.read_xreg(1) != 7
    assert state.read_xreg(2) == initial_x2
    assert state.pc == 8
    assert state.stop_reason is None

    assert core.tick() is True
    assert state.perf.cycles == 4
    assert state.perf.instructions == 1
    assert state.read_xreg(1) == 7
    assert state.read_xreg(2) == initial_x2
    assert state.pc == 8
    assert state.stop_reason is None

    assert core.tick() is False
    assert state.perf.cycles == 5
    assert state.perf.instructions == 2
    assert state.read_xreg(2) == 8
    assert state.pc == 8
    assert state.stop_reason is not None


def test_tick_models_dma_wait_as_a_per_channel_decode_fence() -> None:
    state = fresh_arch_state()
    core = Sim(state=state, config=state.config)
    size_ch0 = 96
    size_ch1 = 32
    payload_ch0 = torch.arange(size_ch0, dtype=torch.uint8)
    payload_ch1 = torch.arange(0x80, 0x80 + size_ch1, dtype=torch.uint8)

    state.write_xreg(1, DRAM_BASE + 0x100)
    state.write_xreg(2, VMEM_BASE + 0x100)
    state.write_xreg(3, size_ch0)
    state.write_xreg(4, DRAM_BASE + 0x300)
    state.write_xreg(5, VMEM_BASE + 0x300)
    state.write_xreg(6, size_ch1)
    state.dram.write(DRAM_BASE + 0x100, payload_ch0)
    state.dram.write(DRAM_BASE + 0x300, payload_ch1)

    program = [
        Instruction("dma.load.ch0", DMAType(dram_rs=1, vmem_rs=2, size_rs=3)),
        Instruction("dma.load.ch1", DMAType(dram_rs=4, vmem_rs=5, size_rs=6)),
        Instruction("dma.wait.ch1", EmptyType()),
        Instruction("saddi", IType(rd=10, rs1=0, imm=1)),
        Instruction("dma.wait.ch0", EmptyType()),
    ]

    core.load_program(program)

    for _ in range(4):
        assert core.tick() is True
    ready_ch1 = state.dma_channels[1].pending.ready_cycle
    ready_ch0 = state.dma_channels[0].pending.ready_cycle
    assert state.perf.cycles == 4
    assert state.perf.instructions == 1
    assert state.pc == 16
    assert state.dma_channels[0].busy is True
    assert state.dma_channels[1].busy is True

    assert core.tick() is True
    assert state.perf.cycles == 5
    assert state.perf.instructions == 2
    assert state.pc == 16
    assert state.dma_channels[1].busy is True
    assert ready_ch0 > ready_ch1

    while state.perf.cycles < ready_ch1 - 1:
        assert core.tick() is True
        assert state.pc == 16
        assert state.dma_channels[1].busy is True
        assert state.read_xreg(10) != 1

    assert core.tick() is True
    assert state.perf.cycles == ready_ch1
    assert state.perf.instructions == 3
    assert state.pc == 16
    assert state.dma_channels[1].busy is False
    assert state.dma_channels[0].busy is True
    assert torch.equal(state.vmem.read(VMEM_BASE + 0x300, size_ch1), payload_ch1)

    assert core.tick() is True
    assert state.perf.cycles == ready_ch1 + 1
    assert state.read_xreg(10) != 1
    assert state.pc == 20
    assert state.dma_channels[0].busy is True

    assert core.tick() is True
    assert state.perf.cycles == ready_ch1 + 2
    assert state.perf.instructions == 4
    assert state.read_xreg(10) == 1
    assert state.pc == 20
    assert state.dma_channels[0].busy is True

    while state.stop_reason is None:
        core.tick()

    assert state.perf.cycles == ready_ch0
    assert state.perf.instructions == 5
    assert state.dma_channels[0].busy is False
    assert torch.equal(state.vmem.read(VMEM_BASE + 0x100, size_ch0), payload_ch0)


def test_tick_defers_vpu_writeback_until_latency_boundary() -> None:
    state = fresh_arch_state()
    core = Sim(state=state, config=state.config)
    lhs = _tile([[1.0, -2.0], [3.5, 4.0]])
    rhs = _tile([[0.5, 8.0], [-1.5, 2.0]])
    state.store_mreg(1, bf16_tile_to_bytes(lhs, config=state.config))
    state.store_mreg(2, bf16_tile_to_bytes(rhs, config=state.config))
    initial_dest = state.load_mreg(3).clone()

    core.load_program([Instruction("vadd", VPUBinaryType(md=3, ms1=1, ms2=2))])

    assert core.tick() is True
    assert state.perf.cycles == 1
    assert state.perf.instructions == 0
    assert state.pc == 4
    assert state.stop_reason is None
    assert torch.equal(state.load_mreg(3), initial_dest)

    assert core.tick() is True
    assert state.perf.cycles == 2
    assert state.perf.instructions == 0
    assert torch.equal(state.load_mreg(3), initial_dest)

    assert core.tick() is True
    assert state.perf.cycles == 3
    assert state.perf.instructions == 0
    assert torch.equal(state.load_mreg(3), initial_dest)

    assert core.tick() is True
    assert state.perf.cycles == 4
    assert state.perf.instructions == 0
    assert torch.equal(state.load_mreg(3), initial_dest)

    assert core.tick() is False
    assert state.perf.cycles == 5
    assert state.perf.instructions == 1
    assert state.stop_reason is not None
    expected = torch.add(lhs, rhs).to(torch.bfloat16).to(torch.float32)
    observed = bf16_tile_from_bytes(state.load_mreg(3), config=state.config).to(torch.float32)
    assert torch.equal(observed, expected)
