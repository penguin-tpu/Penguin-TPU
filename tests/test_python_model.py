"""Tests for the Penguin scalar integer functional model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from penguin_model import (
    DMA_CHANNEL_COUNT,
    DRAM_BASE,
    DRAM_SIZE,
    INSTRUCTION_LATENCY,
    INSTRUCTION_SPECS,
    IMEM_BASE,
    IMEM_SIZE,
    ArchState,
    BType,
    EmptyType,
    IType,
    Memory,
    PenguinCore,
    RType,
    StopReason,
    UType,
    VMEM_BASE,
    VMEM_SIZE,
    saddi,
    sbeq,
    sbge,
    sbgeu,
    sblt,
    sbltu,
    sbne,
    sld,
)
from penguin_model.testbench import load_scalar_program


EXPECTED_BASE_MNEMONICS = {
    "slui",
    "sauipc",
    "sjal",
    "sjalr",
    "sbeq",
    "sbne",
    "sblt",
    "sbge",
    "sbltu",
    "sbgeu",
    "sld",
    "sst",
    "saddi",
    "sslti",
    "ssltiu",
    "sxori",
    "sori",
    "sandi",
    "sslli",
    "ssrli",
    "ssrai",
    "sadd",
    "ssub",
    "ssll",
    "sslt",
    "ssltu",
    "sxor",
    "ssrl",
    "ssra",
    "sor",
    "sand",
    "sfence",
    "secall",
    "sebreak",
}

EXPECTED_DMA_MNEMONICS = {
    *(f"dma.load.ch{channel}" for channel in range(DMA_CHANNEL_COUNT)),
    *(f"dma.store.ch{channel}" for channel in range(DMA_CHANNEL_COUNT)),
    *(f"dma.wait.ch{channel}" for channel in range(DMA_CHANNEL_COUNT)),
}

EXPECTED_MNEMONICS = EXPECTED_BASE_MNEMONICS | EXPECTED_DMA_MNEMONICS

TEST_DRAM_SIZE = 4 * 1024
TEST_VMEM_SIZE = 4 * 1024
TEST_IMEM_SIZE = 1 * 1024


def _fresh_state() -> ArchState:
    return ArchState.with_memory_sizes(
        dram_size=TEST_DRAM_SIZE,
        vmem_size=TEST_VMEM_SIZE,
        imem_size=TEST_IMEM_SIZE,
    )


def _store_bytes(memory: Memory, address: int, data: list[int]) -> None:
    memory.write(address, torch.tensor(data, dtype=torch.uint8))


def _program(name: str):
    return load_scalar_program(f"scalar/model/{name}.S")


def test_registered_scalar_subset_matches_spec_and_has_one_cycle_latency() -> None:
    assert set(INSTRUCTION_SPECS) == EXPECTED_MNEMONICS
    assert {mnemonic: INSTRUCTION_LATENCY[mnemonic] for mnemonic in EXPECTED_MNEMONICS} == {
        mnemonic: 1 for mnemonic in EXPECTED_MNEMONICS
    }


def test_memory_regions_are_independent_and_little_endian() -> None:
    state = _fresh_state()

    state.vmem.store_u32(VMEM_BASE + 0x10, 0x1122_3344)
    state.dram.store_u32(DRAM_BASE + 0x10, 0xAABB_CCDD)

    assert state.vmem.base == VMEM_BASE
    assert state.imem.base == IMEM_BASE
    assert state.dram.base == DRAM_BASE
    assert state.vmem.load_u32(VMEM_BASE + 0x10) == 0x1122_3344
    assert state.dram.load_u32(DRAM_BASE + 0x10) == 0xAABB_CCDD
    assert state.vmem.read(VMEM_BASE + 0x10, 4).tolist() == [0x44, 0x33, 0x22, 0x11]


def test_default_memory_layout_matches_spec() -> None:
    state = ArchState.with_memory_sizes()

    assert state.imem.base == IMEM_BASE
    assert state.imem.size == IMEM_SIZE
    assert state.vmem.base == VMEM_BASE
    assert state.vmem.size == VMEM_SIZE
    assert state.dram.base == DRAM_BASE
    assert state.dram.size == DRAM_SIZE
    assert state.dram.paged is True
    assert len(state.dma_channels) == DMA_CHANNEL_COUNT


def test_instruction_semantics_are_stateless_functions_over_state_and_params() -> None:
    state = _fresh_state()
    state.vmem.store_u32(VMEM_BASE + 0x08, 0xABCD_EF01)

    sld(state, IType(rd=5, rs1=0, imm=VMEM_BASE + 0x08))

    assert state.read_xreg(5) == 0xABCD_EF01


def test_instruction_decorator_registers_mnemonic_metadata() -> None:
    assert sld.mnemonic == "sld"
    assert INSTRUCTION_SPECS["sld"].semantics is sld
    assert INSTRUCTION_SPECS["sld"].params_type is IType


@pytest.mark.parametrize(
    ("mnemonic", "params", "expected"),
    [
        pytest.param("slui", UType(rd=10, imm=0x12345), 0x1234_5000, id="slui"),
        pytest.param("sauipc", UType(rd=10, imm=0x2), 0x0000_2020, id="sauipc"),
        pytest.param("saddi", IType(rd=10, rs1=1, imm=-1), 2, id="saddi"),
        pytest.param("sslti", IType(rd=10, rs1=2, imm=-1), 1, id="sslti"),
        pytest.param("ssltiu", IType(rd=10, rs1=2, imm=-1), 1, id="ssltiu"),
        pytest.param("sxori", IType(rd=10, rs1=1, imm=0xF), 0x0000_000C, id="sxori"),
        pytest.param("sori", IType(rd=10, rs1=1, imm=0x8), 0x0000_000B, id="sori"),
        pytest.param("sandi", IType(rd=10, rs1=2, imm=0xF), 0, id="sandi"),
        pytest.param("sslli", IType(rd=10, rs1=1, imm=1), 6, id="sslli"),
        pytest.param("ssrli", IType(rd=10, rs1=2, imm=4), 0x0FFF_FFFF, id="ssrli"),
        pytest.param("ssrai", IType(rd=10, rs1=2, imm=4), 0xFFFF_FFFF, id="ssrai"),
        pytest.param("sadd", RType(rd=10, rs1=1, rs2=3), 5, id="sadd"),
        pytest.param("ssub", RType(rd=10, rs1=1, rs2=3), 1, id="ssub"),
        pytest.param("ssll", RType(rd=10, rs1=1, rs2=3), 12, id="ssll"),
        pytest.param("sslt", RType(rd=10, rs1=2, rs2=1), 1, id="sslt"),
        pytest.param("ssltu", RType(rd=10, rs1=1, rs2=2), 1, id="ssltu"),
        pytest.param("sxor", RType(rd=10, rs1=1, rs2=2), 0xFFFF_FFF3, id="sxor"),
        pytest.param("ssrl", RType(rd=10, rs1=2, rs2=3), 0x3FFF_FFFC, id="ssrl"),
        pytest.param("ssra", RType(rd=10, rs1=2, rs2=3), 0xFFFF_FFFC, id="ssra"),
        pytest.param("sor", RType(rd=10, rs1=1, rs2=3), 3, id="sor"),
        pytest.param("sand", RType(rd=10, rs1=1, rs2=3), 2, id="sand"),
    ],
)
def test_integer_alu_semantics(mnemonic: str, params: object, expected: int) -> None:
    state = _fresh_state()
    state.pc = 0x20
    state.write_xreg(1, 3)
    state.write_xreg(2, 0xFFFF_FFF0)
    state.write_xreg(3, 2)

    INSTRUCTION_SPECS[mnemonic].semantics(state, params)

    assert state.read_xreg(10) == expected
    assert state.stop_reason is None


def test_x0_hardwired_to_zero() -> None:
    state = _fresh_state()
    state.write_xreg(1, 3)

    saddi(state, IType(rd=0, rs1=1, imm=99))

    assert state.read_xreg(0) == 0


def test_sfence_is_noop() -> None:
    state = _fresh_state()

    INSTRUCTION_SPECS["sfence"].semantics(state, EmptyType())

    assert state.stop_reason is None


@pytest.mark.parametrize(
    ("semantic", "lhs", "rhs", "taken"),
    [
        (sbeq, 5, 5, True),
        (sbne, 5, 6, True),
        (sblt, -1, 1, True),
        (sbge, 7, -3, True),
        (sbltu, 1, 2, True),
        (sbgeu, -1, 2, True),
        (sbeq, 5, 6, False),
        (sbne, 5, 5, False),
        (sblt, 4, -1, False),
        (sbge, -4, 2, False),
        (sbltu, -1, 2, False),
        (sbgeu, 1, -1, False),
    ],
)
def test_branch_semantics_set_next_pc_only_when_taken(
    semantic: object, lhs: int, rhs: int, taken: bool
) -> None:
    state = _fresh_state()
    state.pc = 0x40
    state.write_xreg(1, lhs)
    state.write_xreg(2, rhs)

    semantic(state, BType(rs1=1, rs2=2, imm=12))

    assert state.next_pc == (0x4C if taken else None)
    assert state.delay_slots_remaining == (2 if taken else 0)
    assert state.control_transfer_set is taken
    assert state.stop_reason is None


def test_core_executes_sjal_with_two_delay_slots_and_link_register() -> None:
    core = PenguinCore(state=_fresh_state())

    perf = core.execute(_program("core_sjal_delay_slots"))

    assert core.state.read_xreg(1) == 11
    assert core.state.read_xreg(2) == 22
    assert core.state.read_xreg(3) == 0
    assert core.state.read_xreg(4) == 4
    assert core.state.read_xreg(5) == 55
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 4
    assert perf.cycles == 4


def test_core_executes_sjalr_with_two_delay_slots_and_clears_lsb() -> None:
    core = PenguinCore(state=_fresh_state())

    perf = core.execute(_program("core_sjalr_delay_slots"))

    assert core.state.read_xreg(2) == 2
    assert core.state.read_xreg(3) == 3
    assert core.state.read_xreg(4) == 0
    assert core.state.read_xreg(5) == 8
    assert core.state.read_xreg(8) == 8
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 5
    assert perf.cycles == 5


def test_younger_control_transfer_in_delay_slot_replaces_older_redirect() -> None:
    core = PenguinCore(state=_fresh_state())

    perf = core.execute(_program("younger_control_transfer"))

    assert core.state.read_xreg(1) == 4
    assert core.state.read_xreg(2) == 8
    assert core.state.read_xreg(3) == 3
    assert core.state.read_xreg(4) == 4
    assert core.state.read_xreg(5) == 0
    assert core.state.read_xreg(6) == 0
    assert core.state.read_xreg(7) == 0
    assert core.state.read_xreg(8) == 8
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 5
    assert perf.cycles == 5


@pytest.mark.parametrize(
    ("mnemonic", "params", "expected"),
    [
        pytest.param("sslli", IType(rd=10, rs1=1, imm=37), 0x0000_0060, id="sslli-mask"),
        pytest.param("ssrli", IType(rd=10, rs1=2, imm=36), 0x0FFF_FFFF, id="ssrli-mask"),
        pytest.param("ssrai", IType(rd=10, rs1=2, imm=36), 0xFFFF_FFFF, id="ssrai-mask"),
        pytest.param("ssll", RType(rd=10, rs1=1, rs2=4), 0x0000_000C, id="ssll-mask"),
        pytest.param("ssrl", RType(rd=10, rs1=2, rs2=4), 0x3FFF_FFFC, id="ssrl-mask"),
        pytest.param("ssra", RType(rd=10, rs1=2, rs2=4), 0xFFFF_FFFC, id="ssra-mask"),
    ],
)
def test_shift_instructions_mask_shift_amount_to_low_five_bits(
    mnemonic: str, params: object, expected: int
) -> None:
    state = _fresh_state()
    state.write_xreg(1, 3)
    state.write_xreg(2, 0xFFFF_FFF0)
    state.write_xreg(4, 34)

    INSTRUCTION_SPECS[mnemonic].semantics(state, params)

    assert state.read_xreg(10) == expected


def test_scalar_load_store_and_branch_loop_program_uses_vmem_only() -> None:
    state = _fresh_state()
    for index, value in enumerate((3, 5, 7, 11)):
        state.vmem.store_u32(VMEM_BASE + 0x40 + index * 4, value)
        state.dram.store_u32(DRAM_BASE + 0x40 + index * 4, 0xDEAD_0000 + index)

    core = PenguinCore(state=state)
    perf = core.execute(_program("vmem_sum_loop"))

    assert state.vmem.load_u32(VMEM_BASE + 0x80) == 26
    assert state.dram.load_u32(DRAM_BASE + 0x80) == 0
    assert core.state.read_xreg(3) == 26
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 32
    assert perf.cycles == 32
    assert perf.bytes_read == 16
    assert perf.bytes_written == 4


def test_dma_load_wait_moves_bytes_from_dram_to_vmem() -> None:
    state = _fresh_state()
    _store_bytes(state.dram, DRAM_BASE + 0x100, [0x10, 0x20, 0x30, 0x40, 0x50, 0x60])

    core = PenguinCore(state=state)
    perf = core.execute(_program("dma_load_wait"))

    assert state.vmem.read(VMEM_BASE + 0x80, 6).tolist() == [0x10, 0x20, 0x30, 0x40, 0x50, 0x60]
    assert state.dram.read(DRAM_BASE + 0x100, 6).tolist() == [0x10, 0x20, 0x30, 0x40, 0x50, 0x60]
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 5
    assert perf.cycles == 13
    assert perf.bytes_read == 6
    assert perf.bytes_written == 6


def test_dma_store_wait_moves_bytes_from_vmem_to_dram() -> None:
    state = _fresh_state()
    _store_bytes(state.vmem, VMEM_BASE + 0x40, [1, 2, 3, 4, 5, 6, 7, 8])

    core = PenguinCore(state=state)
    perf = core.execute(_program("dma_store_wait"))

    assert state.dram.read(DRAM_BASE + 0x180, 8).tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 5
    assert perf.cycles == 13
    assert perf.bytes_read == 8
    assert perf.bytes_written == 8


def test_dma_load_requires_wait_before_vmem_sees_data() -> None:
    state = _fresh_state()
    state.dram.store_u32(DRAM_BASE + 0x100, 0xDEAD_BEEF)

    core = PenguinCore(state=state)
    perf = core.execute(_program("dma_requires_wait"))

    assert core.state.read_xreg(4) == 0
    assert core.state.read_xreg(5) == 0xDEAD_BEEF
    assert perf.instructions == 7
    assert perf.cycles == 14
    assert perf.bytes_read == 12
    assert perf.bytes_written == 4


def test_salu_progresses_while_dma_is_in_flight() -> None:
    state = _fresh_state()
    state.dram.store_u32(DRAM_BASE + 0x100, 0xCAFE_BABE)

    core = PenguinCore(state=state)
    perf = core.execute(_program("salu_progress_while_dma"))

    assert core.state.read_xreg(6) == 10
    assert core.state.read_xreg(7) == 0xCAFE_BABE
    assert perf.instructions == 16
    assert perf.cycles == 16


def test_dma_channel_busy_stops_execution() -> None:
    core = PenguinCore(state=_fresh_state())
    perf = core.execute(_program("dma_channel_busy"))

    assert core.state.stop_reason == StopReason.DMA_CHANNEL_BUSY
    assert perf.instructions == 5


def test_dma_wait_without_pending_transfer_is_one_cycle_noop() -> None:
    core = PenguinCore(state=_fresh_state())

    perf = core.execute(_program("dma_wait_noop"))

    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 1
    assert perf.cycles == 1
    assert perf.bytes_read == 0
    assert perf.bytes_written == 0


def test_dma_channels_operate_independently() -> None:
    state = _fresh_state()
    _store_bytes(state.dram, DRAM_BASE + 0x100, [1, 2, 3, 4])
    _store_bytes(state.dram, DRAM_BASE + 0x120, [5, 6, 7, 8])

    core = PenguinCore(state=state)
    perf = core.execute(_program("dma_channels_independent"))

    assert state.vmem.read(VMEM_BASE + 0x40, 4).tolist() == [1, 2, 3, 4]
    assert state.vmem.read(VMEM_BASE + 0x80, 4).tolist() == [5, 6, 7, 8]
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 9
    assert perf.cycles == 17
    assert perf.bytes_read == 8
    assert perf.bytes_written == 8


def test_misaligned_load_and_store_stop_execution() -> None:
    core = PenguinCore(state=_fresh_state())

    perf = core.execute(_program("misaligned_load"))

    assert core.state.stop_reason == StopReason.MISALIGNED_LOAD
    assert perf.instructions == 2
    assert perf.bytes_read == 0

    core.reset()
    perf = core.execute(_program("misaligned_store"))

    assert core.state.stop_reason == StopReason.MISALIGNED_STORE
    assert perf.instructions == 3
    assert perf.bytes_written == 0


def test_misaligned_jump_target_stops_execution() -> None:
    core = PenguinCore(state=_fresh_state())

    perf = core.execute(_program("misaligned_jump_target"))

    assert core.state.stop_reason == StopReason.INSTRUCTION_ADDRESS_MISALIGNED
    assert perf.instructions == 1


def test_taken_branch_with_misaligned_target_stops_before_delay_slots_execute() -> None:
    core = PenguinCore(state=_fresh_state())

    perf = core.execute(_program("misaligned_branch_target"))

    assert core.state.stop_reason == StopReason.INSTRUCTION_ADDRESS_MISALIGNED
    assert core.state.read_xreg(2) == 0
    assert core.state.read_xreg(3) == 0
    assert perf.instructions == 2


def test_reset_clears_architectural_state_and_dma_but_preserves_memory() -> None:
    state = _fresh_state()
    state.vmem.store_u32(VMEM_BASE + 0x20, 0x1234_5678)
    state.dram.store_u32(DRAM_BASE + 0x100, 0xCAFE_BABE)
    core = PenguinCore(state=state)

    perf = core.execute(_program("reset_dma_inflight"))

    assert perf.instructions == 4
    assert core.state.dma_channels[0].busy is True

    core.reset()

    assert core.state.pc == 0
    assert core.state.perf.instructions == 0
    assert core.state.perf.cycles == 0
    assert core.state.stop_reason is None
    assert core.state.read_xreg(1) == 0
    assert all(channel.busy is False for channel in core.state.dma_channels)
    assert core.state.vmem.load_u32(VMEM_BASE + 0x20) == 0x1234_5678
    assert core.state.dram.load_u32(DRAM_BASE + 0x100) == 0xCAFE_BABE


@pytest.mark.parametrize(
    ("mnemonic", "expected_reason"),
    [
        ("secall", StopReason.ECALL),
        ("sebreak", StopReason.EBREAK),
    ],
)
def test_environment_instructions_stop_with_explicit_status(
    mnemonic: str, expected_reason: StopReason
) -> None:
    core = PenguinCore(state=_fresh_state())

    perf = core.execute(_program(f"env_{mnemonic}"))

    assert core.state.stop_reason == expected_reason
    assert perf.instructions == 1
    assert perf.cycles == 1


def test_step_limit_stops_infinite_loop() -> None:
    core = PenguinCore(state=_fresh_state())

    perf = core.execute(_program("step_limit_loop"), max_instructions=6)

    assert core.state.stop_reason == StopReason.STEP_LIMIT
    assert perf.instructions == 6
    assert perf.cycles == 6


def test_dump_json_trace_emits_region_aware_trace(tmp_path: Path) -> None:
    state = _fresh_state()
    state.dram.store_u32(DRAM_BASE + 0x20, 7)
    core = PenguinCore(state=state)
    trace_path = tmp_path / "trace.json"

    perf = core.dump_json_trace(_program("trace_dma_flow"), trace_path)

    events = json.loads(trace_path.read_text())
    dma_transfer_event = next(
        event
        for event in events
        if event.get("cat") == "transfer" and "dma.load.ch0" in event["name"]
    )
    wait_fetch_event = next(
        event
        for event in events
        if event.get("cat") == "fetch" and "dma.wait.ch0" in event["name"]
    )

    assert perf.instructions == 7
    assert state.vmem.load_u32(VMEM_BASE + 0x80) == 7
    assert state.vmem.load_u32(VMEM_BASE + 0x90) == 7
    assert wait_fetch_event["ts"] < dma_transfer_event["ts"] + dma_transfer_event["dur"]
    assert any(
        event.get("name") == "thread_name"
        and event.get("pid") == 0
        and event.get("tid") == 0
        and event["args"]["name"] == "IFU"
        for event in events
    )
    assert any(
        event.get("name") == "thread_name"
        and event.get("pid") == 0
        and event.get("tid") == 1
        and event["args"]["name"] == "IDU"
        for event in events
    )
    assert any(
        event.get("name") == "thread_name"
        and event.get("pid") == 0
        and event.get("tid") == 2
        and event["args"]["name"] == "EXU.SALU"
        for event in events
    )
    assert any(
        event.get("name") == "thread_name"
        and event.get("pid") == 0
        and event.get("tid") == 3
        and event["args"]["name"] == "EXU.DMA"
        for event in events
    )
    assert any(
        event.get("cat") == "fetch" and "dma.load.ch0" in event["name"] and event.get("tid") == 0
        for event in events
    )
    assert any(
        event.get("cat") == "dispatch"
        and "dma.load.ch0" in event["name"]
        and event.get("tid") == 1
        for event in events
    )
    assert any(
        event.get("cat") == "execute"
        and "dma.load.ch0" in event["name"]
        and event.get("tid") == 3
        for event in events
    )
    assert any(
        event.get("cat") == "transfer"
        and "dma.load.ch0" in event["name"]
        and event.get("tid") == 3
        for event in events
    )
    assert any(
        event.get("cat") == "memory"
        and event["args"]["region"] == "dram"
        and event["name"] == "dma-read"
        for event in events
    )
    assert any(
        event.get("cat") == "memory"
        and event["args"]["region"] == "vmem"
        and event["name"] == "dma-write"
        for event in events
    )
    assert any(
        event.get("cat") == "memory"
        and event["args"]["region"] == "vmem"
        and event["name"] == "load"
        for event in events
    )
    assert any(
        event.get("pid") == 1 and event.get("ph") == "C" and event["name"] == "pc"
        for event in events
    )
