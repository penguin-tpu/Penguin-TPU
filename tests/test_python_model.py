"""Tests for the Penguin scalar integer functional model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from trace_utils import trace_output_path

from penguin_model import (
    DMA_ALIGNMENT_BYTES,
    DMA_CHANNEL_COUNT,
    DMAType,
    DEFAULT_PENGUIN_CORE_CONFIG,
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
    Instruction,
    InitializationConfig,
    Memory,
    MREG_BYTES,
    Sim,
    PenguinCoreConfig,
    RType,
    SType,
    StopReason,
    TensorMemType,
    UType,
    VMEM_BASE,
    VMEM_SIZE,
    addi,
    beq,
    bge,
    bgeu,
    blt,
    bltu,
    bne,
    lw,
)
from penguin_model.testbench import TEST_CORE_CONFIG, load_scalar_program


EXPECTED_BASE_MNEMONICS = {
    "lui",
    "auipc",
    "jal",
    "jalr",
    "beq",
    "bne",
    "blt",
    "bge",
    "bltu",
    "bgeu",
    "lb",
    "lbu",
    "lh",
    "lhu",
    "lw",
    "seli",
    "seld",
    "sb",
    "sh",
    "sw",
    "addi",
    "slti",
    "sltiu",
    "xori",
    "ori",
    "andi",
    "slli",
    "srli",
    "srai",
    "add",
    "sub",
    "sll",
    "slt",
    "sltu",
    "xor",
    "srl",
    "sra",
    "or",
    "and",
    "fence",
    "ecall",
    "ebreak",
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
    return ArchState.from_config(TEST_CORE_CONFIG)


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
    config = PenguinCoreConfig()
    state = ArchState.from_config(config)

    assert state.imem.base == IMEM_BASE
    assert state.imem.size == IMEM_SIZE
    assert state.vmem.base == VMEM_BASE
    assert state.vmem.size == VMEM_SIZE
    assert state.dram.base == DRAM_BASE
    assert state.dram.size == DRAM_SIZE
    assert state.dram.paged is True
    assert len(state.dma_channels) == DMA_CHANNEL_COUNT
    assert state.config == DEFAULT_PENGUIN_CORE_CONFIG


def test_power_on_state_uses_deterministic_pseudo_random_contents() -> None:
    state_a = _fresh_state()
    state_b = _fresh_state()

    assert state_a.read_xreg(0) == 0
    assert state_b.read_xreg(0) == 0
    assert state_a.read_xreg(1) == state_b.read_xreg(1)
    assert state_a.vmem.load_u32(VMEM_BASE + 0x00) == state_b.vmem.load_u32(VMEM_BASE + 0x00)
    assert state_a.dram.load_u32(DRAM_BASE + 0x00) == state_b.dram.load_u32(DRAM_BASE + 0x00)
    assert bool(state_a.load_mreg(0).any().item())


def test_power_on_randomization_can_be_disabled_in_config() -> None:
    config = PenguinCoreConfig(
        initialization=InitializationConfig(
            seed=DEFAULT_PENGUIN_CORE_CONFIG.initialization.seed,
            randomize_dram=False,
            randomize_vmem=False,
            randomize_scalar_registers=False,
            randomize_tensor_registers=False,
            randomize_weight_slots=False,
        )
    )
    state = ArchState.from_config(config)

    assert state.read_xreg(1) == 0
    assert state.vmem.load_u32(VMEM_BASE + 0x00) == 0
    assert state.dram.load_u32(DRAM_BASE + 0x00) == 0
    assert not bool(state.load_mreg(0).any().item())


def test_instruction_semantics_are_stateless_functions_over_state_and_params() -> None:
    state = _fresh_state()
    state.vmem.store_u32(VMEM_BASE + 0x08, 0xABCD_EF01)

    lw(state, IType(rd=5, rs1=0, imm=VMEM_BASE + 0x08))

    assert state.read_xreg(5) == 0xABCD_EF01


def test_instruction_decorator_registers_mnemonic_metadata() -> None:
    assert lw.mnemonic == "lw"
    assert INSTRUCTION_SPECS["lw"].semantics is lw
    assert INSTRUCTION_SPECS["lw"].params_type is IType


@pytest.mark.parametrize(
    ("mnemonic", "bytes_", "expected"),
    [
        pytest.param("lb", [0x80], 0xFFFF_FF80, id="lb-sign-extend"),
        pytest.param("lbu", [0x80], 0x0000_0080, id="lbu-zero-extend"),
        pytest.param("lh", [0x34, 0xF2], 0xFFFF_F234, id="lh-sign-extend"),
        pytest.param("lhu", [0x34, 0xF2], 0x0000_F234, id="lhu-zero-extend"),
        pytest.param("lw", [0x78, 0x56, 0x34, 0xF2], 0xF234_5678, id="lw-word"),
    ],
)
def test_scalar_load_variants_follow_rv32i_sign_and_zero_extension(
    mnemonic: str, bytes_: list[int], expected: int
) -> None:
    state = _fresh_state()
    address = VMEM_BASE + 0x20
    _store_bytes(state.vmem, address, bytes_)

    INSTRUCTION_SPECS[mnemonic].semantics(state, IType(rd=7, rs1=0, imm=address))

    assert state.read_xreg(7) == expected
    assert state.stop_reason is None
    assert state.perf.bytes_read == len(bytes_)


@pytest.mark.parametrize(
    ("mnemonic", "value", "expected_bytes"),
    [
        pytest.param("sb", 0x1234_5678, [0x78], id="sb"),
        pytest.param("sh", 0x1234_5678, [0x78, 0x56], id="sh"),
        pytest.param("sw", 0x1234_5678, [0x78, 0x56, 0x34, 0x12], id="sw"),
    ],
)
def test_scalar_store_variants_write_low_order_bytes_little_endian(
    mnemonic: str, value: int, expected_bytes: list[int]
) -> None:
    state = _fresh_state()
    address = VMEM_BASE + 0x40
    state.write_xreg(3, address)
    state.write_xreg(4, value)

    INSTRUCTION_SPECS[mnemonic].semantics(state, SType(rs1=3, rs2=4, imm=0))

    assert state.vmem.read(address, len(expected_bytes)).tolist() == expected_bytes
    assert state.stop_reason is None
    assert state.perf.bytes_written == len(expected_bytes)


@pytest.mark.parametrize(
    ("mnemonic", "params", "expected"),
    [
        pytest.param("lui", UType(rd=10, imm=0x12345), 0x1234_5000, id="lui"),
        pytest.param("auipc", UType(rd=10, imm=0x2), 0x0000_2020, id="auipc"),
        pytest.param("addi", IType(rd=10, rs1=1, imm=-1), 2, id="addi"),
        pytest.param("slti", IType(rd=10, rs1=2, imm=-1), 1, id="slti"),
        pytest.param("sltiu", IType(rd=10, rs1=2, imm=-1), 1, id="sltiu"),
        pytest.param("xori", IType(rd=10, rs1=1, imm=0xF), 0x0000_000C, id="xori"),
        pytest.param("ori", IType(rd=10, rs1=1, imm=0x8), 0x0000_000B, id="ori"),
        pytest.param("andi", IType(rd=10, rs1=2, imm=0xF), 0, id="andi"),
        pytest.param("slli", IType(rd=10, rs1=1, imm=1), 6, id="slli"),
        pytest.param("srli", IType(rd=10, rs1=2, imm=4), 0x0FFF_FFFF, id="srli"),
        pytest.param("srai", IType(rd=10, rs1=2, imm=4), 0xFFFF_FFFF, id="srai"),
        pytest.param("add", RType(rd=10, rs1=1, rs2=3), 5, id="add"),
        pytest.param("sub", RType(rd=10, rs1=1, rs2=3), 1, id="sub"),
        pytest.param("sll", RType(rd=10, rs1=1, rs2=3), 12, id="sll"),
        pytest.param("slt", RType(rd=10, rs1=2, rs2=1), 1, id="slt"),
        pytest.param("sltu", RType(rd=10, rs1=1, rs2=2), 1, id="sltu"),
        pytest.param("xor", RType(rd=10, rs1=1, rs2=2), 0xFFFF_FFF3, id="xor"),
        pytest.param("srl", RType(rd=10, rs1=2, rs2=3), 0x3FFF_FFFC, id="srl"),
        pytest.param("sra", RType(rd=10, rs1=2, rs2=3), 0xFFFF_FFFC, id="sra"),
        pytest.param("or", RType(rd=10, rs1=1, rs2=3), 3, id="or"),
        pytest.param("and", RType(rd=10, rs1=1, rs2=3), 2, id="and"),
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

    addi(state, IType(rd=0, rs1=1, imm=99))

    assert state.read_xreg(0) == 0


def test_sfence_is_noop() -> None:
    state = _fresh_state()

    INSTRUCTION_SPECS["fence"].semantics(state, EmptyType())

    assert state.stop_reason is None


@pytest.mark.parametrize(
    ("semantic", "lhs", "rhs", "taken"),
    [
        (beq, 5, 5, True),
        (bne, 5, 6, True),
        (blt, -1, 1, True),
        (bge, 7, -3, True),
        (bltu, 1, 2, True),
        (bgeu, -1, 2, True),
        (beq, 5, 6, False),
        (bne, 5, 5, False),
        (blt, 4, -1, False),
        (bge, -4, 2, False),
        (bltu, -1, 2, False),
        (bgeu, 1, -1, False),
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


def test_core_executes_jal_with_two_delay_slots_and_link_register() -> None:
    core = Sim(state=_fresh_state(), config=TEST_CORE_CONFIG)
    initial_x3 = core.state.read_xreg(3)

    perf = core.execute(_program("core_sjal_delay_slots"))

    assert core.state.read_xreg(1) == 11
    assert core.state.read_xreg(2) == 22
    assert core.state.read_xreg(3) == initial_x3
    assert core.state.read_xreg(4) == IMEM_BASE + 1
    assert core.state.read_xreg(5) == 55
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 4
    assert perf.cycles == 7


def test_core_executes_jalr_with_two_delay_slots() -> None:
    core = Sim(state=_fresh_state(), config=TEST_CORE_CONFIG)
    initial_x4 = core.state.read_xreg(4)

    perf = core.execute(_program("core_sjalr_delay_slots"))

    assert core.state.read_xreg(2) == 2
    assert core.state.read_xreg(3) == 3
    assert core.state.read_xreg(4) == initial_x4
    assert core.state.read_xreg(5) == IMEM_BASE + 2
    assert core.state.read_xreg(8) == 8
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 5
    assert perf.cycles == 8


def test_control_transfer_in_delay_slot_is_illegal() -> None:
    core = Sim(state=_fresh_state(), config=TEST_CORE_CONFIG)
    initial_x1 = core.state.read_xreg(1)
    initial_x2 = core.state.read_xreg(2)
    initial_x5 = core.state.read_xreg(5)
    initial_x6 = core.state.read_xreg(6)
    initial_x7 = core.state.read_xreg(7)

    perf = core.execute(_program("younger_control_transfer"))

    assert core.state.read_xreg(1) == initial_x1
    assert core.state.read_xreg(2) == initial_x2
    assert core.state.read_xreg(5) == initial_x5
    assert core.state.read_xreg(6) == initial_x6
    assert core.state.read_xreg(7) == initial_x7
    assert core.state.stop_reason == StopReason.ILLEGAL_INSTRUCTION
    assert perf.instructions == 0


@pytest.mark.parametrize(
    ("mnemonic", "params", "expected"),
    [
        pytest.param("slli", IType(rd=10, rs1=1, imm=37), 0x0000_0060, id="slli-mask"),
        pytest.param("srli", IType(rd=10, rs1=2, imm=36), 0x0FFF_FFFF, id="srli-mask"),
        pytest.param("srai", IType(rd=10, rs1=2, imm=36), 0xFFFF_FFFF, id="srai-mask"),
        pytest.param("sll", RType(rd=10, rs1=1, rs2=4), 0x0000_000C, id="sll-mask"),
        pytest.param("srl", RType(rd=10, rs1=2, rs2=4), 0x3FFF_FFFC, id="srl-mask"),
        pytest.param("sra", RType(rd=10, rs1=2, rs2=4), 0xFFFF_FFFC, id="sra-mask"),
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

    initial_dram_word = state.dram.load_u32(DRAM_BASE + 0x80)
    core = Sim(state=state, config=state.config)
    perf = core.execute(_program("vmem_sum_loop"))

    assert state.vmem.load_u32(VMEM_BASE + 0x80) == 26
    assert state.dram.load_u32(DRAM_BASE + 0x80) == initial_dram_word
    assert core.state.read_xreg(3) == 26
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 32
    assert perf.cycles == 35
    assert perf.bytes_read == 16
    assert perf.bytes_written == 4


def test_dma_load_wait_moves_bytes_from_dram_to_vmem() -> None:
    state = _fresh_state()
    _store_bytes(state.dram, DRAM_BASE + 0x100, list(range(0x10, 0x30)))

    core = Sim(state=state, config=state.config)
    perf = core.execute(_program("dma_load_wait"))

    assert state.vmem.read(VMEM_BASE + 0x80, 32).tolist() == list(range(0x10, 0x30))
    assert state.dram.read(DRAM_BASE + 0x100, 32).tolist() == list(range(0x10, 0x30))
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 5
    assert perf.cycles == 26
    assert perf.bytes_read == 32
    assert perf.bytes_written == 32


def test_dma_store_wait_moves_bytes_from_vmem_to_dram() -> None:
    state = _fresh_state()
    _store_bytes(state.vmem, VMEM_BASE + 0x40, list(range(1, 33)))

    core = Sim(state=state, config=state.config)
    perf = core.execute(_program("dma_store_wait"))

    assert state.dram.read(DRAM_BASE + 0x180, 32).tolist() == list(range(1, 33))
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 5
    assert perf.cycles == 26
    assert perf.bytes_read == 32
    assert perf.bytes_written == 32


def test_dma_load_requires_wait_before_vmem_sees_data() -> None:
    state = _fresh_state()
    state.dram.store_u32(DRAM_BASE + 0x100, 0xDEAD_BEEF)
    initial_vmem_word = state.vmem.load_u32(VMEM_BASE + 0x20)

    core = Sim(state=state, config=state.config)
    perf = core.execute(_program("dma_requires_wait"))

    assert core.state.read_xreg(4) == initial_vmem_word
    assert core.state.read_xreg(5) == 0xDEAD_BEEF
    assert perf.instructions == 7
    assert perf.cycles == 28
    assert perf.bytes_read == 40
    assert perf.bytes_written == 32


def test_salu_progresses_while_dma_is_in_flight() -> None:
    state = _fresh_state()
    state.dram.store_u32(DRAM_BASE + 0x100, 0xCAFE_BABE)

    core = Sim(state=state, config=state.config)
    perf = core.execute(_program("salu_progress_while_dma"))

    assert core.state.read_xreg(6) == 10
    assert core.state.read_xreg(7) == 0xCAFE_BABE
    assert perf.instructions == 16
    assert perf.cycles == 28


def test_dma_channel_busy_stops_execution() -> None:
    core = Sim(state=_fresh_state(), config=TEST_CORE_CONFIG)
    perf = core.execute(_program("dma_channel_busy"))

    assert core.state.stop_reason == StopReason.DMA_CHANNEL_BUSY
    assert perf.instructions == 4


def test_dma_wait_without_pending_transfer_is_one_cycle_noop() -> None:
    core = Sim(state=_fresh_state(), config=TEST_CORE_CONFIG)

    perf = core.execute(_program("dma_wait_noop"))

    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 1
    assert perf.cycles == 3
    assert perf.bytes_read == 0
    assert perf.bytes_written == 0


def test_dma_channels_operate_independently() -> None:
    state = _fresh_state()
    _store_bytes(state.dram, DRAM_BASE + 0x100, list(range(1, 33)))
    _store_bytes(state.dram, DRAM_BASE + 0x120, list(range(33, 65)))

    core = Sim(state=state, config=state.config)
    perf = core.execute(_program("dma_channels_independent"))

    assert state.vmem.read(VMEM_BASE + 0x40, 32).tolist() == list(range(1, 33))
    assert state.vmem.read(VMEM_BASE + 0x80, 32).tolist() == list(range(33, 65))
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 9
    assert perf.cycles == 30
    assert perf.bytes_read == 64
    assert perf.bytes_written == 64


def test_dma_store_channels_operate_independently() -> None:
    state = _fresh_state()
    payload_a = list(range(0x10, 0x30))
    payload_b = list(range(0x80, 0xA0))
    _store_bytes(state.vmem, VMEM_BASE + 0x40, payload_a)
    _store_bytes(state.vmem, VMEM_BASE + 0x80, payload_b)

    core = Sim(state=state, config=state.config)
    perf = core.execute(
        [
            Instruction("addi", IType(rd=1, rs1=0, imm=DRAM_BASE + 0x100)),
            Instruction("addi", IType(rd=2, rs1=0, imm=VMEM_BASE + 0x40)),
            Instruction("addi", IType(rd=3, rs1=0, imm=DMA_ALIGNMENT_BYTES)),
            Instruction("dma.store.ch0", DMAType(dram_rs=1, vmem_rs=2, size_rs=3)),
            Instruction("addi", IType(rd=4, rs1=0, imm=DRAM_BASE + 0x120)),
            Instruction("addi", IType(rd=5, rs1=0, imm=VMEM_BASE + 0x80)),
            Instruction("dma.store.ch1", DMAType(dram_rs=4, vmem_rs=5, size_rs=3)),
            Instruction("dma.wait.ch1", EmptyType()),
            Instruction("dma.wait.ch0", EmptyType()),
        ]
    )

    assert state.dram.read(DRAM_BASE + 0x100, 32).tolist() == payload_a
    assert state.dram.read(DRAM_BASE + 0x120, 32).tolist() == payload_b
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 9
    assert perf.cycles == 30
    assert perf.bytes_read == 64
    assert perf.bytes_written == 64


def test_dma_store_captures_vmem_payload_at_issue_time() -> None:
    state = _fresh_state()
    original = list(range(1, 33))
    _store_bytes(state.vmem, VMEM_BASE + 0x40, original)

    core = Sim(state=state, config=state.config)
    perf = core.execute(
        [
            Instruction("addi", IType(rd=1, rs1=0, imm=DRAM_BASE + 0x100)),
            Instruction("addi", IType(rd=2, rs1=0, imm=VMEM_BASE + 0x40)),
            Instruction("addi", IType(rd=3, rs1=0, imm=DMA_ALIGNMENT_BYTES)),
            Instruction("dma.store.ch0", DMAType(dram_rs=1, vmem_rs=2, size_rs=3)),
            Instruction("addi", IType(rd=4, rs1=0, imm=VMEM_BASE + 0x40)),
            Instruction("addi", IType(rd=5, rs1=0, imm=0xA5A5_5A5A)),
            Instruction("sw", SType(rs1=4, rs2=5, imm=0)),
            Instruction("dma.wait.ch0", EmptyType()),
        ]
    )

    assert state.dram.read(DRAM_BASE + 0x100, 32).tolist() == original
    assert state.vmem.load_u32(VMEM_BASE + 0x40) == 0xA5A5_5A5A
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 8
    assert perf.cycles == 26
    assert perf.bytes_read == 32
    assert perf.bytes_written == 36


def test_dma_load_captures_dram_payload_at_issue_time_before_later_store() -> None:
    state = _fresh_state()
    original = list(range(0x20, 0x40))
    replacement = list(range(0x90, 0xB0))
    _store_bytes(state.dram, DRAM_BASE + 0x100, original)
    _store_bytes(state.vmem, VMEM_BASE + 0x140, replacement)

    core = Sim(state=state, config=state.config)
    perf = core.execute(
        [
            Instruction("addi", IType(rd=1, rs1=0, imm=DRAM_BASE + 0x100)),
            Instruction("addi", IType(rd=2, rs1=0, imm=VMEM_BASE + 0x80)),
            Instruction("addi", IType(rd=3, rs1=0, imm=DMA_ALIGNMENT_BYTES)),
            Instruction("dma.load.ch0", DMAType(dram_rs=1, vmem_rs=2, size_rs=3)),
            Instruction("addi", IType(rd=4, rs1=0, imm=DRAM_BASE + 0x100)),
            Instruction("addi", IType(rd=5, rs1=0, imm=VMEM_BASE + 0x140)),
            Instruction("dma.store.ch1", DMAType(dram_rs=4, vmem_rs=5, size_rs=3)),
            Instruction("dma.wait.ch1", EmptyType()),
            Instruction("dma.wait.ch0", EmptyType()),
        ]
    )

    assert state.vmem.read(VMEM_BASE + 0x80, 32).tolist() == original
    assert state.dram.read(DRAM_BASE + 0x100, 32).tolist() == replacement
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == 9
    assert perf.cycles == 30
    assert perf.bytes_read == 64
    assert perf.bytes_written == 64


def test_dma_load_with_misaligned_address_stops_execution() -> None:
    state = _fresh_state()
    state.write_xreg(1, DRAM_BASE + DMA_ALIGNMENT_BYTES)
    state.write_xreg(2, VMEM_BASE + 4)
    state.write_xreg(3, DMA_ALIGNMENT_BYTES)

    INSTRUCTION_SPECS["dma.load.ch0"].semantics(
        state,
        DMAType(dram_rs=1, vmem_rs=2, size_rs=3),
    )

    assert state.stop_reason == StopReason.DMA_MISALIGNED_ADDRESS
    assert state.dma_channels[0].busy is False


def test_dma_store_with_misaligned_address_stops_execution() -> None:
    state = _fresh_state()
    state.write_xreg(1, DRAM_BASE + 4)
    state.write_xreg(2, VMEM_BASE + DMA_ALIGNMENT_BYTES)
    state.write_xreg(3, DMA_ALIGNMENT_BYTES)

    INSTRUCTION_SPECS["dma.store.ch0"].semantics(
        state,
        DMAType(dram_rs=1, vmem_rs=2, size_rs=3),
    )

    assert state.stop_reason == StopReason.DMA_MISALIGNED_ADDRESS
    assert state.dma_channels[0].busy is False


def test_dma_load_with_misaligned_size_stops_execution() -> None:
    state = _fresh_state()
    state.write_xreg(1, DRAM_BASE + DMA_ALIGNMENT_BYTES)
    state.write_xreg(2, VMEM_BASE + DMA_ALIGNMENT_BYTES)
    state.write_xreg(3, DMA_ALIGNMENT_BYTES - 4)

    INSTRUCTION_SPECS["dma.load.ch0"].semantics(
        state,
        DMAType(dram_rs=1, vmem_rs=2, size_rs=3),
    )

    assert state.stop_reason == StopReason.DMA_MISALIGNED_SIZE
    assert state.dma_channels[0].busy is False


def test_dma_store_with_misaligned_size_stops_execution() -> None:
    state = _fresh_state()
    state.write_xreg(1, DRAM_BASE + DMA_ALIGNMENT_BYTES)
    state.write_xreg(2, VMEM_BASE + DMA_ALIGNMENT_BYTES)
    state.write_xreg(3, DMA_ALIGNMENT_BYTES - 4)

    INSTRUCTION_SPECS["dma.store.ch0"].semantics(
        state,
        DMAType(dram_rs=1, vmem_rs=2, size_rs=3),
    )

    assert state.stop_reason == StopReason.DMA_MISALIGNED_SIZE
    assert state.dma_channels[0].busy is False


def test_misaligned_load_and_store_stop_execution() -> None:
    core = Sim(state=_fresh_state(), config=TEST_CORE_CONFIG)

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
    core = Sim(state=_fresh_state(), config=TEST_CORE_CONFIG)

    perf = core.execute(_program("misaligned_jump_target"))

    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert core.state.pc == IMEM_BASE + 1
    assert perf.instructions == 1
    assert perf.cycles == 4


def test_legacy_byte_pc_branch_target_is_legal_under_word_indexed_pc() -> None:
    core = Sim(state=_fresh_state(), config=TEST_CORE_CONFIG)

    perf = core.execute(_program("misaligned_branch_target"))

    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert core.state.read_xreg(2) == 99
    assert core.state.read_xreg(3) == 77
    assert core.state.pc == IMEM_BASE + 6
    assert perf.instructions == 4
    assert perf.cycles == 7


def test_reset_clears_architectural_state_and_dma_but_preserves_memory() -> None:
    state = _fresh_state()
    state.vmem.store_u32(VMEM_BASE + 0x20, 0x1234_5678)
    state.dram.store_u32(DRAM_BASE + 0x100, 0xCAFE_BABE)
    core = Sim(state=state, config=state.config)

    perf = core.execute(_program("reset_dma_inflight"))

    assert perf.instructions == 4
    assert core.state.dma_channels[0].busy is True

    core.reset()
    reference_state = _fresh_state()

    assert core.state.pc == 0
    assert core.state.perf.instructions == 0
    assert core.state.perf.cycles == 0
    assert core.state.stop_reason is None
    assert core.state.read_xreg(0) == 0
    assert core.state.read_xreg(1) == reference_state.read_xreg(1)
    assert all(channel.busy is False for channel in core.state.dma_channels)
    assert core.state.vmem.load_u32(VMEM_BASE + 0x20) == 0x1234_5678
    assert core.state.dram.load_u32(DRAM_BASE + 0x100) == 0xCAFE_BABE


@pytest.mark.parametrize(
    ("mnemonic", "expected_reason"),
    [
        ("ecall", StopReason.ECALL),
        ("ebreak", StopReason.EBREAK),
    ],
)
def test_environment_instructions_stop_with_explicit_status(
    mnemonic: str, expected_reason: StopReason
) -> None:
    core = Sim(state=_fresh_state(), config=TEST_CORE_CONFIG)

    source_name = "env_secall" if mnemonic == "ecall" else "env_sebreak"
    perf = core.execute(_program(source_name))

    assert core.state.stop_reason == expected_reason
    assert perf.instructions == 1
    assert perf.cycles == 4


def test_step_limit_stops_infinite_loop() -> None:
    core = Sim(state=_fresh_state(), config=TEST_CORE_CONFIG)

    perf = core.execute(_program("step_limit_loop"), max_instructions=6)

    assert core.state.stop_reason == StopReason.STEP_LIMIT
    assert perf.instructions == 7
    assert perf.cycles == 10


def test_dump_json_trace_emits_region_aware_trace(tmp_path: Path) -> None:
    state = _fresh_state()
    state.dram.store_u32(DRAM_BASE + 0x20, 7)
    core = Sim(state=state, config=state.config)
    trace_path = trace_output_path("python_model_trace_dma_flow.json")

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
    wait_dispatch_event = next(
        event
        for event in events
        if event.get("cat") == "dispatch" and "dma.wait.ch0" in event["name"]
    )
    stalled_fetch_event = next(
        event
        for event in events
        if event.get("cat") == "fetch" and "lw x4, 0(x2)" in event["name"]
    )
    early_fetch_events = [
        event
        for event in events
        if event.get("cat") == "fetch" and int(event.get("ts", -1)) < int(wait_fetch_event["ts"])
    ]

    assert perf.instructions == 7
    assert state.vmem.load_u32(VMEM_BASE + 0x80) == 7
    assert state.vmem.load_u32(VMEM_BASE + 0x90) == 7
    early_fetch_timestamps = [int(event["ts"]) for event in early_fetch_events]
    assert early_fetch_timestamps == sorted(early_fetch_timestamps)
    assert early_fetch_timestamps == list(
        range(early_fetch_timestamps[0], early_fetch_timestamps[0] + len(early_fetch_timestamps))
    )
    assert all(int(event.get("dur", 0)) == 1 for event in early_fetch_events)
    assert wait_fetch_event["ts"] < dma_transfer_event["ts"] + dma_transfer_event["dur"]
    assert not any(
        event.get("cat") == "execute" and "dma.wait.ch0" in event["name"] for event in events
    )
    assert wait_dispatch_event["ts"] + wait_dispatch_event["dur"] <= (
        dma_transfer_event["ts"] + dma_transfer_event["dur"]
    )
    assert int(stalled_fetch_event["dur"]) == 1
    assert int(stalled_fetch_event["ts"]) == int(wait_dispatch_event["ts"])
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
        for event in events
    )
    assert any(
        event.get("name") == "thread_name"
        and event.get("pid") == 0
        and event.get("tid") == 30
        and event["args"]["name"] == "DMA.XFER.CH0"
        for event in events
    )
    assert any(
        event.get("cat") == "memory"
        and event["args"]["region"] == "dram"
        and event["args"]["access_type"] == "dma-read"
        and str(event["name"]).startswith("dma-read ")
        and "0x" in str(event["name"])
        for event in events
    )
    assert any(
        event.get("cat") == "memory"
        and event["args"]["region"] == "vmem"
        and event["args"]["access_type"] == "dma-write"
        and str(event["name"]).startswith("dma-write ")
        and "0x" in str(event["name"])
        for event in events
    )
    assert any(
        event.get("cat") == "memory"
        and event["args"]["region"] == "vmem"
        and event["name"] == "load"
        for event in events
    )
    pc_events = [
        event for event in events if event.get("pid") == 1 and event.get("ph") == "C" and event["name"] == "pc"
    ]
    assert pc_events
    assert pc_events[0]["args"]["value"] == 0
    cycle_events = [
        event
        for event in events
        if event.get("pid") == 1 and event.get("ph") == "C" and event["name"] == "cycle"
    ]
    assert cycle_events
    assert [event["args"]["value"] for event in cycle_events] == list(range(len(cycle_events)))
    assert cycle_events[-1]["args"]["value"] == max(
        int(event["ts"]) + int(event.get("dur", 0)) for event in events if "ts" in event
    )
    assert [event["ts"] for event in cycle_events] == [
        cycle * state.config.trace.ticks_per_cycle for cycle in range(len(cycle_events))
    ]
    timestamped_events = [int(event["ts"]) for event in events if "ts" in event]
    assert timestamped_events == sorted(timestamped_events)


def test_trace_wait_blocks_following_tensor_memory_op_until_later_dma_wait_retires(
    tmp_path: Path,
) -> None:
    state = _fresh_state()
    state.dram.write(DRAM_BASE + 0x000, torch.arange(0, 128, dtype=torch.uint8).repeat(16))
    state.dram.write(DRAM_BASE + 0x800, torch.arange(128, 256, dtype=torch.uint8).repeat(16))
    core = Sim(state=state, config=state.config)
    trace_path = trace_output_path("python_model_dma_wait_blocks_vload.json")

    perf = core.dump_json_trace(
        [
            Instruction("addi", IType(rd=1, rs1=0, imm=DRAM_BASE + 0x000)),
            Instruction("addi", IType(rd=2, rs1=0, imm=VMEM_BASE + 0x000)),
            Instruction("addi", IType(rd=3, rs1=0, imm=MREG_BYTES)),
            Instruction("dma.load.ch1", DMAType(dram_rs=1, vmem_rs=2, size_rs=3)),
            Instruction("addi", IType(rd=4, rs1=0, imm=DRAM_BASE + 0x800)),
            Instruction("addi", IType(rd=5, rs1=0, imm=VMEM_BASE + 0x800)),
            Instruction("dma.load.ch0", DMAType(dram_rs=4, vmem_rs=5, size_rs=3)),
            Instruction("dma.wait.ch1", EmptyType()),
            Instruction("dma.wait.ch0", EmptyType()),
            Instruction("vload", TensorMemType(mreg=1, rs1=5, imm=0)),
        ],
        trace_path,
    )

    events = json.loads(trace_path.read_text())
    wait0_dispatch = next(
        event
        for event in events
        if event.get("cat") == "dispatch" and "dma.wait.ch0" in event["name"]
    )
    vload_execute = next(
        event for event in events if event.get("cat") == "execute" and "vload" in event["name"]
    )

    assert perf.instructions == 10
    assert not any(
        event.get("cat") == "execute" and "dma.wait.ch0" in event["name"] for event in events
    )
    assert wait0_dispatch["ts"] + wait0_dispatch["dur"] <= vload_execute["ts"]
