"""Scalar workload-style tests for the Penguin functional model."""

from __future__ import annotations

from penguin_model import DMAType, EmptyType, IType, Instruction, RType, SType, StopReason
from penguin_model.testbench import DRAM_BASE, VMEM_BASE, ScalarProgramBuilder, run_scalar_program


def _li(builder: ScalarProgramBuilder, rd: int, value: int) -> None:
    builder.i("saddi", rd=rd, rs1=0, imm=value)


def _build_reduction_program(*, src_addr: int, out_addr: int, count: int) -> list[Instruction]:
    builder = ScalarProgramBuilder()
    _li(builder, 1, src_addr)
    _li(builder, 2, count)
    _li(builder, 3, 0)
    _li(builder, 5, out_addr)
    builder.label("loop")
    builder.i("sld", rd=4, rs1=1, imm=0)
    builder.r("sadd", rd=3, rs1=3, rs2=4)
    builder.i("saddi", rd=1, rs1=1, imm=4)
    builder.i("saddi", rd=2, rs1=2, imm=-1)
    builder.branch("sbne", rs1=2, rs2=0, target="loop")
    builder.delay_slots()
    builder.s("sst", rs1=5, rs2=3, imm=0)
    return builder.build()


def _build_address_generation_program(
    *, base_addr: int, out_addr: int, rows: int, cols: int, row_stride: int
) -> list[Instruction]:
    builder = ScalarProgramBuilder()
    _li(builder, 1, out_addr)
    _li(builder, 2, base_addr)
    _li(builder, 3, rows)
    _li(builder, 7, row_stride)
    builder.label("row_loop")
    _li(builder, 4, cols)
    builder.i("saddi", rd=5, rs1=2, imm=0)
    builder.label("col_loop")
    builder.s("sst", rs1=1, rs2=5, imm=0)
    builder.i("saddi", rd=1, rs1=1, imm=4)
    builder.i("saddi", rd=5, rs1=5, imm=4)
    builder.i("saddi", rd=4, rs1=4, imm=-1)
    builder.branch("sbne", rs1=4, rs2=0, target="col_loop")
    builder.delay_slots()
    builder.r("sadd", rd=2, rs1=2, rs2=7)
    builder.i("saddi", rd=3, rs1=3, imm=-1)
    builder.branch("sbne", rs1=3, rs2=0, target="row_loop")
    builder.delay_slots()
    return builder.build()


def _build_copy_and_checksum_program(
    *, src_addr: int, dst_addr: int, checksum_addr: int, count: int
) -> list[Instruction]:
    builder = ScalarProgramBuilder()
    _li(builder, 1, src_addr)
    _li(builder, 2, dst_addr)
    _li(builder, 3, count)
    _li(builder, 4, 0)
    _li(builder, 5, checksum_addr)
    builder.label("loop")
    builder.i("sld", rd=6, rs1=1, imm=0)
    builder.s("sst", rs1=2, rs2=6, imm=0)
    builder.r("sadd", rd=4, rs1=4, rs2=6)
    builder.i("saddi", rd=1, rs1=1, imm=4)
    builder.i("saddi", rd=2, rs1=2, imm=4)
    builder.i("saddi", rd=3, rs1=3, imm=-1)
    builder.branch("sbne", rs1=3, rs2=0, target="loop")
    builder.delay_slots()
    builder.s("sst", rs1=5, rs2=4, imm=0)
    return builder.build()


def test_scalar_workload_reduces_staged_tensor_tile() -> None:
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    stateful_program = [
        Instruction("saddi", IType(rd=10, rs1=0, imm=DRAM_BASE + 0x100)),
        Instruction("saddi", IType(rd=11, rs1=0, imm=VMEM_BASE + 0x200)),
        Instruction("saddi", IType(rd=12, rs1=0, imm=32)),
        Instruction("dma.load.ch0", DMAType(dram_rs=10, vmem_rs=11, size_rs=12)),
        Instruction("dma.wait.ch0", EmptyType()),
        *_build_reduction_program(
            src_addr=VMEM_BASE + 0x200,
            out_addr=VMEM_BASE + 0x280,
            count=len(values),
        ),
    ]

    core, perf = run_scalar_program(
        stateful_program,
        dram_words={0x100 + index * 4: value for index, value in enumerate(values)},
    )

    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert core.state.vmem.load_u32(VMEM_BASE + 0x280) == sum(values)
    assert [core.state.vmem.load_u32(VMEM_BASE + 0x200 + index * 4) for index in range(8)] == values
    assert perf.bytes_read == 64
    assert perf.bytes_written == 36


def test_scalar_workload_generates_row_major_tensor_addresses() -> None:
    program = _build_address_generation_program(
        base_addr=VMEM_BASE + 0x400,
        out_addr=VMEM_BASE + 0x500,
        rows=2,
        cols=3,
        row_stride=0x20,
    )

    core, perf = run_scalar_program(program)

    expected = [
        VMEM_BASE + 0x400,
        VMEM_BASE + 0x404,
        VMEM_BASE + 0x408,
        VMEM_BASE + 0x420,
        VMEM_BASE + 0x424,
        VMEM_BASE + 0x428,
    ]
    actual = [core.state.vmem.load_u32(VMEM_BASE + 0x500 + index * 4) for index in range(6)]

    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert actual == expected
    assert perf.bytes_written == 24


def test_scalar_workload_copies_tile_and_accumulates_checksum() -> None:
    values = [9, 4, 7, 1, 5, 3]
    program = _build_copy_and_checksum_program(
        src_addr=VMEM_BASE + 0x600,
        dst_addr=VMEM_BASE + 0x700,
        checksum_addr=VMEM_BASE + 0x780,
        count=len(values),
    )

    core, perf = run_scalar_program(
        program,
        vmem_words={0x600 + index * 4: value for index, value in enumerate(values)},
    )

    copied = [core.state.vmem.load_u32(VMEM_BASE + 0x700 + index * 4) for index in range(len(values))]

    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert copied == values
    assert core.state.vmem.load_u32(VMEM_BASE + 0x780) == sum(values)
    assert perf.bytes_read == len(values) * 4
    assert perf.bytes_written == len(values) * 4 + 4
