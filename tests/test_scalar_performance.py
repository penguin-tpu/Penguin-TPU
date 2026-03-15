"""Performance-oriented scalar regression tests."""

from __future__ import annotations

from penguin_model import DMAType, EmptyType, IType, Instruction
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


def test_scalar_benchmark_dma_stage_and_reduce_perf() -> None:
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    program = [
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
        program,
        dram_words={0x100 + index * 4: value for index, value in enumerate(values)},
    )

    assert core.state.vmem.load_u32(VMEM_BASE + 0x280) == sum(values)
    assert perf.instructions == 66
    assert perf.cycles == 74
    assert perf.bytes_read == 64
    assert perf.bytes_written == 36


def test_scalar_benchmark_address_generation_perf() -> None:
    program = _build_address_generation_program(
        base_addr=VMEM_BASE + 0x400,
        out_addr=VMEM_BASE + 0x500,
        rows=2,
        cols=3,
        row_stride=0x20,
    )

    core, perf = run_scalar_program(program)

    assert core.state.vmem.load_u32(VMEM_BASE + 0x500) == VMEM_BASE + 0x400
    assert perf.instructions == 60
    assert perf.cycles == 60
    assert perf.bytes_read == 0
    assert perf.bytes_written == 24


def test_scalar_benchmark_dma_overlap_hides_transfer_latency() -> None:
    program = [
        Instruction("saddi", IType(rd=1, rs1=0, imm=DRAM_BASE + 0x100)),
        Instruction("saddi", IType(rd=2, rs1=0, imm=VMEM_BASE + 0x100)),
        Instruction("saddi", IType(rd=3, rs1=0, imm=16)),
        Instruction("dma.load.ch0", DMAType(dram_rs=1, vmem_rs=2, size_rs=3)),
        Instruction("saddi", IType(rd=20, rs1=20, imm=1)),
        Instruction("saddi", IType(rd=20, rs1=20, imm=1)),
        Instruction("saddi", IType(rd=20, rs1=20, imm=1)),
        Instruction("saddi", IType(rd=20, rs1=20, imm=1)),
        Instruction("saddi", IType(rd=20, rs1=20, imm=1)),
        Instruction("saddi", IType(rd=20, rs1=20, imm=1)),
        Instruction("saddi", IType(rd=20, rs1=20, imm=1)),
        Instruction("saddi", IType(rd=20, rs1=20, imm=1)),
        Instruction("saddi", IType(rd=20, rs1=20, imm=1)),
        Instruction("dma.wait.ch0", EmptyType()),
    ]

    core, perf = run_scalar_program(
        program,
        dram_words={0x100 + index * 4: value for index, value in enumerate((9, 8, 7, 6))},
    )

    assert [core.state.vmem.load_u32(VMEM_BASE + 0x100 + index * 4) for index in range(4)] == [
        9,
        8,
        7,
        6,
    ]
    assert core.state.read_xreg(20) == 9
    assert perf.instructions == 14
    assert perf.cycles == 14
    assert perf.bytes_read == 16
    assert perf.bytes_written == 16
