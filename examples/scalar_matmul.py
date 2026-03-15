"""Runnable example for the Penguin scalar integer functional model."""

from __future__ import annotations

from penguin_model import (
    ArchState,
    BType,
    DMAType,
    DRAM_BASE,
    EmptyType,
    IType,
    Instruction,
    PenguinCore,
    RType,
    SType,
    VMEM_BASE,
)


def main() -> int:
    state = ArchState.with_memory_sizes()
    for index, value in enumerate((3, 5, 7, 11)):
        state.dram.store_u32(DRAM_BASE + 0x100 + index * 4, value)

    core = PenguinCore(state=state)
    program = [
        Instruction("saddi", IType(rd=10, rs1=0, imm=DRAM_BASE + 0x100)),
        Instruction("saddi", IType(rd=11, rs1=0, imm=VMEM_BASE + 0x040)),
        Instruction("saddi", IType(rd=12, rs1=0, imm=16)),
        Instruction("dma.load.ch0", DMAType(dram_rs=10, vmem_rs=11, size_rs=12)),
        Instruction("saddi", IType(rd=1, rs1=0, imm=VMEM_BASE + 0x40)),
        Instruction("saddi", IType(rd=2, rs1=0, imm=4)),
        Instruction("saddi", IType(rd=3, rs1=0, imm=0)),
        Instruction("dma.wait.ch0", EmptyType()),
        Instruction("sld", IType(rd=4, rs1=1, imm=0)),
        Instruction("sadd", RType(rd=3, rs1=3, rs2=4)),
        Instruction("saddi", IType(rd=1, rs1=1, imm=4)),
        Instruction("saddi", IType(rd=2, rs1=2, imm=-1)),
        Instruction("sbne", BType(rs1=2, rs2=0, imm=-16)),
        Instruction("sst", SType(rs1=0, rs2=3, imm=VMEM_BASE + 0x80)),
    ]
    trace_path = "scalar_trace.json"
    perf = core.dump_json_trace(program, trace_path)

    print("Input words in DRAM:")
    print([state.dram.load_u32(DRAM_BASE + 0x100 + index * 4) for index in range(4)])
    print()
    print("Staged words in VMEM:")
    print([state.vmem.load_u32(VMEM_BASE + 0x40 + index * 4) for index in range(4)])
    print()
    print(f"Stored sum in VMEM: {state.vmem.load_u32(VMEM_BASE + 0x80)}")
    print(f"Stop reason: {core.state.stop_reason}")
    print("Performance counters:")
    print(f"  instructions: {perf.instructions}")
    print(f"  cycles: {perf.cycles}")
    print(f"  bytes_read: {perf.bytes_read}")
    print(f"  bytes_written: {perf.bytes_written}")
    print(f"  instructions_by_opcode: {perf.instructions_by_opcode}")
    print(f"  trace: {trace_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
