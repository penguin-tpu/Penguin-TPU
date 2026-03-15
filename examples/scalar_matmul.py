"""Runnable example for the Penguin scalar integer functional model."""

from __future__ import annotations

from pathlib import Path

from penguin_model import ArchState, DRAM_BASE, PenguinCore, VMEM_BASE, assemble_file


def main() -> int:
    state = ArchState.with_memory_sizes()
    for index, value in enumerate((3, 5, 7, 11)):
        state.dram.store_u32(DRAM_BASE + 0x100 + index * 4, value)

    program_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "vectors"
        / "programs"
        / "scalar"
        / "examples"
        / "scalar_matmul.S"
    )
    core = PenguinCore(state=state)
    trace_path = "scalar_trace.json"
    perf = core.dump_json_trace(assemble_file(program_path), trace_path)

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
