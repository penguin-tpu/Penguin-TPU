"""Run a multi-channel DMA example with differently sized transfers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from penguin_model import PenguinCoreConfig, Sim, load_mapped_program


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("outputs/examples/dma_multichannel_trace.json"),
        help="Path to the Perfetto-compatible JSON trace output.",
    )
    return parser


def _pattern(start: int, size: int) -> torch.Tensor:
    return torch.arange(start, start + size, dtype=torch.int64).to(torch.uint8)


def _first_word(data: torch.Tensor) -> int:
    return int.from_bytes(bytes(data[:4].tolist()), byteorder="little", signed=False)


def main() -> int:
    args = _build_parser().parse_args()
    config = PenguinCoreConfig()
    core = Sim(config=config)
    state = core.state

    load_transfers = [
        {
            "channel": 0,
            "size": 96,
            "dram_address": config.memory_map.dram.base + 0x100,
            "vmem_address": config.memory_map.vmem.base + 0x040,
            "marker_address": config.memory_map.vmem.base + 0x308,
            "payload": _pattern(0x10, 96),
        },
        {
            "channel": 1,
            "size": 32,
            "dram_address": config.memory_map.dram.base + 0x500,
            "vmem_address": config.memory_map.vmem.base + 0x100,
            "marker_address": config.memory_map.vmem.base + 0x300,
            "payload": _pattern(0x90, 32),
        },
        {
            "channel": 2,
            "size": 64,
            "dram_address": config.memory_map.dram.base + 0x300,
            "vmem_address": config.memory_map.vmem.base + 0x200,
            "marker_address": config.memory_map.vmem.base + 0x304,
            "payload": _pattern(0xC0, 64),
        },
    ]
    store_transfers = [
        {
            "channel": 1,
            "size": 32,
            "vmem_address": config.memory_map.vmem.base + 0x180,
            "dram_address": config.memory_map.dram.base + 0x400,
            "payload": _pattern(0x30, 32),
        },
        {
            "channel": 2,
            "size": 64,
            "vmem_address": config.memory_map.vmem.base + 0x280,
            "dram_address": config.memory_map.dram.base + 0x600,
            "payload": _pattern(0xE0, 64),
        },
    ]

    for transfer in load_transfers:
        state.dram.write(transfer["dram_address"], transfer["payload"])
    for transfer in store_transfers:
        state.vmem.write(transfer["vmem_address"], transfer["payload"])

    program_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "vectors"
        / "programs"
        / "scalar"
        / "examples"
        / "dma_multichannel_demo.S"
    )
    perf = core.dump_json_trace(load_mapped_program(program_path), args.trace)

    all_match = True
    print("Multi-channel DMA example completed.")
    print("  loads:")
    for transfer in load_transfers:
        staged = state.vmem.read(transfer["vmem_address"], transfer["size"])
        marker = state.vmem.load_u32(transfer["marker_address"])
        expected_marker = _first_word(transfer["payload"])
        matches = torch.equal(staged, transfer["payload"]) and marker == expected_marker
        all_match = all_match and matches
        print(
            f"  ch{transfer['channel']}: {transfer['size']} bytes "
            f"dram=0x{transfer['dram_address']:08X} -> vmem=0x{transfer['vmem_address']:08X} "
            f"match={matches}"
        )
        print(
            f"    marker@0x{transfer['marker_address']:08X}=0x{marker:08X} "
            f"(expected 0x{expected_marker:08X})"
        )
    print("  stores:")
    for transfer in store_transfers:
        drained = state.dram.read(transfer["dram_address"], transfer["size"])
        matches = torch.equal(drained, transfer["payload"])
        all_match = all_match and matches
        print(
            f"  ch{transfer['channel']}: {transfer['size']} bytes "
            f"vmem=0x{transfer['vmem_address']:08X} -> dram=0x{transfer['dram_address']:08X} "
            f"match={matches}"
        )

    print(f"  stop_reason: {core.state.stop_reason}")
    print(f"  instructions: {perf.instructions}")
    print(f"  cycles: {perf.cycles}")
    print(f"  bytes_read: {perf.bytes_read}")
    print(f"  bytes_written: {perf.bytes_written}")
    print(f"  trace: {args.trace}")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
