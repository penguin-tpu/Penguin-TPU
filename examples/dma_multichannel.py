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

    streams = [
        {
            "channel": 0,
            "kind": "xreg",
            "size": 32,
            "dram_address": config.memory_map.dram.base + 0x100,
            "vmem_address": config.memory_map.vmem.base + 0x000,
            "roundtrip_vmem_address": config.memory_map.vmem.base + 0x080,
            "dram_output_address": config.memory_map.dram.base + 0x7000,
            "payload": _pattern(0x10, 32),
        },
        {
            "channel": 1,
            "kind": "mreg",
            "size": 2 * config.mreg_bytes,
            "dram_address": config.memory_map.dram.base + 0x1000,
            "vmem_address": config.memory_map.vmem.base + 0x1000,
            "roundtrip_vmem_address": config.memory_map.vmem.base + 0x2000,
            "dram_output_address": config.memory_map.dram.base + 0x8000,
            "payload": _pattern(0x40, 2 * config.mreg_bytes),
        },
        {
            "channel": 2,
            "kind": "mreg",
            "size": 3 * config.mreg_bytes,
            "dram_address": config.memory_map.dram.base + 0x4000,
            "vmem_address": config.memory_map.vmem.base + 0x4000,
            "roundtrip_vmem_address": config.memory_map.vmem.base + 0x8000,
            "dram_output_address": config.memory_map.dram.base + 0xB000,
            "payload": _pattern(0x90, 3 * config.mreg_bytes),
        },
    ]

    for stream in streams:
        state.dram.write(stream["dram_address"], stream["payload"])

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
    for stream in streams:
        staged = state.vmem.read(stream["vmem_address"], stream["size"])
        roundtrip = state.vmem.read(stream["roundtrip_vmem_address"], stream["size"])
        drained = state.dram.read(stream["dram_output_address"], stream["size"])
        matches = (
            torch.equal(staged, stream["payload"])
            and torch.equal(roundtrip, stream["payload"])
            and torch.equal(drained, stream["payload"])
        )
        all_match = all_match and matches
        print(
            f"  ch{stream['channel']} ({stream['kind']}): {stream['size']} bytes "
            f"dram=0x{stream['dram_address']:08X} -> vmem=0x{stream['vmem_address']:08X} "
            f"match={matches}"
        )
        print(
            f"    roundtrip_vmem=0x{stream['roundtrip_vmem_address']:08X} "
            f"dram_out=0x{stream['dram_output_address']:08X} "
            f"first_word=0x{_first_word(stream['payload']):08X}"
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
