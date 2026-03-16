"""Command-line entry point for running Penguin programs and bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .assembler import assemble_file
from .arch_state import ArchState, StopReason
from .bundle import (
    ExecutableBundle,
    LoadedExecutableBundle,
    load_executable_bundle,
    load_mapped_program,
    preload_loaded_bundle_symbols,
    program_symbol_table_path,
)
from .core import PenguinCore
from .core_config import PenguinCoreConfig


def _parse_int(value: str) -> int:
    return int(value, 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--program",
        type=Path,
        help="Assembly source to execute. Uses the adjacent *.symbols.json5 sidecar if present.",
    )
    source_group.add_argument(
        "--bundle",
        type=Path,
        help="Executable bundle directory containing program.S, manifest.json5, and sidecars.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        help="Optional Perfetto-compatible JSON trace output path.",
    )
    parser.add_argument(
        "--base-address",
        type=_parse_int,
        help="IMEM base for plain --program assembly sources without a sidecar.",
    )
    parser.add_argument(
        "--start-pc",
        type=_parse_int,
        help="Optional start PC override.",
    )
    parser.add_argument(
        "--max-instructions",
        type=int,
        help="Optional step limit for execution.",
    )
    parser.add_argument(
        "--mem-base",
        type=_parse_int,
        default=0,
        help="Initial MEM_BASE CSR value to install before execution.",
    )
    return parser


def _load_program_and_state(
    args: argparse.Namespace,
    *,
    config: PenguinCoreConfig,
) -> tuple[ArchState, object, LoadedExecutableBundle | None]:
    state = ArchState.from_config(config)
    state.write_mem_base(args.mem_base)

    if args.bundle is not None:
        loaded = load_executable_bundle(ExecutableBundle.from_directory(args.bundle))
        preload_loaded_bundle_symbols(state, loaded)
        return state, loaded.program, loaded

    assert args.program is not None
    sidecar_path = program_symbol_table_path(args.program)
    if sidecar_path.exists():
        return state, load_mapped_program(args.program), None

    base_address = (
        config.memory_map.imem.base if args.base_address is None else args.base_address
    )
    return state, assemble_file(args.program, base_address=base_address), None


def _print_summary(
    *,
    source_label: str,
    core: PenguinCore,
    perf,
    trace_path: Path | None,
    loaded_bundle: LoadedExecutableBundle | None,
) -> None:
    print(f"source: {source_label}")
    print(f"stop_reason: {core.state.stop_reason}")
    print(f"pc: 0x{core.state.pc:08x}")
    print(f"instructions: {perf.instructions}")
    print(f"cycles: {perf.cycles}")
    print(f"bytes_read: {perf.bytes_read}")
    print(f"bytes_written: {perf.bytes_written}")
    print(f"instructions_by_opcode: {perf.instructions_by_opcode}")
    if trace_path is not None:
        print(f"trace: {trace_path}")
    if loaded_bundle is not None and loaded_bundle.symbol_data:
        print(f"preloaded_symbols: {sorted(loaded_bundle.symbol_data)}")
    if loaded_bundle is not None and loaded_bundle.constants:
        # TODO: constants.bin is bundle collateral today, but the manifest still lacks a
        # runtime mapping that would let the model CLI stage it into DRAM or VMEM.
        print(
            f"constants_blob: {len(loaded_bundle.constants)} bytes "
            "(present but not memory-mapped by the current bundle contract)"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    config = PenguinCoreConfig()
    state, program, loaded_bundle = _load_program_and_state(args, config=config)
    core = PenguinCore(state=state, config=config)

    if args.trace is None:
        perf = core.execute(
            program,
            start_pc=args.start_pc,
            max_instructions=args.max_instructions,
        )
    else:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        perf = core.dump_json_trace(
            program,
            args.trace,
            start_pc=args.start_pc,
            max_instructions=args.max_instructions,
        )

    source_label = str(args.bundle if args.bundle is not None else args.program)
    _print_summary(
        source_label=source_label,
        core=core,
        perf=perf,
        trace_path=args.trace,
        loaded_bundle=loaded_bundle,
    )
    if core.state.stop_reason not in {
        StopReason.PROGRAM_END,
        StopReason.ECALL,
        StopReason.EBREAK,
    }:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
