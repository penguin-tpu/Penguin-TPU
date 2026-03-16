"""CLI helpers for packaging Penguin assembly into executable bundles.

Direct PyTorch-to-Penguin lowering remains a later milestone. The current CLI turns a
checked-in or user-authored assembly program plus optional symbol/payload metadata into
the executable-bundle contract consumed by `penguin-model`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from penguin_model import IMEM_BASE, assemble_file

from .bundle import (
    BundleManifest,
    BundleSymbol,
    BundleSymbolTable,
    write_executable_bundle,
)
from .rtl import write_verilog_rom_init


def _parse_int(value: str) -> int:
    return int(value, 0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle_parser = subparsers.add_parser(
        "bundle",
        help="Package assembly source plus sidecar metadata into an executable bundle.",
    )
    bundle_parser.add_argument("--program", type=Path, required=True, help="Assembly source path.")
    bundle_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination bundle directory.",
    )
    bundle_parser.add_argument(
        "--symbol-table",
        type=Path,
        help="Optional existing program.symbols.json5 sidecar to copy into the bundle.",
    )
    bundle_parser.add_argument(
        "--entry-symbol",
        default="program",
        help="Entry symbol name in the symbol table.",
    )
    bundle_parser.add_argument(
        "--default-program-address",
        type=_parse_int,
        default=IMEM_BASE,
        help="IMEM base address used when auto-generating a minimal symbol table.",
    )
    bundle_parser.add_argument(
        "--program-name",
        default="program.S",
        help="Program filename written inside the bundle directory.",
    )
    bundle_parser.add_argument(
        "--symbol-table-name",
        help="Override the symbol-table filename inside the bundle.",
    )
    bundle_parser.add_argument(
        "--manifest-name",
        default="manifest.json5",
        help="Manifest filename inside the bundle.",
    )
    bundle_parser.add_argument(
        "--constants",
        type=Path,
        help="Optional constants.bin payload to copy into the bundle.",
    )
    bundle_parser.add_argument(
        "--constants-name",
        default="constants.bin",
        help="Constants filename inside the bundle.",
    )
    bundle_parser.add_argument(
        "--symbol-file",
        action="append",
        default=[],
        metavar="RELATIVE_PATH=HOST_PATH",
        help="Additional file-backed bundle payload to copy into the bundle.",
    )

    rtl_rom_parser = subparsers.add_parser(
        "rtl-rom",
        help="Assemble a scalar program into a Verilog ROM-init include.",
    )
    rtl_rom_parser.add_argument("--program", type=Path, required=True, help="Assembly source path.")
    rtl_rom_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination Verilog include path.",
    )
    rtl_rom_parser.add_argument(
        "--array-name",
        default="imem",
        help="Verilog memory array identifier used on the left-hand side.",
    )
    rtl_rom_parser.add_argument(
        "--base-address",
        type=_parse_int,
        default=0,
        help="Program base address used when resolving labels.",
    )
    return parser


def _adjacent_symbol_table_path(program_path: Path) -> Path:
    return program_path.with_name(f"{program_path.stem}.symbols.json5")


def _parse_symbol_file(argument: str) -> tuple[str, bytes]:
    relative_path, separator, host_path = argument.partition("=")
    if not separator or not relative_path or not host_path:
        raise ValueError(
            f"Invalid --symbol-file value '{argument}'; expected RELATIVE_PATH=HOST_PATH"
        )
    return relative_path, Path(host_path).read_bytes()


def _auto_symbol_table(
    *,
    program_path: Path,
    entry_symbol: str,
    default_program_address: int,
) -> BundleSymbolTable:
    program = assemble_file(program_path, base_address=default_program_address)
    return BundleSymbolTable(
        symbols={
            entry_symbol: BundleSymbol(
                name=entry_symbol,
                kind="program",
                region="imem",
                address=default_program_address,
                size_bytes=len(program) * 4,
                file=program_path.name,
            )
        }
    )


def _load_symbol_table(
    *,
    program_path: Path,
    explicit_symbol_table: Path | None,
    entry_symbol: str,
    default_program_address: int,
) -> tuple[BundleSymbolTable, Path | None]:
    if explicit_symbol_table is not None:
        return BundleSymbolTable.read_json5(explicit_symbol_table), explicit_symbol_table.parent

    adjacent = _adjacent_symbol_table_path(program_path)
    if adjacent.exists():
        return BundleSymbolTable.read_json5(adjacent), adjacent.parent

    return (
        _auto_symbol_table(
            program_path=program_path,
            entry_symbol=entry_symbol,
            default_program_address=default_program_address,
        ),
        None,
    )


def _load_symbol_files_from_table(
    symbol_table: BundleSymbolTable,
    *,
    source_root: Path | None,
) -> dict[str, bytes]:
    if source_root is None:
        return {}

    payloads: dict[str, bytes] = {}
    for symbol in symbol_table.symbols.values():
        if symbol.file is None:
            continue
        path = source_root / symbol.file
        if path.exists():
            payloads[symbol.file] = path.read_bytes()
    return payloads


def _bundle_command(args: argparse.Namespace) -> int:
    symbol_table, symbol_table_root = _load_symbol_table(
        program_path=args.program,
        explicit_symbol_table=args.symbol_table,
        entry_symbol=args.entry_symbol,
        default_program_address=args.default_program_address,
    )
    manifest = BundleManifest(
        entry_symbol=args.entry_symbol,
        symbol_table=(
            args.symbol_table_name
            if args.symbol_table_name is not None
            else f"{Path(args.program_name).stem}.symbols.json5"
        ),
    )
    symbol_files = _load_symbol_files_from_table(symbol_table, source_root=symbol_table_root)
    symbol_files.update(dict(_parse_symbol_file(argument) for argument in args.symbol_file))
    constants = b"" if args.constants is None else args.constants.read_bytes()
    bundle = write_executable_bundle(
        args.output_dir,
        program_text=args.program.read_text(),
        symbol_table=symbol_table,
        manifest=manifest,
        constants=constants,
        symbol_files=symbol_files,
        program_name=args.program_name,
        symbol_table_name=args.symbol_table_name,
        manifest_name=args.manifest_name,
        constants_name=args.constants_name,
    )
    print(f"wrote bundle: {bundle.root}")
    print(f"  program: {bundle.program.name}")
    print(f"  symbol_table: {bundle.symbol_table.name}")
    print(f"  manifest: {bundle.manifest.name}")
    print(f"  constants: {bundle.constants.name}")
    return 0


def _rtl_rom_command(args: argparse.Namespace) -> int:
    output = write_verilog_rom_init(
        args.output,
        args.program,
        array_name=args.array_name,
        base_address=args.base_address,
    )
    print(f"wrote rtl rom init: {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "bundle":
        return _bundle_command(args)
    if args.command == "rtl-rom":
        return _rtl_rom_command(args)
    raise ValueError(f"Unsupported command '{args.command}'")


if __name__ == "__main__":
    raise SystemExit(main())
