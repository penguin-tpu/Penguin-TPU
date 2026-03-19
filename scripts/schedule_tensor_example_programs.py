"""Rewrite checked-in tensor example programs with compiler-inserted `delay` instructions."""

from __future__ import annotations

from pathlib import Path

from penguin_compiler import BundleSymbolTable, schedule_assembly_text
from penguin_model import DEFAULT_PENGUIN_CORE_CONFIG, IMEM_BASE, program_symbol_table_path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOT = REPO_ROOT / "tests" / "vectors" / "programs" / "tensor" / "examples"


def _program_base_address(program_path: Path) -> int:
    symbol_path = program_symbol_table_path(program_path)
    if not symbol_path.exists():
        return IMEM_BASE
    table = BundleSymbolTable.read_json5(symbol_path)
    program_symbol = table.symbols.get("program")
    if program_symbol is None:
        return IMEM_BASE
    return program_symbol.address


def main() -> int:
    for program_path in sorted(PROGRAM_ROOT.glob("*.S")):
        scheduled = schedule_assembly_text(
            program_path.read_text(),
            config=DEFAULT_PENGUIN_CORE_CONFIG,
            base_address=_program_base_address(program_path),
            source_name=str(program_path),
        )
        program_path.write_text(scheduled)
        print(f"scheduled {program_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
