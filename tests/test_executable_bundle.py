"""Tests for executable bundle symbol-table generation and loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from trace_utils import trace_output_path

from penguin_compiler import (
    BundleManifest as CompilerBundleManifest,
    BundleSymbol as CompilerBundleSymbol,
    BundleSymbolTable as CompilerBundleSymbolTable,
    write_executable_bundle,
)
from penguin_compiler.cli import main as compiler_main
from penguin_model import (
    ArchState,
    DRAM_BASE,
    IMEM_BASE,
    ExecutableBundle,
    Sim,
    StopReason,
    load_executable_bundle,
    load_mapped_program,
    load_program_symbol_table,
    preload_loaded_bundle_symbols,
    program_symbol_table_path,
)
from penguin_model.cli import main as model_main
from penguin_model.testbench import TEST_CORE_CONFIG

PROGRAM_ROOT = Path("tests/vectors/programs")


def test_bundle_loader_reads_symbol_table_and_file_backed_inputs(tmp_path: Path) -> None:
    bundle_dir = _write_sample_bundle(tmp_path)

    loaded = load_executable_bundle(ExecutableBundle.from_directory(bundle_dir))

    assert loaded.bundle.symbol_table.name == "program.symbols.json5"
    assert loaded.program.base_address == IMEM_BASE
    assert loaded.program.labels == {
        "start": IMEM_BASE,
        "target": IMEM_BASE + 2,
    }
    assert loaded.manifest.symbol_table == "program.symbols.json5"
    assert loaded.symbol_table.symbol("program").address == IMEM_BASE
    assert loaded.symbol_table.symbol("input").address == DRAM_BASE + 0x100
    assert loaded.symbol_table.symbol("output").address == DRAM_BASE + 0x200
    assert loaded.symbol_data == {"input": b"penguin-input"}
    assert loaded.constants == b"\xCA\xFE"

    symbol_table_text = loaded.bundle.symbol_table.read_text()
    # JSON5 writer emits bare identifier keys (no quotes), so check for the
    # unquoted address fields instead of JSON-style quoted keys.
    assert "address: 0x00020000" in symbol_table_text
    assert "address: 0x80000100" in symbol_table_text


def test_core_executes_bundle_program_from_mapped_imem_base(tmp_path: Path) -> None:
    bundle_dir = _write_sample_bundle(tmp_path)
    loaded = load_executable_bundle(ExecutableBundle.from_directory(bundle_dir))

    core = Sim()
    core.execute(loaded.program)

    assert core.state.read_xreg(1) == IMEM_BASE + 2
    assert core.state.stop_reason == StopReason.EBREAK


def test_bundle_preload_stages_file_backed_payloads_into_mapped_memory(tmp_path: Path) -> None:
    bundle_dir = _write_sample_bundle(tmp_path)
    loaded = load_executable_bundle(ExecutableBundle.from_directory(bundle_dir))
    state = ArchState.from_config(TEST_CORE_CONFIG)

    preload_loaded_bundle_symbols(state, loaded)

    assert bytes(state.dram.read(DRAM_BASE + 0x100, len(b"penguin-input")).tolist()) == b"penguin-input"


@pytest.mark.parametrize(
    "program_path",
    sorted(PROGRAM_ROOT.rglob("*.S")),
    ids=lambda path: path.relative_to(PROGRAM_ROOT).as_posix(),
)
def test_checked_in_programs_have_symbol_table_sidecars(program_path: Path) -> None:
    assert program_symbol_table_path(program_path).exists()

    symbol_table = load_program_symbol_table(program_path)
    program = load_mapped_program(program_path)
    program_symbol = symbol_table.symbol("program")

    assert program_symbol.address == IMEM_BASE
    assert program.base_address == program_symbol.address
    assert len(program) * 4 == program_symbol.size_bytes


def test_load_mapped_program_rejects_non_program_entry_symbol(tmp_path: Path) -> None:
    program_path = tmp_path / "non_program_entry.S"
    program_path.write_text("sebreak\n")
    CompilerBundleSymbolTable(
        symbols={
            "input": CompilerBundleSymbol(
                name="input",
                kind="input",
                region="dram",
                address=DRAM_BASE,
                size_bytes=4,
            )
        }
    ).write_json5(program_symbol_table_path(program_path))

    with pytest.raises(ValueError, match="must have kind 'program'"):
        load_mapped_program(program_path, entry_symbol="input")


def test_load_mapped_program_rejects_symbol_size_mismatch(tmp_path: Path) -> None:
    program_path = tmp_path / "size_mismatch.S"
    program_path.write_text("nop\nsebreak\n")
    CompilerBundleSymbolTable(
        symbols={
            "program": CompilerBundleSymbol(
                name="program",
                kind="program",
                region="imem",
                address=IMEM_BASE,
                size_bytes=4,
                file=program_path.name,
            )
        }
    ).write_json5(program_symbol_table_path(program_path))

    with pytest.raises(ValueError, match="size mismatch"):
        load_mapped_program(program_path)


def test_load_executable_bundle_rejects_symbol_payload_size_mismatch(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bad_payload_bundle"
    write_executable_bundle(
        bundle_dir,
        program_text="sebreak\n",
        symbol_table=CompilerBundleSymbolTable(
            symbols={
                "program": CompilerBundleSymbol(
                    name="program",
                    kind="program",
                    region="imem",
                    address=IMEM_BASE,
                    size_bytes=4,
                    file="program.S",
                ),
                "input": CompilerBundleSymbol(
                    name="input",
                    kind="input",
                    region="dram",
                    address=DRAM_BASE + 0x200,
                    size_bytes=8,
                    file="input.bin",
                ),
            }
        ),
        symbol_files={"input.bin": b"tiny"},
    )

    with pytest.raises(ValueError, match="payload size mismatch"):
        load_executable_bundle(ExecutableBundle.from_directory(bundle_dir))


def test_compiler_cli_writes_bundle_with_auto_generated_symbol_table(tmp_path: Path) -> None:
    program_path = tmp_path / "sample.S"
    bundle_dir = tmp_path / "bundle_out"
    program_path.write_text("start:\n    nop\n    sebreak\n")

    assert (
        compiler_main(
            [
                "bundle",
                "--program",
                str(program_path),
                "--output-dir",
                str(bundle_dir),
            ]
        )
        == 0
    )

    loaded = load_executable_bundle(ExecutableBundle.from_directory(bundle_dir))
    assert loaded.program.base_address == IMEM_BASE
    assert loaded.symbol_table.symbol("program").size_bytes == 8


def test_compiler_cli_copies_adjacent_sidecar_payloads_into_bundle(tmp_path: Path) -> None:
    program_path = tmp_path / "program.S"
    program_path.write_text("sebreak\n")
    CompilerBundleSymbolTable(
        symbols={
            "program": CompilerBundleSymbol(
                name="program",
                kind="program",
                region="imem",
                address=IMEM_BASE,
                size_bytes=4,
                file="program.S",
            ),
            "input": CompilerBundleSymbol(
                name="input",
                kind="input",
                region="dram",
                address=DRAM_BASE + 0x400,
                size_bytes=5,
                file="input.bin",
            ),
        }
    ).write_json5(program_symbol_table_path(program_path))
    (tmp_path / "input.bin").write_bytes(b"hello")

    bundle_dir = tmp_path / "bundle_out"
    assert (
        compiler_main(
            [
                "bundle",
                "--program",
                str(program_path),
                "--output-dir",
                str(bundle_dir),
            ]
        )
        == 0
    )

    loaded = load_executable_bundle(ExecutableBundle.from_directory(bundle_dir))
    assert loaded.symbol_data == {"input": b"hello"}


def test_model_cli_runs_bundle_and_writes_trace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bundle_dir = _write_sample_bundle(tmp_path)
    trace_path = trace_output_path("executable_bundle_model_cli.json")

    assert model_main(["--bundle", str(bundle_dir), "--trace", str(trace_path)]) == 0

    captured = capsys.readouterr().out
    assert "stop_reason: StopReason.EBREAK" in captured
    assert "preloaded_symbols: ['input']" in captured
    assert trace_path.exists()


def _write_sample_bundle(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "sample_bundle"
    write_executable_bundle(
        bundle_dir,
        program_text="""
start:
    li x1, target
    sjal x0, target
target:
    sebreak
""",
        manifest=CompilerBundleManifest(
            symbol_table="program.symbols.json5",
        ),
        symbol_table=CompilerBundleSymbolTable(
            symbols={
                "program": CompilerBundleSymbol(
                    name="program",
                    kind="program",
                    region="imem",
                    address=IMEM_BASE,
                    size_bytes=12,
                    file="program.S",
                ),
                "input": CompilerBundleSymbol(
                    name="input",
                    kind="input",
                    region="dram",
                    address=DRAM_BASE + 0x100,
                    size_bytes=len(b"penguin-input"),
                    file="input.bin",
                ),
                "output": CompilerBundleSymbol(
                    name="output",
                    kind="output",
                    region="dram",
                    address=DRAM_BASE + 0x200,
                    size_bytes=32,
                    file="output.bin",
                ),
            }
        ),
        constants=b"\xCA\xFE",
        symbol_files={"input.bin": b"penguin-input"},
    )
    return bundle_dir
