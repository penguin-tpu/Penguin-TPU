from __future__ import annotations

from pathlib import Path

from penguin_compiler import render_verilog_rom_init
from penguin_compiler.cli import main as compiler_main


def test_render_verilog_rom_init_renders_expected_words(tmp_path: Path) -> None:
    program_path = tmp_path / "sample.S"
    program_path.write_text("start:\n    nop\n    sebreak\n")

    rendered = render_verilog_rom_init(program_path)

    assert rendered == (
        "        imem[0] = 32'h00000013;\n"
        "        imem[1] = 32'h00100073;\n"
    )


def test_compiler_cli_writes_rtl_rom_init_file(tmp_path: Path) -> None:
    program_path = tmp_path / "sample.S"
    output_path = tmp_path / "sample_program_init.vh"
    program_path.write_text(
        "start:\n"
        "    li x1, target\n"
        "    sjal x0, target\n"
        "target:\n"
        "    sebreak\n"
    )

    assert (
        compiler_main(
            [
                "rtl-rom",
                "--program",
                str(program_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    assert output_path.read_text() == (
        "        imem[0] = 32'h00800093;\n"
        "        imem[1] = 32'h0040006f;\n"
        "        imem[2] = 32'h00100073;\n"
    )
