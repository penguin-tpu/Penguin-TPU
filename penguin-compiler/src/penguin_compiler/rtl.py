"""Helpers for generating RTL-facing program initialization artifacts."""

from __future__ import annotations

from pathlib import Path

from penguin_model import assemble_file
from penguin_model.scalar_encoding import encode_scalar_instruction


def render_verilog_rom_init(
    program_path: str | Path,
    *,
    array_name: str = "imem",
    base_address: int = 0,
) -> str:
    """Render one Verilog include with word assignments for an assembled program."""

    program = assemble_file(program_path, base_address=base_address)
    return "".join(
        f"        {array_name}[{index}] = 32'h{encode_scalar_instruction(instruction):08x};\n"
        for index, instruction in enumerate(program.instructions)
    )


def write_verilog_rom_init(
    output_path: str | Path,
    program_path: str | Path,
    *,
    array_name: str = "imem",
    base_address: int = 0,
) -> Path:
    """Assemble a Penguin program and write a Verilog ROM-init include."""

    output = Path(output_path)
    output.write_text(
        render_verilog_rom_init(
            program_path,
            array_name=array_name,
            base_address=base_address,
        )
    )
    return output


__all__ = ["render_verilog_rom_init", "write_verilog_rom_init"]
