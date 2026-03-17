"""Generate checked-in symbol-table sidecars for assembly test programs."""

from __future__ import annotations

from pathlib import Path

from penguin_compiler import BundleSymbol, BundleSymbolTable
from penguin_model import DEFAULT_PENGUIN_CORE_CONFIG, IMEM_BASE, assemble_file

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOT = REPO_ROOT / "tests" / "vectors" / "programs"
DEFAULT_CONFIG = DEFAULT_PENGUIN_CORE_CONFIG
VMEM_BASE = DEFAULT_CONFIG.memory_map.vmem.base
DRAM_BASE = DEFAULT_CONFIG.memory_map.dram.base
MREG_BYTES = DEFAULT_CONFIG.mreg_bytes
WEIGHT_SLOT_BYTES = DEFAULT_CONFIG.weight_slot_bytes
MATMUL_RESULT_BYTES = DEFAULT_CONFIG.matmul_result_bytes


def _default_program_symbol(program_path: Path) -> BundleSymbol:
    program = assemble_file(program_path)
    return BundleSymbol(
        name="program",
        kind="program",
        region="imem",
        address=IMEM_BASE,
        size_bytes=len(program) * 4,
        file=program_path.name,
    )


def _example_addresses() -> dict[str, int]:
    mreg_bytes = MREG_BYTES
    return {
        "activation0": VMEM_BASE + 0 * mreg_bytes,
        "activation1": VMEM_BASE + 1 * mreg_bytes,
        "weight0": VMEM_BASE + 2 * mreg_bytes,
        "weight1": VMEM_BASE + 3 * mreg_bytes,
        "bias0": VMEM_BASE + 4 * mreg_bytes,
        "bias1": VMEM_BASE + 6 * mreg_bytes,
        "bias2": VMEM_BASE + 8 * mreg_bytes,
        "output00": VMEM_BASE + 10 * mreg_bytes,
        "output01": VMEM_BASE + 12 * mreg_bytes,
        "output10": VMEM_BASE + 14 * mreg_bytes,
        "output11": VMEM_BASE + 16 * mreg_bytes,
        "dma_act_scratch": VMEM_BASE + 0 * mreg_bytes,
        "dma_weight_scratch": VMEM_BASE + 1 * mreg_bytes,
        "dma_output_scratch": VMEM_BASE + 2 * mreg_bytes,
        "dma_act_dram_base": DRAM_BASE + 0x0100_0000,
        "dma_weight_dram_base": DRAM_BASE + 0x0200_0000,
        "dma_output_dram_base": DRAM_BASE + 0x0300_0000,
    }


def _custom_symbols(relative_path: str) -> dict[str, BundleSymbol]:
    addresses = _example_addresses()

    if relative_path == "scalar/examples/scalar_matmul.S":
        return {
            "dram_input": BundleSymbol(
                name="dram_input",
                kind="input",
                region="dram",
                address=DRAM_BASE + 0x100,
                size_bytes=32,
            ),
            "vmem_stage": BundleSymbol(
                name="vmem_stage",
                kind="scratch",
                region="vmem",
                address=VMEM_BASE + 0x40,
                size_bytes=32,
            ),
            "vmem_output": BundleSymbol(
                name="vmem_output",
                kind="output",
                region="vmem",
                address=VMEM_BASE + 0x80,
                size_bytes=4,
            ),
        }

    if relative_path == "tensor/examples/matmul.S":
        return {
            "activation": BundleSymbol(
                name="activation",
                kind="input",
                region="vmem",
                address=addresses["activation0"],
                size_bytes=MREG_BYTES,
            ),
            "weights": BundleSymbol(
                name="weights",
                kind="input",
                region="vmem",
                address=addresses["weight0"],
                size_bytes=WEIGHT_SLOT_BYTES,
            ),
            "output": BundleSymbol(
                name="output",
                kind="output",
                region="vmem",
                address=addresses["output00"],
                size_bytes=MATMUL_RESULT_BYTES,
            ),
        }

    if relative_path == "tensor/examples/linear.S":
        return {
            "activation": BundleSymbol(
                name="activation",
                kind="input",
                region="vmem",
                address=addresses["activation0"],
                size_bytes=2 * MREG_BYTES,
            ),
            "weights": BundleSymbol(
                name="weights",
                kind="input",
                region="vmem",
                address=addresses["weight0"],
                size_bytes=2 * WEIGHT_SLOT_BYTES,
            ),
            "bias": BundleSymbol(
                name="bias",
                kind="input",
                region="vmem",
                address=addresses["bias0"],
                size_bytes=2 * MATMUL_RESULT_BYTES,
            ),
            "output": BundleSymbol(
                name="output",
                kind="output",
                region="vmem",
                address=addresses["output00"],
                size_bytes=4 * MATMUL_RESULT_BYTES,
            ),
        }

    if relative_path == "tensor/examples/matmul_large.S":
        return {
            "activation_tiles": BundleSymbol(
                name="activation_tiles",
                kind="input",
                region="dram",
                address=addresses["dma_act_dram_base"],
                size_bytes=4 * MREG_BYTES,
            ),
            "weight_tiles": BundleSymbol(
                name="weight_tiles",
                kind="input",
                region="dram",
                address=addresses["dma_weight_dram_base"],
                size_bytes=4 * WEIGHT_SLOT_BYTES,
            ),
            "output_tiles": BundleSymbol(
                name="output_tiles",
                kind="output",
                region="dram",
                address=addresses["dma_output_dram_base"],
                size_bytes=4 * MATMUL_RESULT_BYTES,
            ),
            "activation_scratch": BundleSymbol(
                name="activation_scratch",
                kind="scratch",
                region="vmem",
                address=addresses["dma_act_scratch"],
                size_bytes=MREG_BYTES,
            ),
            "weight_scratch": BundleSymbol(
                name="weight_scratch",
                kind="scratch",
                region="vmem",
                address=addresses["dma_weight_scratch"],
                size_bytes=WEIGHT_SLOT_BYTES,
            ),
            "output_scratch": BundleSymbol(
                name="output_scratch",
                kind="scratch",
                region="vmem",
                address=addresses["dma_output_scratch"],
                size_bytes=MATMUL_RESULT_BYTES,
            ),
        }

    if relative_path == "tensor/examples/linear_large.S":
        return {
            "activation_tiles": BundleSymbol(
                name="activation_tiles",
                kind="input",
                region="dram",
                address=addresses["dma_act_dram_base"],
                size_bytes=6 * MREG_BYTES,
            ),
            "weight_tiles": BundleSymbol(
                name="weight_tiles",
                kind="input",
                region="dram",
                address=addresses["dma_weight_dram_base"],
                size_bytes=6 * WEIGHT_SLOT_BYTES,
            ),
            "bias": BundleSymbol(
                name="bias",
                kind="input",
                region="vmem",
                address=addresses["bias0"],
                size_bytes=3 * MATMUL_RESULT_BYTES,
            ),
            "output_tiles": BundleSymbol(
                name="output_tiles",
                kind="output",
                region="dram",
                address=addresses["dma_output_dram_base"],
                size_bytes=9 * MATMUL_RESULT_BYTES,
            ),
            "activation_scratch": BundleSymbol(
                name="activation_scratch",
                kind="scratch",
                region="vmem",
                address=addresses["dma_act_scratch"],
                size_bytes=MREG_BYTES,
            ),
            "weight_scratch": BundleSymbol(
                name="weight_scratch",
                kind="scratch",
                region="vmem",
                address=addresses["dma_weight_scratch"],
                size_bytes=WEIGHT_SLOT_BYTES,
            ),
            "output_scratch": BundleSymbol(
                name="output_scratch",
                kind="scratch",
                region="vmem",
                address=addresses["dma_output_scratch"],
                size_bytes=MATMUL_RESULT_BYTES,
            ),
        }

    if relative_path == "tensor/examples/gemma_linear_64x32.S":
        return {
            "activation": BundleSymbol(
                name="activation",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x0000,
                size_bytes=MREG_BYTES,
            ),
            "weights": BundleSymbol(
                name="weights",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x1000,
                size_bytes=WEIGHT_SLOT_BYTES,
            ),
            "output": BundleSymbol(
                name="output",
                kind="output",
                region="vmem",
                address=VMEM_BASE + 0x2000,
                size_bytes=MREG_BYTES,
            ),
        }

    if relative_path == "tensor/examples/gemma_mlp_gate_64x32.S":
        return {
            "gate": BundleSymbol(
                name="gate",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x0000,
                size_bytes=MREG_BYTES,
            ),
            "up": BundleSymbol(
                name="up",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x1000,
                size_bytes=MREG_BYTES,
            ),
            "constants": BundleSymbol(
                name="constants",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x2000,
                size_bytes=6 * MREG_BYTES,
            ),
            "output": BundleSymbol(
                name="output",
                kind="output",
                region="vmem",
                address=VMEM_BASE + 0x8000,
                size_bytes=MREG_BYTES,
            ),
        }

    if relative_path == "tensor/examples/gemma_vadd_64x32.S":
        return {
            "lhs": BundleSymbol(
                name="lhs",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x0000,
                size_bytes=MREG_BYTES,
            ),
            "rhs": BundleSymbol(
                name="rhs",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x1000,
                size_bytes=MREG_BYTES,
            ),
            "output": BundleSymbol(
                name="output",
                kind="output",
                region="vmem",
                address=VMEM_BASE + 0x2000,
                size_bytes=MREG_BYTES,
            ),
        }

    if relative_path == "tensor/examples/gemma_transpose_64x32.S":
        return {
            "input": BundleSymbol(
                name="input",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x0000,
                size_bytes=MREG_BYTES,
            ),
            "output": BundleSymbol(
                name="output",
                kind="output",
                region="vmem",
                address=VMEM_BASE + 0x1000,
                size_bytes=MREG_BYTES,
            ),
        }

    if relative_path == "tensor/examples/gemma_attention_scores_64x64.S":
        return {
            "activation": BundleSymbol(
                name="activation",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x0000,
                size_bytes=MREG_BYTES,
            ),
            "weights": BundleSymbol(
                name="weights",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x1000,
                size_bytes=2 * WEIGHT_SLOT_BYTES,
            ),
            "output": BundleSymbol(
                name="output",
                kind="output",
                region="vmem",
                address=VMEM_BASE + 0x3000,
                size_bytes=2 * MREG_BYTES,
            ),
        }

    if relative_path == "tensor/examples/gemma_attention_scores_64x16.S":
        return {
            "activation": BundleSymbol(
                name="activation",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x0000,
                size_bytes=MREG_BYTES,
            ),
            "weights": BundleSymbol(
                name="weights",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x1000,
                size_bytes=WEIGHT_SLOT_BYTES,
            ),
            "output": BundleSymbol(
                name="output",
                kind="output",
                region="vmem",
                address=VMEM_BASE + 0x2000,
                size_bytes=MREG_BYTES,
            ),
        }

    if relative_path == "tensor/examples/gemma_softmax_64x16.S":
        return {
            "input": BundleSymbol(
                name="input",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x0000,
                size_bytes=MREG_BYTES,
            ),
            "output": BundleSymbol(
                name="output",
                kind="output",
                region="vmem",
                address=VMEM_BASE + 0x2000,
                size_bytes=MREG_BYTES,
            ),
        }

    if relative_path == "tensor/examples/gemma_attention_context_64x32.S":
        return {
            "activation": BundleSymbol(
                name="activation",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x0000,
                size_bytes=MREG_BYTES,
            ),
            "weights": BundleSymbol(
                name="weights",
                kind="input",
                region="vmem",
                address=VMEM_BASE + 0x1000,
                size_bytes=WEIGHT_SLOT_BYTES,
            ),
            "output": BundleSymbol(
                name="output",
                kind="output",
                region="vmem",
                address=VMEM_BASE + 0x2000,
                size_bytes=MREG_BYTES,
            ),
        }

    return {}


def main() -> int:
    for stale_path in PROGRAM_ROOT.rglob("*.symbols.json"):
        stale_path.unlink()

    generated = 0
    for program_path in sorted(PROGRAM_ROOT.rglob("*.S")):
        relative_path = program_path.relative_to(PROGRAM_ROOT).as_posix()
        symbols = {"program": _default_program_symbol(program_path)}
        symbols.update(_custom_symbols(relative_path))
        BundleSymbolTable(symbols=symbols).write_json5(
            program_path.with_name(f"{program_path.stem}.symbols.json5")
        )
        generated += 1

    print(f"Generated {generated} symbol-table sidecars under {PROGRAM_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
