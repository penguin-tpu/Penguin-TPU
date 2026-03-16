"""Run the fixed-shape Gemma MLP example and emit a Perfetto trace."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.gemma_workloads import run_gemma_mlp_example


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("outputs/examples/gemma_mlp_trace.json"),
        help="Path to the Perfetto-compatible JSON trace output.",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("outputs/examples/bundles/gemma_mlp"),
        help="Directory where stage bundle directories will be written.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = run_gemma_mlp_example(trace_path=args.trace, bundle_root=args.bundle_root)

    print("Gemma MLP example completed.")
    print(f"  output shape: {tuple(result.output.shape)}")
    print(f"  exact match with PyTorch golden: {torch.equal(result.output, result.golden)}")
    print(f"  instructions: {result.perf.instructions}")
    print(f"  cycles: {result.perf.cycles}")
    print(f"  bytes_read: {result.perf.bytes_read}")
    print(f"  bytes_written: {result.perf.bytes_written}")
    print(f"  trace: {result.trace_path}")
    print(f"  bundle_root: {result.bundle_root}")
    print("  open the trace in https://ui.perfetto.dev/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
