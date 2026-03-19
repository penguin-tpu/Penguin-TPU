"""Run the MXU linear examples and emit a Perfetto trace."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from penguin_model import PenguinCoreConfig, run_large_linear_example, run_linear_example


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("outputs/examples/linear_trace.json"),
        help="Path to the Perfetto-compatible JSON trace output.",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Run the DMA-backed large linear example instead of the small tiled one.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = PenguinCoreConfig()
    if args.large:
        result = run_large_linear_example(trace_path=args.trace, config=config)
        title = "Large linear example"
    else:
        result = run_linear_example(trace_path=args.trace, config=config)
        title = "Linear example"

    print(f"{title} completed.")
    print(f"  output shape: {tuple(result.output.shape)}")
    print(f"  exact match with PyTorch golden: {torch.equal(result.output, result.golden)}")
    print(f"  instructions: {result.perf.instructions}")
    print(f"  cycles: {result.perf.cycles}")
    print(f"  bytes_read: {result.perf.bytes_read}")
    print(f"  bytes_written: {result.perf.bytes_written}")
    print(f"  trace: {result.trace_path}")
    print("  open the trace in https://ui.perfetto.dev/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
