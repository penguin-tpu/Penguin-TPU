"""Run the tiled MXU linear example and emit a Perfetto trace."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from penguin_model import PenguinCoreConfig, run_linear_example


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("examples/out/linear_trace.json"),
        help="Path to the Perfetto-compatible JSON trace output.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = PenguinCoreConfig()
    result = run_linear_example(trace_path=args.trace, config=config)

    print("Linear example completed.")
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
