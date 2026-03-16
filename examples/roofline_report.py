"""Generate a PI0 workload roofline report and plot for the current Penguin design."""

from __future__ import annotations

import argparse
from pathlib import Path

from penguin_model import (
    DEFAULT_PENGUIN_CORE_CONFIG,
    format_pi0_workload_report,
    plot_pi0_workload_roofline,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/examples/penguin_roofline.png"),
        help="Path to the generated PI0 roofline PNG.",
    )
    parser.add_argument(
        "--core-frequency-mhz",
        type=float,
        help="Optional core frequency used to scale normalized metrics into GB/s and GOPS.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of dominant kernels to include in the printed report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    core_frequency_hz = (
        None if args.core_frequency_mhz is None else args.core_frequency_mhz * 1e6
    )

    print(
        format_pi0_workload_report(
            config=DEFAULT_PENGUIN_CORE_CONFIG,
            core_frequency_hz=core_frequency_hz,
            limit=args.limit,
        )
    )
    plot_path = plot_pi0_workload_roofline(
        args.output,
        config=DEFAULT_PENGUIN_CORE_CONFIG,
    )
    print()
    print(f"plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
