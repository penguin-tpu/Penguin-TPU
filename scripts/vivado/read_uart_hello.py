#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time

import serial


def read_until(
    port: str,
    baud: int,
    timeout_s: float,
    expected: bytes | None,
    min_occurrences: int,
) -> tuple[bytes, list[float]]:
    deadline = time.monotonic() + timeout_s
    collected = bytearray()
    match_times: list[float] = []
    scan_offset = 0

    with serial.Serial(port=port, baudrate=baud, timeout=0.25) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        while time.monotonic() < deadline:
            chunk = ser.read(1)
            if not chunk:
                continue

            collected.extend(chunk)
            if expected is not None:
                while True:
                    match_index = collected.find(expected, scan_offset)
                    if match_index < 0:
                        scan_offset = max(0, len(collected) - len(expected) + 1)
                        break
                    match_times.append(time.monotonic())
                    scan_offset = match_index + len(expected)
                    if len(match_times) >= min_occurrences:
                        return bytes(collected), match_times

    return bytes(collected), match_times


def main() -> int:
    parser = argparse.ArgumentParser(description="Read FPGA UART output and optionally check for a substring.")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--expect", default="Hello World")
    parser.add_argument("--min-occurrences", type=int, default=1)
    parser.add_argument("--min-period", type=float, default=None)
    parser.add_argument("--max-period", type=float, default=None)
    args = parser.parse_args()

    expected = args.expect.encode("ascii") if args.expect else None
    payload, match_times = read_until(
        args.port,
        args.baud,
        args.timeout,
        expected,
        args.min_occurrences,
    )

    display_payload = payload
    if expected is not None:
        expected_offset = payload.find(expected)
        if expected_offset >= 0:
            display_payload = payload[expected_offset:]

    sys.stdout.buffer.write(display_payload)
    if display_payload and display_payload[-1:] != b"\n":
        sys.stdout.write("\n")

    if expected is not None and len(match_times) < args.min_occurrences:
        sys.stderr.write(
            f"expected substring {args.expect!r} observed {len(match_times)} time(s),"
            f" need {args.min_occurrences}, within {args.timeout:.1f}s on {args.port}\n"
        )
        return 1

    if len(match_times) >= 2:
        periods = [later - earlier for earlier, later in zip(match_times, match_times[1:])]
        if args.min_period is not None and any(period < args.min_period for period in periods):
            sys.stderr.write(
                f"observed substring period below minimum {args.min_period:.3f}s on {args.port}:"
                f" {periods}\n"
            )
            return 1
        if args.max_period is not None and any(period > args.max_period for period in periods):
            sys.stderr.write(
                f"observed substring period above maximum {args.max_period:.3f}s on {args.port}:"
                f" {periods}\n"
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
