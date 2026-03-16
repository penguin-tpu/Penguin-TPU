#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time

import serial


def read_until(port: str, baud: int, timeout_s: float, expected: bytes | None) -> bytes:
    deadline = time.monotonic() + timeout_s
    collected = bytearray()

    with serial.Serial(port=port, baudrate=baud, timeout=0.25) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        while time.monotonic() < deadline:
            chunk = ser.read(256)
            if not chunk:
                continue

            collected.extend(chunk)
            if expected is not None and expected in collected:
                break

    return bytes(collected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read FPGA UART output and optionally check for a substring.")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--expect", default="Hello World")
    args = parser.parse_args()

    expected = args.expect.encode("ascii") if args.expect else None
    payload = read_until(args.port, args.baud, args.timeout, expected)

    display_payload = payload
    if expected is not None:
        expected_offset = payload.find(expected)
        if expected_offset >= 0:
            display_payload = payload[expected_offset:]

    sys.stdout.buffer.write(display_payload)
    if display_payload and display_payload[-1:] != b"\n":
        sys.stdout.write("\n")

    if expected is not None and expected not in payload:
        sys.stderr.write(
            f"expected substring {args.expect!r} not observed within {args.timeout:.1f}s on {args.port}\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
