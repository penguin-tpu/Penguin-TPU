"""Helpers for asserting instruction timing from JSON execution traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


def trace_output_path(name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    trace_root = repo_root / "outputs" / "tests"
    trace_root.mkdir(parents=True, exist_ok=True)
    return trace_root / name


def stage_events(
    events: list[dict[str, Any]],
    *,
    stage: str,
    contains: str,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if event.get("cat") == stage and contains in event.get("name", "")
    ]


def require_stage_event(
    events: list[dict[str, Any]],
    *,
    stage: str,
    contains: str,
    occurrence: int = 0,
) -> dict[str, Any]:
    matches = stage_events(events, stage=stage, contains=contains)
    if occurrence >= len(matches):
        raise AssertionError(
            f"Expected {stage!r} event containing {contains!r}, found {len(matches)}"
        )
    return matches[occurrence]


def event_end(event: dict[str, Any]) -> int:
    return int(event["ts"]) + int(event.get("dur", 0))
