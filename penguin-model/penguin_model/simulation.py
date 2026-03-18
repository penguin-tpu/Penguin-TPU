"""Simulation wrapper around the Penguin hardware core."""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
import re
from typing import Iterable

from .arch_state import PerformanceCounters
from .core import Core
from .logging import TraceLogger, TraceLoggerConfig

_AUTO_TRACE_COUNTERS: dict[str, int] = {}


class Sim:
    """User-facing simulator wrapper around the cycle-driven core."""

    def __init__(self, *args, **kwargs) -> None:
        self.core = Core(*args, **kwargs)

    @property
    def state(self):
        return self.core.state

    @property
    def config(self):
        return self.core.config

    @property
    def memory(self):
        return self.core.memory

    @property
    def perf(self) -> PerformanceCounters:
        return self.core.perf

    def reset(self) -> None:
        self.core.reset()

    def load_program(self, *args, **kwargs) -> None:
        self.core.load_program(*args, **kwargs)

    def tick(self) -> bool:
        return self.core.tick()

    def execute_instruction(self, instruction) -> None:
        self.execute([instruction])

    def execute(
        self,
        program: Iterable,
        *,
        start_pc: int | None = None,
        max_instructions: int | None = None,
        trace_logger: TraceLogger | None = None,
    ) -> PerformanceCounters:
        if trace_logger is None:
            auto_trace_path = _pytest_auto_trace_path()
            if auto_trace_path is not None:
                with TraceLogger(
                    TraceLoggerConfig(
                        filename=str(auto_trace_path),
                        ticks_per_cycle=self.state.config.trace.ticks_per_cycle,
                    )
                ) as auto_trace_logger:
                    return self.execute(
                        program,
                        start_pc=start_pc,
                        max_instructions=max_instructions,
                        trace_logger=auto_trace_logger,
                    )

        self.load_program(
            program,
            start_pc=start_pc,
            max_instructions=max_instructions,
            trace_logger=trace_logger,
        )

        while self.state.stop_reason is None:
            self.tick()

        self.state.trace_logger = None
        return self.state.perf

    def dump_json_trace(
        self,
        program: Iterable,
        trace_path: str | PathLike[str],
        *,
        start_pc: int | None = None,
        max_instructions: int | None = None,
    ) -> PerformanceCounters:
        with TraceLogger(
            TraceLoggerConfig(
                filename=str(trace_path),
                ticks_per_cycle=self.state.config.trace.ticks_per_cycle,
            )
        ) as trace_logger:
            return self.execute(
                program,
                start_pc=start_pc,
                max_instructions=max_instructions,
                trace_logger=trace_logger,
            )


def _pytest_auto_trace_path() -> Path | None:
    current_test = os.environ.get("PYTEST_CURRENT_TEST")
    if current_test is None:
        return None

    base = current_test.rsplit(" ", 1)[0]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    count = _AUTO_TRACE_COUNTERS.get(sanitized, 0)
    _AUTO_TRACE_COUNTERS[sanitized] = count + 1

    repo_root = Path(__file__).resolve().parents[2]
    trace_root = repo_root / "outputs" / "tests"
    trace_root.mkdir(parents=True, exist_ok=True)
    if count == 0:
        return trace_root / f"{sanitized}.json"
    return trace_root / f"{sanitized}_{count}.json"
