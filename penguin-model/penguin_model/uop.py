"""Dynamic in-flight instruction state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .instructions import Instruction


@dataclass(slots=True)
class PipelineUop:
    """Dynamic instruction instance flowing through the simulated pipeline."""

    pc: int
    instruction: Instruction
    insn_id: int
    fetch_stage_open: bool = True
    dispatch_start_cycle: int | None = None
    dispatch_delay_remaining: int = 0
    unit_key: str | None = None
    execute_start_cycle: int | None = None
    completion_cycle: int | None = None
    on_execute_start: Callable[[int], None] | None = None
    on_execute_complete: Callable[[int], None] | None = None
