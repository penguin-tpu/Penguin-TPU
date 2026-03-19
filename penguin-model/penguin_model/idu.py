"""Instruction-decode / dispatch unit."""

from __future__ import annotations

from collections.abc import Callable
from collections import deque

from .stage_data import StageData
from .uop import PipelineUop


class InstructionDecode:
    """Single-uop decode stage with one output buffer per execution unit."""

    def __init__(self, unit_keys: tuple[str, ...]) -> None:
        self._unit_keys = unit_keys
        self.outputs = {unit_key: deque[PipelineUop]() for unit_key in unit_keys}
        self.current_uop: PipelineUop | None = None
        self._stalled = False

    @property
    def is_stalled(self) -> bool:
        return self._stalled

    def reset(self) -> None:
        self.current_uop = None
        self._stalled = False
        for output in self.outputs.values():
            output.clear()

    def is_finished(self) -> bool:
        return self.current_uop is None and all(len(output) == 0 for output in self.outputs.values())

    def tick(
        self,
        cycle: int,
        ifu_output: StageData[PipelineUop | None],
        *,
        on_claim_from_ifu: Callable[[PipelineUop, int], None],
        try_retire_decode_only: Callable[[PipelineUop, int], bool],
        try_dispatch: Callable[[PipelineUop, int], str | None],
    ) -> None:
        self._stalled = False
        started_with_uop = self.current_uop is not None

        if self.current_uop is None:
            claimed = ifu_output.claim()
            if claimed is None:
                return
            claimed.dispatch_start_cycle = cycle
            on_claim_from_ifu(claimed, cycle)
            self.current_uop = claimed
            started_with_uop = False

        assert self.current_uop is not None

        if self.current_uop.dispatch_delay_remaining > 0:
            self.current_uop.dispatch_delay_remaining -= 1
            self._stalled = True
            return

        if try_retire_decode_only(self.current_uop, cycle):
            self.current_uop = None
            claimed = ifu_output.claim()
            if claimed is None:
                return
            claimed.dispatch_start_cycle = cycle
            on_claim_from_ifu(claimed, cycle)
            self.current_uop = claimed
            if self.current_uop.dispatch_delay_remaining > 0:
                self.current_uop.dispatch_delay_remaining -= 1
                self._stalled = True
                return
            unit_key = try_dispatch(self.current_uop, cycle)
            if unit_key is None:
                self._stalled = True
                return
            self.outputs[unit_key].append(self.current_uop)
            self.current_uop = None
            return

        unit_key = try_dispatch(self.current_uop, cycle)
        if unit_key is None:
            self._stalled = True
            return

        self.outputs[unit_key].append(self.current_uop)
        self.current_uop = None
        if started_with_uop:
            claimed = ifu_output.claim()
            if claimed is not None:
                claimed.dispatch_start_cycle = cycle
                on_claim_from_ifu(claimed, cycle)
                self.current_uop = claimed
