"""Execution-unit building blocks."""

from __future__ import annotations

from dataclasses import dataclass

from .logging import TraceLogger
from .stage_data import StageData
from .uop import PipelineUop


@dataclass(slots=True)
class _ActiveExecution:
    uop: PipelineUop


class BufferedExecutionUnit:
    """Single-inflight execution unit fed by a one-entry StageData buffer."""

    def __init__(
        self,
        *,
        name: str,
        lane_id: int,
        logger: TraceLogger | None,
    ) -> None:
        self.name = name
        self.lane_id = lane_id
        self.logger = logger
        self._active: _ActiveExecution | None = None
        self._complete_count = 0
        self._total_instructions = 0
        self._busy_cycles = 0
        self._busy_this_cycle = False

    def reset(self) -> None:
        self._active = None
        self._complete_count = 0
        self._total_instructions = 0
        self._busy_cycles = 0

    def complete_cycle(self, cycle: int) -> None:
        self._complete_count = 0
        self._busy_this_cycle = self._active is not None

        if self._active is not None:
            completion_cycle = self._active.uop.completion_cycle
            if completion_cycle is not None and cycle >= completion_cycle:
                uop = self._active.uop
                if uop.on_execute_complete is not None:
                    uop.on_execute_complete(cycle)
                if self.logger is not None:
                    self.logger.log_stage_end(
                        uop.insn_id,
                        "execute",
                        lane=self.lane_id,
                        cycle=cycle,
                    )
                    self.logger.log_retire(uop.insn_id, lane=self.lane_id, cycle=cycle)
                self._active = None
                self._complete_count = 1

    def start_cycle(self, cycle: int, input_buffer: StageData[PipelineUop | None]) -> None:
        if self._active is None:
            candidate = input_buffer.peek()
            if (
                candidate is not None
                and candidate.execute_start_cycle is not None
                and cycle >= candidate.execute_start_cycle
            ):
                input_buffer.claim()
                if self.logger is not None:
                    self.logger.log_stage_end(candidate.insn_id, "dispatch", lane=1, cycle=cycle)
                    self.logger.log_stage_start(
                        candidate.insn_id,
                        "execute",
                        lane=self.lane_id,
                        cycle=cycle,
                )
                if candidate.on_execute_start is not None:
                    candidate.on_execute_start(cycle)
                self._active = _ActiveExecution(candidate)
                self._total_instructions += 1
                self._busy_this_cycle = True

        if self._busy_this_cycle:
            self._busy_cycles += 1

    def tick(self, cycle: int, input_buffer: StageData[PipelineUop | None]) -> None:
        self.complete_cycle(cycle)
        self.start_cycle(cycle, input_buffer)

    def flush_completions(self, cycle: int) -> None:
        if self._active is None:
            return
        uop = self._active.uop
        if uop.on_execute_complete is not None:
            uop.on_execute_complete(cycle)
        if self.logger is not None:
            self.logger.log_stage_end(uop.insn_id, "execute", lane=self.lane_id, cycle=cycle)
            self.logger.log_retire(uop.insn_id, lane=self.lane_id, cycle=cycle)
        self._active = None

    @property
    def complete_count(self) -> int:
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        return self._busy_cycles

    @property
    def has_in_flight(self) -> bool:
        return self._active is not None


class ScalarExecutionUnit(BufferedExecutionUnit):
    pass


class DmaExecutionUnit(BufferedExecutionUnit):
    pass


class TensorMemoryExecutionUnit(BufferedExecutionUnit):
    pass


class MatrixExecutionUnit(BufferedExecutionUnit):
    pass


class VectorExecutionUnit(BufferedExecutionUnit):
    pass


class XLUExecutionUnit(BufferedExecutionUnit):
    pass
