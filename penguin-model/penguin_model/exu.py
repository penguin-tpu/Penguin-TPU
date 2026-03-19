"""Execution-unit building blocks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from .uop import PipelineUop
from .logging import TraceLogger


@dataclass(slots=True)
class _ActiveExecution:
    uop: PipelineUop


class BufferedExecutionUnit:
    """Single-inflight execution unit fed by an issued-uop queue."""

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
        self._queue: deque[PipelineUop] = deque()
        self._active: list[_ActiveExecution] = []
        self._complete_count = 0
        self._total_instructions = 0
        self._busy_cycles = 0
        self._busy_this_cycle = False

    def reset(self) -> None:
        self._queue.clear()
        self._active.clear()
        self._complete_count = 0
        self._total_instructions = 0
        self._busy_cycles = 0

    def enqueue(self, uop: PipelineUop) -> None:
        self._queue.append(uop)

    def complete_cycle(self, cycle: int) -> None:
        self._complete_count = 0
        self._busy_this_cycle = bool(self._active)

        remaining: list[_ActiveExecution] = []
        for active in self._active:
            completion_cycle = active.uop.completion_cycle
            if completion_cycle is not None and cycle >= completion_cycle:
                uop = active.uop
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
                self._complete_count += 1
            else:
                remaining.append(active)
        self._active = remaining

    def start_cycle(self, cycle: int) -> None:
        candidate = self._queue[0] if self._queue else None
        if (
            candidate is not None
            and candidate.execute_start_cycle is not None
            and cycle >= candidate.execute_start_cycle
        ):
            self._queue.popleft()
            if self.logger is not None:
                if candidate.dispatch_end_cycle is None:
                    self.logger.log_stage_end(candidate.insn_id, "dispatch", lane=1, cycle=cycle)
                self.logger.log_stage_start(
                    candidate.insn_id,
                    "execute",
                    lane=self.lane_id,
                    cycle=cycle,
                )
            if candidate.on_execute_start is not None:
                candidate.on_execute_start(cycle)
            self._active.append(_ActiveExecution(candidate))
            self._total_instructions += 1
            self._busy_this_cycle = True

        if self._busy_this_cycle:
            self._busy_cycles += 1

    def tick(self, cycle: int) -> None:
        self.complete_cycle(cycle)
        self.start_cycle(cycle)

    def flush_completions(self, cycle: int) -> None:
        for active in self._active:
            uop = active.uop
            if uop.on_execute_complete is not None:
                uop.on_execute_complete(cycle)
            if self.logger is not None:
                self.logger.log_stage_end(uop.insn_id, "execute", lane=self.lane_id, cycle=cycle)
                self.logger.log_retire(uop.insn_id, lane=self.lane_id, cycle=cycle)
        self._active.clear()

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
        return bool(self._active) or bool(self._queue)


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
