"""Instruction-fetch unit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

from .instructions import Instruction
from .stage_data import StageData
from .uop import PipelineUop


class InstructionFetch:
    """Single-stream IFU with one buffered output."""

    def __init__(self) -> None:
        self.output: StageData[PipelineUop | None] = StageData(None)
        self._program: Sequence[Instruction] = ()
        self._program_base = 0
        self._program_end = 0
        self._fetch_pc = 0
        self._next_insn_id = 0
        self._stalled = False

    @property
    def fetch_pc(self) -> int:
        return self._fetch_pc

    @property
    def next_insn_id(self) -> int:
        return self._next_insn_id

    def set_fetch_pc(self, pc: int) -> None:
        self._fetch_pc = pc & 0xFFFF_FFFF

    def load_program(
        self,
        program: Sequence[Instruction],
        *,
        program_base: int,
        start_pc: int,
    ) -> None:
        self._program = program
        self._program_base = program_base
        self._program_end = program_base + len(program) * 4
        self._fetch_pc = start_pc & 0xFFFF_FFFF
        self._next_insn_id = 0
        self._stalled = False
        self.output.reset()

    def reset(self, *, start_pc: int | None = None) -> None:
        self.output.reset()
        self._stalled = False
        self._next_insn_id = 0
        if start_pc is not None:
            self._fetch_pc = start_pc & 0xFFFF_FFFF

    def is_finished(self) -> bool:
        return self._fetch_pc >= self._program_end and not self.output.is_valid()

    def tick(
        self,
        cycle: int,
        *,
        allow_fetch: bool,
        on_fetch: Callable[[PipelineUop, int], None],
        on_fetch_stall: Callable[[PipelineUop, int], None],
        on_misaligned_fetch: Callable[[], None],
        on_fetch_pc_advanced: Callable[[int], None],
    ) -> None:
        if self.output.should_stall():
            stalled_uop = self.output.peek()
            if stalled_uop is not None and not self._stalled:
                on_fetch_stall(stalled_uop, cycle)
            self._stalled = True
            return

        self._stalled = False

        if not allow_fetch:
            self.output.prepare(None)
            return

        if self._fetch_pc % 4 != 0:
            on_misaligned_fetch()
            self.output.prepare(None)
            return

        if self._fetch_pc < self._program_base or self._fetch_pc >= self._program_end:
            self.output.prepare(None)
            return

        instruction_index = (self._fetch_pc - self._program_base) // 4
        uop = PipelineUop(
            pc=self._fetch_pc,
            instruction=self._program[instruction_index],
            insn_id=self._next_insn_id,
        )
        self._next_insn_id += 1
        on_fetch(uop, cycle)
        self.output.prepare(uop)
        fetched_pc = self._fetch_pc
        self._fetch_pc = (fetched_pc + 4) & 0xFFFF_FFFF
        on_fetch_pc_advanced(fetched_pc)
