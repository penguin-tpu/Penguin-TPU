"""Cycle-aware control-flow verification against the JSON trace."""

from __future__ import annotations

from pathlib import Path

from penguin_model import Sim, StopReason, assemble_text
from penguin_model.testbench import fresh_arch_state

from trace_utils import event_end, load_trace, require_stage_event, stage_events, trace_output_path


def _dump_trace(source: str, trace_path: Path):
    core = Sim(state=fresh_arch_state())
    perf = core.dump_json_trace(assemble_text(source), trace_path)
    events = load_trace(trace_path)
    return core, perf, events


def test_taken_branch_executes_two_delay_slots_before_target_starts(tmp_path: Path) -> None:
    core, perf, events = _dump_trace(
        """
    li x1, 1
    li x2, 1
    beq x1, x2, target
    li x3, 3
    li x4, 4
    li x5, 99
target:
    li x6, 6
""",
        trace_output_path("branch_taken_delay_slots.json"),
    )

    delay_slot_1 = require_stage_event(events, stage="execute", contains="addi x3, x0, 3")
    delay_slot_2 = require_stage_event(events, stage="execute", contains="addi x4, x0, 4")
    target = require_stage_event(events, stage="execute", contains="addi x6, x0, 6")

    assert perf.instructions == 6
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert stage_events(events, stage="execute", contains="addi x5, x0, 99") == []
    assert event_end(delay_slot_1) <= delay_slot_2["ts"]
    assert event_end(delay_slot_2) <= target["ts"]


def test_not_taken_branch_still_executes_two_delay_slots_then_continues_sequentially(
    tmp_path: Path,
) -> None:
    core, perf, events = _dump_trace(
        """
    li x1, 1
    li x2, 2
    beq x1, x2, target
    li x3, 3
    li x4, 4
    li x5, 5
target:
    li x6, 6
""",
        trace_output_path("branch_not_taken_delay_slots.json"),
    )

    delay_slot_1 = require_stage_event(events, stage="execute", contains="addi x3, x0, 3")
    delay_slot_2 = require_stage_event(events, stage="execute", contains="addi x4, x0, 4")
    sequential = require_stage_event(events, stage="execute", contains="addi x5, x0, 5")
    target = require_stage_event(events, stage="execute", contains="addi x6, x0, 6")

    assert perf.instructions == 7
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert event_end(delay_slot_1) <= delay_slot_2["ts"]
    assert event_end(delay_slot_2) <= sequential["ts"]
    assert event_end(sequential) <= target["ts"]


def test_jump_target_starts_only_after_two_delay_slots_retire(tmp_path: Path) -> None:
    core, perf, events = _dump_trace(
        """
    jal x10, target
    li x1, 11
    li x2, 22
    li x3, 99
target:
    li x4, 44
""",
        trace_output_path("jump_delay_slots.json"),
    )

    delay_slot_1 = require_stage_event(events, stage="execute", contains="addi x1, x0, 11")
    delay_slot_2 = require_stage_event(events, stage="execute", contains="addi x2, x0, 22")
    target = require_stage_event(events, stage="execute", contains="addi x4, x0, 44")

    assert perf.instructions == 4
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert stage_events(events, stage="execute", contains="addi x3, x0, 99") == []
    assert event_end(delay_slot_1) <= delay_slot_2["ts"]
    assert event_end(delay_slot_2) <= target["ts"]


def test_control_transfer_in_delay_slot_is_illegal(
    tmp_path: Path,
) -> None:
    core, perf, events = _dump_trace(
        """
    jal x1, older_target
    jal x2, younger_target
    li x3, 3
    li x4, 4
    li x5, 5
older_target:
    li x6, 6
younger_target:
    li x7, 7
""",
        trace_output_path("illegal_delay_slot_control_trace.json"),
    )

    assert core.state.stop_reason == StopReason.ILLEGAL_INSTRUCTION
    assert perf.instructions == 0
    assert stage_events(events, stage="execute", contains="addi x3, x0, 3") == []
    assert stage_events(events, stage="execute", contains="addi x7, x0, 7") == []


def test_control_transfer_in_second_delay_slot_is_illegal(
    tmp_path: Path,
) -> None:
    core, perf, events = _dump_trace(
        """
    jal x1, older_target
    li x2, 2
    jal x3, younger_target
    li x4, 4
    li x5, 5
older_target:
    li x6, 6
younger_target:
    li x7, 7
""",
        trace_output_path("illegal_second_delay_slot_control_trace.json"),
    )

    assert core.state.stop_reason == StopReason.ILLEGAL_INSTRUCTION
    assert stage_events(events, stage="execute", contains="jal x3") == []
    assert stage_events(events, stage="execute", contains="addi x4, x0, 4") == []
    assert stage_events(events, stage="execute", contains="addi x7, x0, 7") == []


def test_not_taken_branch_with_control_transfer_in_first_delay_slot_is_illegal(
    tmp_path: Path,
) -> None:
    core, perf, events = _dump_trace(
        """
    li x1, 1
    li x2, 2
    beq x1, x2, target
    jal x3, younger_target
    li x4, 4
    li x5, 5
target:
    li x6, 6
younger_target:
    li x7, 7
""",
        trace_output_path("illegal_not_taken_first_delay_slot_control_trace.json"),
    )

    assert core.state.stop_reason == StopReason.ILLEGAL_INSTRUCTION
    assert stage_events(events, stage="execute", contains="jal x3") == []
    assert stage_events(events, stage="execute", contains="addi x4, x0, 4") == []
    assert stage_events(events, stage="execute", contains="addi x7, x0, 7") == []


def test_not_taken_branch_with_control_transfer_in_second_delay_slot_is_illegal(
    tmp_path: Path,
) -> None:
    core, perf, events = _dump_trace(
        """
    li x1, 1
    li x2, 2
    beq x1, x2, target
    li x3, 3
    jal x4, younger_target
    li x5, 5
target:
    li x6, 6
younger_target:
    li x7, 7
""",
        trace_output_path("illegal_not_taken_second_delay_slot_control_trace.json"),
    )

    assert core.state.stop_reason == StopReason.ILLEGAL_INSTRUCTION
    assert stage_events(events, stage="execute", contains="jal x4") == []
    assert stage_events(events, stage="execute", contains="addi x5, x0, 5") == []
    assert stage_events(events, stage="execute", contains="addi x7, x0, 7") == []
