"""Core execution model for the Penguin scalar integer subset."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import os
from os import PathLike
from pathlib import Path
import re
from typing import TYPE_CHECKING, Callable

from .arch_state import ArchState, PerformanceCounters, StopReason
from .core_config import DEFAULT_PENGUIN_CORE_CONFIG, PenguinCoreConfig
from .instructions import (
    ALL_INSTRUCTION_SPECS,
    BType,
    DMAType,
    EmptyType,
    IType,
    Instruction,
    JType,
    MXUMatmulAccType,
    MXUMatmulType,
    RType,
    ScaleImmType,
    ScaleMemType,
    SType,
    TensorMemType,
    UType,
    XLUTransposeType,
    VPUBinaryType,
    VPUUnaryType,
    WeightMemType,
)
from .memory import Memory
from .tensor import (
    compute_bf16_matmul,
    compute_bf16_row_reduce_max,
    compute_bf16_row_reduce_sum,
    compute_bf16_transpose,
    compute_bf16_vadd,
    compute_bf16_vexp,
    compute_bf16_vmax,
    compute_bf16_vmin,
    compute_bf16_vmov,
    compute_bf16_vrecip,
    compute_bf16_vrelu,
    compute_bf16_vsub,
    compute_bf16_vmul,
)

if TYPE_CHECKING:
    from .logging import TraceLogger


class _InstructionLatencyView(Mapping[str, int]):
    """Live view over registered instruction latencies."""

    def __getitem__(self, key: str) -> int:
        return ALL_INSTRUCTION_SPECS[key].latency

    def __iter__(self) -> Iterator[str]:
        return iter(ALL_INSTRUCTION_SPECS)

    def __len__(self) -> int:
        return len(ALL_INSTRUCTION_SPECS)


INSTRUCTION_LATENCY: Mapping[str, int] = _InstructionLatencyView()

TRACE_TICKS_PER_CYCLE = DEFAULT_PENGUIN_CORE_CONFIG.trace.ticks_per_cycle
FETCH_LANE = 0
DISPATCH_LANE = 1
SALU_LANE = 2
DMA_LANE = 3
TMEM_LANE = 4
MXU0_LANE = 5
MXU1_LANE = 6
VPU_LANE = 7
XLU_LANE = 8
DMA_TRANSFER_LANE_BASE = 30
_AUTO_TRACE_COUNTERS: dict[str, int] = {}


def _format_instruction(instruction: Instruction) -> str:
    params = instruction.params
    mnemonic = instruction.mnemonic

    if isinstance(params, RType):
        return f"{mnemonic} x{params.rd}, x{params.rs1}, x{params.rs2}"
    if isinstance(params, IType):
        if mnemonic == "sld":
            return f"{mnemonic} x{params.rd}, {params.imm}(x{params.rs1})"
        if mnemonic == "sjalr":
            return f"{mnemonic} x{params.rd}, x{params.rs1}, {params.imm}"
        return f"{mnemonic} x{params.rd}, x{params.rs1}, {params.imm}"
    if isinstance(params, SType):
        return f"{mnemonic} x{params.rs2}, {params.imm}(x{params.rs1})"
    if isinstance(params, BType):
        return f"{mnemonic} x{params.rs1}, x{params.rs2}, {params.imm}"
    if isinstance(params, UType):
        return f"{mnemonic} x{params.rd}, {params.imm}"
    if isinstance(params, JType):
        return f"{mnemonic} x{params.rd}, {params.imm}"
    if isinstance(params, EmptyType):
        return mnemonic
    if isinstance(params, DMAType):
        return f"{mnemonic} x{params.dram_rs}, x{params.vmem_rs}, x{params.size_rs}"
    if isinstance(params, ScaleImmType):
        return f"{mnemonic} e{params.ed}, {params.imm}"
    if isinstance(params, ScaleMemType):
        return f"{mnemonic} e{params.ed}, {params.imm}(x{params.rs1})"
    if isinstance(params, TensorMemType):
        return f"{mnemonic} m{params.mreg}, {params.imm}(x{params.rs1})"
    if isinstance(params, WeightMemType):
        return f"{mnemonic} w{params.slot}, {params.imm}(x{params.rs1})"
    if isinstance(params, MXUMatmulType):
        return f"{mnemonic} m{params.md}, m{params.ms}, w{params.ws}, e{params.ea}, e{params.eb}"
    if isinstance(params, MXUMatmulAccType):
        return (
            f"{mnemonic} m{params.md}, m{params.ms}, w{params.ws}, "
            f"m{params.mp}, e{params.ea}, e{params.eb}"
        )
    if isinstance(params, VPUBinaryType):
        return f"{mnemonic} m{params.md}, m{params.ms1}, m{params.ms2}"
    if isinstance(params, VPUUnaryType):
        return f"{mnemonic} m{params.md}, m{params.ms}"
    if isinstance(params, XLUTransposeType):
        return f"{mnemonic} m{params.md}, m{params.ms}"
    return f"{mnemonic} {params}"


def _execute_lane_for_instruction(instruction: Instruction) -> int:
    if instruction.mnemonic.startswith("dma."):
        return DMA_LANE
    if instruction.mnemonic in {"vload", "vstore"} or instruction.mnemonic.startswith("mxu.push."):
        return TMEM_LANE
    if isinstance(instruction.params, (VPUBinaryType, VPUUnaryType)):
        return VPU_LANE
    if isinstance(instruction.params, XLUTransposeType):
        return XLU_LANE
    if instruction.mnemonic.endswith(".mxu0"):
        return MXU0_LANE
    if instruction.mnemonic.endswith(".mxu1"):
        return MXU1_LANE
    return SALU_LANE


def _dma_channel_for_instruction(instruction: Instruction) -> int | None:
    parts = instruction.mnemonic.split(".")
    if len(parts) != 3 or parts[0] != "dma":
        return None
    if not parts[2].startswith("ch"):
        return None
    return int(parts[2][2:])


def _is_dma_wait_instruction(instruction: Instruction) -> bool:
    return instruction.mnemonic.startswith("dma.wait.")


def _dma_transfer_lane(channel: int) -> int:
    return DMA_TRANSFER_LANE_BASE + channel


def _is_async_tensor_lane(lane: int) -> bool:
    return lane in {MXU0_LANE, MXU1_LANE, VPU_LANE, XLU_LANE}


def _frontend_latency_cycles(instruction: Instruction, execute_lane: int, total_latency: int) -> int:
    if _is_async_tensor_lane(execute_lane):
        return 1
    if _is_dma_wait_instruction(instruction):
        return total_latency
    if execute_lane == DMA_LANE:
        return 1
    return total_latency


def _lane_occupancy_cycles(instruction: Instruction, execute_lane: int, total_latency: int) -> int:
    if _is_async_tensor_lane(execute_lane):
        return total_latency
    if execute_lane == DMA_LANE and not _is_dma_wait_instruction(instruction):
        return 1
    return total_latency


def _instruction_latency_cycles(
    state: ArchState,
    instruction: Instruction,
    default_latency: int,
) -> int:
    if instruction.mnemonic == "vload":
        return state.config.vload_latency_cycles
    if instruction.mnemonic == "vstore":
        return state.config.vstore_latency_cycles
    if instruction.mnemonic.startswith("mxu.push."):
        return state.config.mxu_push_latency_cycles
    if instruction.mnemonic.startswith("matmul"):
        return state.config.matmul_latency_cycles
    if isinstance(instruction.params, VPUBinaryType):
        return state.config.vpu_simple_op_latency_cycles
    if isinstance(instruction.params, VPUUnaryType):
        if instruction.mnemonic in {"vexp", "vrecip"}:
            return state.config.vpu_non_pipelineable_op_latency_cycles
        return state.config.vpu_simple_op_latency_cycles
    if isinstance(instruction.params, XLUTransposeType):
        return state.config.xlu_transpose_latency_cycles
    return default_latency


@dataclass(slots=True)
class _ScheduledAction:
    cycle: int
    order: int
    callback: Callable[[], None]


@dataclass(slots=True)
class _PipelineSlot:
    pc: int
    instruction: Instruction
    insn_id: int
    dispatch_start_cycle: int | None = None
    fetch_stage_open: bool = True


@dataclass(slots=True)
class _ControlShadow:
    insn_id: int
    delay_slots_seen: int
    resolved: bool = False
    target: int | None = None


def _is_control_transfer_instruction(instruction: Instruction) -> bool:
    return isinstance(instruction.params, (BType, JType)) or instruction.mnemonic == "sjalr"


class Sim:
    """Cycle-driven Penguin simulator with claim-based frontend timing."""

    def __init__(
        self,
        state: ArchState | None = None,
        *,
        memory: Memory | None = None,
        config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
    ) -> None:
        if state is not None and memory is not None:
            raise ValueError("Pass either state or memory, not both")
        if state is None:
            if memory is not None:
                state = ArchState.with_memory_sizes(
                    dram_size=memory.size,
                    vmem_size=memory.size,
                    imem_size=config.memory_map.imem.size,
                    config=config,
                )
                state.vmem = memory
            else:
                state = ArchState.from_config(config)
        else:
            config = state.config
        self.state = state
        self._program: Sequence[Instruction] = ()
        self._program_base = 0
        self._program_end = 0
        self._max_instructions: int | None = None
        self._start_count = 0
        self._step_limit_reached = False
        self._stop_logged = False
        self._next_insn_id = 0
        self._fetch_pc = 0
        self._if_slot: _PipelineSlot | None = None
        self._id_slot: _PipelineSlot | None = None
        self._control_shadows: list[_ControlShadow] = []
        self._pending_actions: dict[int, list[_ScheduledAction]] = {}
        self._next_action_order = 0
        self._reset_scoreboards(self.state.perf.cycles)
        self._reset_lane_state()

    @property
    def config(self) -> PenguinCoreConfig:
        return self.state.config

    @property
    def memory(self) -> Memory:
        return self.state.vmem

    @property
    def perf(self) -> PerformanceCounters:
        return self.state.perf

    def _reset_lane_state(self) -> None:
        self._cycle_exu_ready = {
            SALU_LANE: 0,
            DMA_LANE: 0,
            TMEM_LANE: 0,
            MXU0_LANE: 0,
            MXU1_LANE: 0,
            VPU_LANE: 0,
            XLU_LANE: 0,
        }
        self._pending_actions = {}
        self._next_action_order = 0
        self._if_slot = None
        self._id_slot = None
        self._control_shadows = []
        self._vmem_ready_cycle = 0
        self._next_insn_id = 0
        self._stop_logged = False

    def _reset_scoreboards(self, base_cycle: int) -> None:
        self._xreg_ready = [base_cycle] * self.state.config.scalar.xreg_count
        self._ereg_ready = [base_cycle] * self.state.config.scale.num_ereg
        self._mreg_ready = [base_cycle] * self.state.config.tensor.num_mreg
        self._weight_ready = [
            [base_cycle] * self.state.config.tensor.weight_slots_per_mxu
            for _ in range(self.state.config.tensor.mxu_count)
        ]
        self._vmem_ready_cycle = base_cycle
        self._xreg_ready[0] = base_cycle

    def reset(self) -> None:
        self.state.clear_dma_channels()
        self.state = ArchState(
            dram=self.state.dram,
            vmem=self.state.vmem,
            imem=self.state.imem,
            dma_channels=self.state.dma_channels,
            config=self.state.config,
            mreg=self.state.mreg,
            mxu_weight=self.state.mxu_weight,
            ereg=self.state.ereg,
            mem_base=self.state.mem_base,
        )
        self._program = ()
        self._program_base = 0
        self._program_end = 0
        self._max_instructions = None
        self._start_count = 0
        self._step_limit_reached = False
        self._reset_scoreboards(self.state.perf.cycles)
        self._reset_lane_state()

    def _schedule_action(self, cycle: int, callback: Callable[[], None]) -> None:
        action = _ScheduledAction(cycle=cycle, order=self._next_action_order, callback=callback)
        self._next_action_order += 1
        self._pending_actions.setdefault(cycle, []).append(action)

    def _run_actions_for_cycle(self, cycle: int) -> None:
        actions = self._pending_actions.pop(cycle, ())
        for action in sorted(actions, key=lambda item: item.order):
            action.callback()

    def _has_pending_actions(self) -> bool:
        return any(self._pending_actions.values())

    def _trace_cycle(self, cycle: int) -> int:
        return cycle * self.state.config.trace.ticks_per_cycle

    def _log_stop(self, cycle: int) -> None:
        if self.state.trace_logger is None or self.state.stop_reason is None or self._stop_logged:
            return
        self.state.trace_logger.log_stop(self.state.stop_reason.value, cycle=self._trace_cycle(cycle))
        self._stop_logged = True

    def _set_execute_pc(self, pc: int, *, trace_cycle: int) -> int:
        saved_pc = self.state.pc
        self.state.pc = pc
        self.state.trace_end_cycle = self._trace_cycle(trace_cycle)
        self.state.trace_start_cycle = self._trace_cycle(trace_cycle)
        self.state.control_transfer_set = False
        return saved_pc

    def _restore_fetch_pc(self, saved_pc: int) -> None:
        del saved_pc
        self.state.pc = self._fetch_pc

    def _refresh_fetch_redirect(self) -> None:
        if not self._control_shadows:
            return
        youngest = self._control_shadows[-1]
        if not youngest.resolved or youngest.target is None:
            return
        if youngest.delay_slots_seen < self.state.config.scalar.control_flow_delay_slots:
            return
        self._fetch_pc = youngest.target
        self._control_shadows.clear()

    def _consume_control_transfer(self, insn_id: int) -> None:
        if self.state.stop_reason is not None:
            return
        shadow = next((item for item in self._control_shadows if item.insn_id == insn_id), None)
        if shadow is None:
            self.state.control_transfer_set = False
            self.state.next_pc = None
            self.state.delay_slots_remaining = 0
            return
        if self.state.control_transfer_set and self.state.next_pc is not None:
            shadow.resolved = True
            shadow.target = self.state.next_pc
            shadow.delay_slots_seen = min(
                shadow.delay_slots_seen,
                self.state.config.scalar.control_flow_delay_slots,
            )
            self._refresh_fetch_redirect()
        else:
            self._control_shadows = [item for item in self._control_shadows if item.insn_id != insn_id]
            self._refresh_fetch_redirect()
        self.state.control_transfer_set = False
        self.state.next_pc = None
        self.state.delay_slots_remaining = 0

    def _log_fetch_start(self, slot: _PipelineSlot, cycle: int) -> None:
        logger = self.state.trace_logger
        if logger is None:
            return
        trace_cycle = self._trace_cycle(cycle)
        logger.log_insn(slot.insn_id, _format_instruction(slot.instruction))
        logger.log_arch_value("pc", 0, slot.pc, cycle=trace_cycle)
        logger.log_stage_start(slot.insn_id, "fetch", lane=FETCH_LANE, cycle=trace_cycle)

    def _advance_fetch_pc_after_fetch(self, fetched_pc: int) -> None:
        self._fetch_pc = (fetched_pc + 4) & 0xFFFF_FFFF
        self._refresh_fetch_redirect()

    def _fetch_instruction(self, cycle: int) -> _PipelineSlot | None:
        if self._step_limit_reached:
            return
        if self._fetch_pc % 4 != 0:
            self.state.stop(StopReason.INSTRUCTION_ADDRESS_MISALIGNED)
            return None
        if self._fetch_pc < self._program_base or self._fetch_pc >= self._program_end:
            return None
        instruction_index = (self._fetch_pc - self._program_base) // 4
        slot = _PipelineSlot(
            pc=self._fetch_pc,
            instruction=self._program[instruction_index],
            insn_id=self._next_insn_id,
        )
        self._next_insn_id += 1
        for shadow in self._control_shadows:
            shadow.delay_slots_seen = min(
                shadow.delay_slots_seen + 1,
                self.state.config.scalar.control_flow_delay_slots,
            )
        if _is_control_transfer_instruction(slot.instruction):
            self._control_shadows.append(_ControlShadow(insn_id=slot.insn_id, delay_slots_seen=0))
        self._log_fetch_start(slot, cycle)
        self._advance_fetch_pc_after_fetch(slot.pc)
        self.state.pc = self._fetch_pc
        return slot

    def _advance_if_to_id(self, slot: _PipelineSlot, cycle: int) -> _PipelineSlot:
        slot.dispatch_start_cycle = cycle
        logger = self.state.trace_logger
        if logger is not None:
            trace_cycle = self._trace_cycle(cycle)
            if slot.fetch_stage_open:
                logger.log_stage_end(slot.insn_id, "fetch", lane=FETCH_LANE, cycle=trace_cycle)
            logger.log_stage_start(slot.insn_id, "dispatch", lane=DISPATCH_LANE, cycle=trace_cycle)
        slot.fetch_stage_open = False
        return slot

    def _stall_if_slot(self, slot: _PipelineSlot, cycle: int) -> None:
        if not slot.fetch_stage_open:
            return
        logger = self.state.trace_logger
        if logger is not None:
            logger.log_stage_end(
                slot.insn_id,
                "fetch",
                lane=FETCH_LANE,
                cycle=self._trace_cycle(cycle),
            )
        slot.fetch_stage_open = False

    def _async_completion_callback(
        self,
        slot: _PipelineSlot,
        completion_cycle: int,
    ) -> Callable[[], None]:
        instruction = slot.instruction
        params = instruction.params

        if isinstance(params, MXUMatmulType):
            if not self.state.check_mreg_pair_base(params.md):
                return lambda: None
            activation = self.state.load_mreg(params.ms).clone()
            weights = self.state.load_weight_slot(0 if instruction.mnemonic.endswith("mxu0") else 1, params.ws).clone()
            scale_a = self.state.read_ereg(params.ea)
            scale_b = self.state.read_ereg(params.eb)
            mxu = 0 if instruction.mnemonic.endswith("mxu0") else 1

            def complete() -> None:
                self.state.trace_end_cycle = self._trace_cycle(completion_cycle)
                lo, hi = compute_bf16_matmul(
                    activation,
                    weights,
                    scale_a,
                    scale_b,
                    None,
                    config=self.state.config,
                )
                if self.state.check_mreg_pair_base(params.md):
                    self.state.store_mreg(params.md, lo)
                    self.state.store_mreg(params.md + 1, hi)

            return complete

        if isinstance(params, MXUMatmulAccType):
            if not self.state.check_mreg_pair_base(params.md):
                return lambda: None
            activation = self.state.load_mreg(params.ms).clone()
            weights = self.state.load_weight_slot(0 if instruction.mnemonic.endswith("mxu0") else 1, params.ws).clone()
            partial = (
                self.state.load_mreg(params.mp).clone(),
                self.state.load_mreg(params.mp + 1).clone(),
            )
            scale_a = self.state.read_ereg(params.ea)
            scale_b = self.state.read_ereg(params.eb)

            def complete() -> None:
                self.state.trace_end_cycle = self._trace_cycle(completion_cycle)
                lo, hi = compute_bf16_matmul(
                    activation,
                    weights,
                    scale_a,
                    scale_b,
                    partial,
                    config=self.state.config,
                )
                if self.state.check_mreg_pair_base(params.md):
                    self.state.store_mreg(params.md, lo)
                    self.state.store_mreg(params.md + 1, hi)

            return complete

        if isinstance(params, VPUBinaryType):
            lhs = self.state.load_mreg(params.ms1).clone()
            rhs = self.state.load_mreg(params.ms2).clone()
            op = {
                "vadd": compute_bf16_vadd,
                "vmul": compute_bf16_vmul,
                "vsub": compute_bf16_vsub,
                "vmax": compute_bf16_vmax,
                "vmin": compute_bf16_vmin,
            }[instruction.mnemonic]

            def complete() -> None:
                self.state.trace_end_cycle = self._trace_cycle(completion_cycle)
                self.state.store_mreg(params.md, op(lhs, rhs, config=self.state.config))

            return complete

        if isinstance(params, VPUUnaryType):
            src = self.state.load_mreg(params.ms).clone()
            op = {
                "vrelu": compute_bf16_vrelu,
                "vmov": compute_bf16_vmov,
                "vexp": compute_bf16_vexp,
                "vrecip": compute_bf16_vrecip,
            }[instruction.mnemonic]

            def complete() -> None:
                self.state.trace_end_cycle = self._trace_cycle(completion_cycle)
                self.state.store_mreg(params.md, op(src, config=self.state.config))

            return complete

        if isinstance(params, XLUTransposeType):
            src = self.state.load_mreg(params.ms).clone()
            op = {
                "transpose.xlu": compute_bf16_transpose,
                "reduce.max.xlu": compute_bf16_row_reduce_max,
                "reduce.sum.xlu": compute_bf16_row_reduce_sum,
            }[instruction.mnemonic]

            def complete() -> None:
                self.state.trace_end_cycle = self._trace_cycle(completion_cycle)
                self.state.store_mreg(params.md, op(src, config=self.state.config))

            return complete

        raise TypeError(f"Unsupported async instruction '{instruction.mnemonic}'")

    def _mark_step_limit_if_reached(self) -> None:
        self._step_limit_reached = (
            self._max_instructions is not None
            and self.state.perf.instructions - self._start_count >= self._max_instructions
        )

    def _issue_decode_slot(self, slot: _PipelineSlot, cycle: int) -> _PipelineSlot | None:
        instruction = slot.instruction
        if instruction.mnemonic not in ALL_INSTRUCTION_SPECS:
            raise KeyError(f"Unknown mnemonic '{instruction.mnemonic}'")
        spec = ALL_INSTRUCTION_SPECS[instruction.mnemonic]
        if not isinstance(instruction.params, spec.params_type):
            raise TypeError(
                f"{instruction.mnemonic} expects {spec.params_type.__name__}, "
                f"got {type(instruction.params).__name__}"
            )
        assert slot.dispatch_start_cycle is not None
        self.state.instruction_extra_cycles = 0
        self.state.control_transfer_set = False
        if _is_dma_wait_instruction(instruction):
            wait_channel = _dma_channel_for_instruction(instruction)
            if wait_channel is None:
                raise ValueError(f"Malformed DMA wait mnemonic '{instruction.mnemonic}'")
            retire_cycle = self.state.dma_wait_completion_cycle(wait_channel, slot.dispatch_start_cycle)
            if cycle < retire_cycle:
                return slot
            logger = self.state.trace_logger
            if logger is not None:
                trace_cycle = self._trace_cycle(cycle)
                logger.log_stage_end(slot.insn_id, "dispatch", lane=DISPATCH_LANE, cycle=trace_cycle)
                logger.log_retire(slot.insn_id, lane=DISPATCH_LANE, cycle=trace_cycle)
            self.state.trace_end_cycle = self._trace_cycle(cycle)
            self.state.perf.record_instruction(instruction.mnemonic)
            self.state.retire_dma_wait(wait_channel, retire_cycle=cycle)
            self._mark_step_limit_if_reached()
            return None
        execute_lane = _execute_lane_for_instruction(instruction)
        operand_ready_cycle = self._instruction_operand_ready_cycle(instruction)
        if cycle < max(slot.dispatch_start_cycle + 1, self._cycle_exu_ready[execute_lane], operand_ready_cycle):
            return slot
        total_latency = _instruction_latency_cycles(self.state, instruction, spec.latency)
        lane_occupancy = _lane_occupancy_cycles(instruction, execute_lane, total_latency)
        completion_cycle = cycle + total_latency
        logger = self.state.trace_logger
        if logger is not None:
            trace_cycle = self._trace_cycle(cycle)
            logger.log_stage_end(slot.insn_id, "dispatch", lane=DISPATCH_LANE, cycle=trace_cycle)
            logger.log_stage_start(slot.insn_id, "execute", lane=execute_lane, cycle=trace_cycle)
        self._cycle_exu_ready[execute_lane] = cycle + lane_occupancy
        self._reserve_instruction_destinations(instruction, completion_cycle)
        if instruction.mnemonic.startswith(("dma.load.", "dma.store.")):
            saved_pc = self._set_execute_pc(slot.pc, trace_cycle=cycle)
            spec.semantics(self.state, instruction.params)
            self._restore_fetch_pc(saved_pc)
            if logger is not None and self.state.stop_reason is None:
                channel = _dma_channel_for_instruction(instruction)
                if channel is not None:
                    transfer = self.state.dma_channels[channel].pending
                    if transfer is not None:
                        logger.log_stage_start(
                            slot.insn_id,
                            "transfer",
                            lane=_dma_transfer_lane(channel),
                            cycle=self._trace_cycle(cycle + 1),
                        )
                        logger.log_stage_end(
                            slot.insn_id,
                            "transfer",
                            lane=_dma_transfer_lane(channel),
                            cycle=self._trace_cycle(transfer.ready_cycle),
                        )

            def complete_dma_issue() -> None:
                self.state.trace_end_cycle = self._trace_cycle(completion_cycle)
                self.state.perf.record_instruction(instruction.mnemonic)
                if logger is not None:
                    logger.log_stage_end(
                        slot.insn_id,
                        "execute",
                        lane=execute_lane,
                        cycle=self._trace_cycle(completion_cycle),
                    )
                    logger.log_retire(
                        slot.insn_id,
                        lane=execute_lane,
                        cycle=self._trace_cycle(completion_cycle),
                    )
                self._mark_step_limit_if_reached()
                self._log_stop(completion_cycle)

            self._schedule_action(completion_cycle, complete_dma_issue)
            return None
        if _is_async_tensor_lane(execute_lane):
            completion = self._async_completion_callback(slot, completion_cycle)

            def complete_async() -> None:
                completion()
                self.state.perf.record_instruction(instruction.mnemonic)
                if logger is not None:
                    logger.log_stage_end(
                        slot.insn_id,
                        "execute",
                        lane=execute_lane,
                        cycle=self._trace_cycle(completion_cycle),
                    )
                    logger.log_retire(
                        slot.insn_id,
                        lane=execute_lane,
                        cycle=self._trace_cycle(completion_cycle),
                    )
                self._mark_step_limit_if_reached()
                self._log_stop(completion_cycle)

            self._schedule_action(completion_cycle, complete_async)
            return None

        def complete_blocking() -> None:
            self.state.trace_end_cycle = self._trace_cycle(completion_cycle)
            saved_pc = self._set_execute_pc(slot.pc, trace_cycle=completion_cycle)
            spec.semantics(self.state, instruction.params)
            self._consume_control_transfer(slot.insn_id)
            self._restore_fetch_pc(saved_pc)
            self.state.perf.record_instruction(instruction.mnemonic)
            if logger is not None:
                logger.log_stage_end(
                    slot.insn_id,
                    "execute",
                    lane=execute_lane,
                    cycle=self._trace_cycle(completion_cycle),
                )
                logger.log_retire(
                    slot.insn_id,
                    lane=execute_lane,
                    cycle=self._trace_cycle(completion_cycle),
                )
            self._mark_step_limit_if_reached()
            self._log_stop(completion_cycle)

        self._schedule_action(completion_cycle, complete_blocking)
        return None

    def _instruction_operand_ready_cycle(self, instruction: Instruction) -> int:
        ready_cycle = self.state.perf.cycles
        params = instruction.params

        def require_xreg(index: int) -> None:
            nonlocal ready_cycle
            if index != 0:
                ready_cycle = max(ready_cycle, self._xreg_ready[index])

        def require_ereg(index: int) -> None:
            nonlocal ready_cycle
            ready_cycle = max(ready_cycle, self._ereg_ready[index])

        def require_mreg(index: int) -> None:
            nonlocal ready_cycle
            if 0 <= index < len(self._mreg_ready):
                ready_cycle = max(ready_cycle, self._mreg_ready[index])

        def require_weight(mxu: int, slot: int) -> None:
            nonlocal ready_cycle
            ready_cycle = max(ready_cycle, self._weight_ready[mxu][slot])

        def require_vmem() -> None:
            nonlocal ready_cycle
            ready_cycle = max(ready_cycle, self._vmem_ready_cycle)

        if isinstance(params, RType):
            require_xreg(params.rs1)
            require_xreg(params.rs2)
            require_xreg(params.rd)
        elif isinstance(params, IType):
            require_xreg(params.rs1)
            require_xreg(params.rd)
            if instruction.mnemonic == "sld":
                require_vmem()
        elif isinstance(params, SType):
            require_xreg(params.rs1)
            require_xreg(params.rs2)
            require_vmem()
        elif isinstance(params, BType):
            require_xreg(params.rs1)
            require_xreg(params.rs2)
        elif isinstance(params, UType | JType):
            require_xreg(params.rd)
        elif isinstance(params, DMAType):
            require_xreg(params.dram_rs)
            require_xreg(params.vmem_rs)
            require_xreg(params.size_rs)
            require_vmem()
        elif isinstance(params, ScaleImmType):
            require_ereg(params.ed)
        elif isinstance(params, ScaleMemType):
            require_ereg(params.ed)
            require_xreg(params.rs1)
            require_vmem()
        elif isinstance(params, TensorMemType):
            require_xreg(params.rs1)
            require_mreg(params.mreg)
            require_vmem()
        elif isinstance(params, WeightMemType):
            require_xreg(params.rs1)
            require_weight(0 if instruction.mnemonic.endswith("mxu0") else 1, params.slot)
            require_vmem()
        elif isinstance(params, MXUMatmulType):
            require_mreg(params.ms)
            require_mreg(params.md)
            require_mreg(params.md + 1)
            require_weight(0 if instruction.mnemonic.endswith("mxu0") else 1, params.ws)
            require_ereg(params.ea)
            require_ereg(params.eb)
        elif isinstance(params, MXUMatmulAccType):
            require_mreg(params.ms)
            require_mreg(params.mp)
            require_mreg(params.mp + 1)
            require_mreg(params.md)
            require_mreg(params.md + 1)
            require_weight(0 if instruction.mnemonic.endswith("mxu0") else 1, params.ws)
            require_ereg(params.ea)
            require_ereg(params.eb)
        elif isinstance(params, VPUBinaryType):
            require_mreg(params.ms1)
            require_mreg(params.ms2)
            require_mreg(params.md)
        elif isinstance(params, VPUUnaryType | XLUTransposeType):
            require_mreg(params.ms)
            require_mreg(params.md)

        return ready_cycle

    def _reserve_instruction_destinations(self, instruction: Instruction, completion_cycle: int) -> None:
        params = instruction.params

        def reserve_xreg(index: int) -> None:
            if index != 0:
                self._xreg_ready[index] = completion_cycle

        def reserve_ereg(index: int) -> None:
            self._ereg_ready[index] = completion_cycle

        def reserve_mreg(index: int) -> None:
            if 0 <= index < len(self._mreg_ready):
                self._mreg_ready[index] = completion_cycle

        def reserve_weight(mxu: int, slot: int) -> None:
            self._weight_ready[mxu][slot] = completion_cycle

        def reserve_vmem(ready_cycle: int) -> None:
            self._vmem_ready_cycle = max(self._vmem_ready_cycle, ready_cycle)

        if isinstance(params, RType | IType | UType | JType):
            reserve_xreg(params.rd)
            if isinstance(params, IType) and instruction.mnemonic == "sld":
                reserve_vmem(completion_cycle)
        elif isinstance(params, ScaleImmType | ScaleMemType):
            reserve_ereg(params.ed)
            if isinstance(params, ScaleMemType):
                reserve_vmem(completion_cycle)
        elif isinstance(params, SType):
            reserve_vmem(completion_cycle)
        elif isinstance(params, TensorMemType):
            reserve_vmem(completion_cycle)
            if instruction.mnemonic == "vload":
                reserve_mreg(params.mreg)
        elif isinstance(params, WeightMemType):
            reserve_vmem(completion_cycle)
            reserve_weight(0 if instruction.mnemonic.endswith("mxu0") else 1, params.slot)
        elif isinstance(params, DMAType):
            reserve_vmem(completion_cycle)
        elif isinstance(params, MXUMatmulType | MXUMatmulAccType):
            reserve_mreg(params.md)
            reserve_mreg(params.md + 1)
        elif isinstance(params, VPUBinaryType | VPUUnaryType | XLUTransposeType):
            reserve_mreg(params.md)

    def load_program(
        self,
        program: Iterable[Instruction],
        *,
        start_pc: int | None = None,
        max_instructions: int | None = None,
        trace_logger: TraceLogger | None = None,
    ) -> None:
        instructions = list(program) if not isinstance(program, Sequence) else program
        program_base = getattr(program, "base_address", 0)
        if start_pc is None:
            start_pc = program_base
        self._program = instructions
        self._program_base = program_base
        self._program_end = program_base + len(instructions) * 4
        self._max_instructions = max_instructions
        self._start_count = self.state.perf.instructions
        self._step_limit_reached = False
        self._fetch_pc = start_pc & 0xFFFF_FFFF
        self._reset_scoreboards(self.state.perf.cycles)
        self._reset_lane_state()
        self.state.pc = self._fetch_pc
        self.state.clear_stop()
        self.state.trace_logger = trace_logger
        if trace_logger is not None:
            trace_logger.log_arch_value(
                "pc",
                0,
                self._fetch_pc,
                cycle=self._trace_cycle(self.state.perf.cycles),
            )

    def tick(self) -> bool:
        if not self._program:
            raise RuntimeError("No program loaded; call load_program() or execute() first")
        if self.state.stop_reason is not None:
            return False

        current_cycle = self.state.perf.cycles
        active_cycle = current_cycle + 1
        if self._max_instructions is not None:
            executed = self.state.perf.instructions - self._start_count
            if executed >= self._max_instructions:
                self._step_limit_reached = True
                self._if_slot = None
                self._id_slot = None
        if (
            self._step_limit_reached
            and self._if_slot is None
            and self._id_slot is None
            and not self._has_pending_actions()
        ):
            self.state.stop(StopReason.STEP_LIMIT)
            self._log_stop(current_cycle)
            return False
        if (
            self._if_slot is None
            and self._id_slot is None
            and not self._has_pending_actions()
            and self._fetch_pc >= self._program_end
        ):
            self.state.stop(StopReason.PROGRAM_END)
            self._log_stop(current_cycle)
            return False

        next_cycle = current_cycle + 1
        self.state.perf.cycles = active_cycle
        self._run_actions_for_cycle(active_cycle)

        if self.state.stop_reason is not None:
            self._if_slot = None
            self._id_slot = None
            self._log_stop(active_cycle)
            return False

        current_if_slot = self._if_slot
        current_id_slot = self._id_slot
        next_if_slot = current_if_slot
        next_id_slot = current_id_slot
        frontend_fenced = (
            current_id_slot is not None and _is_dma_wait_instruction(current_id_slot.instruction)
        )

        if current_id_slot is not None:
            next_id_slot = self._issue_decode_slot(current_id_slot, active_cycle)

        if self._step_limit_reached:
            next_if_slot = None
            next_id_slot = None
        else:
            if current_if_slot is not None and next_id_slot is None:
                next_id_slot = self._advance_if_to_id(current_if_slot, active_cycle)
                next_if_slot = None

            if next_if_slot is None and not frontend_fenced:
                fetched_slot = self._fetch_instruction(active_cycle)
                if fetched_slot is not None:
                    next_if_slot = fetched_slot

        if next_if_slot is current_if_slot and current_if_slot is not None:
            self._stall_if_slot(current_if_slot, active_cycle)

        self._if_slot = next_if_slot
        self._id_slot = next_id_slot
        self.state.pc = self._fetch_pc
        if (
            self._step_limit_reached
            and self._if_slot is None
            and self._id_slot is None
            and not self._has_pending_actions()
        ):
            self.state.stop(StopReason.STEP_LIMIT)
            self._log_stop(active_cycle)
            return False
        if (
            self._if_slot is None
            and self._id_slot is None
            and not self._has_pending_actions()
            and self._fetch_pc >= self._program_end
        ):
            self.state.stop(StopReason.PROGRAM_END)
            self._log_stop(active_cycle)
            return False
        return True

    def execute_instruction(self, instruction: Instruction) -> None:
        self.execute([instruction])

    def execute(
        self,
        program: Iterable[Instruction],
        *,
        start_pc: int | None = None,
        max_instructions: int | None = None,
        trace_logger: TraceLogger | None = None,
    ) -> PerformanceCounters:
        if trace_logger is None:
            auto_trace_path = _pytest_auto_trace_path()
            if auto_trace_path is not None:
                from .logging import TraceLogger, TraceLoggerConfig

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

        if self.state.pc % 4 != 0:
            self.state.stop(StopReason.INSTRUCTION_ADDRESS_MISALIGNED)
        while self.state.stop_reason is None:
            self.tick()

        self.state.trace_logger = None
        return self.state.perf

    def dump_json_trace(
        self,
        program: Iterable[Instruction],
        trace_path: str | PathLike[str],
        *,
        start_pc: int | None = None,
        max_instructions: int | None = None,
    ) -> PerformanceCounters:
        from .logging import TraceLogger, TraceLoggerConfig

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
    return trace_root / f"{sanitized}__{count:02d}.json"


__all__ = [
    "ArchState",
    "INSTRUCTION_LATENCY",
    "PerformanceCounters",
    "Sim",
    "StopReason",
]
