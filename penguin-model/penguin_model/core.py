"""Cycle-accurate Penguin core wiring and scheduling."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from .arch_state import ArchState, PerformanceCounters, StopReason
from .core_config import DEFAULT_PENGUIN_CORE_CONFIG, PenguinCoreConfig
from .exu import (
    DmaExecutionUnit,
    MatrixExecutionUnit,
    ScalarExecutionUnit,
    TensorMemoryExecutionUnit,
    VectorExecutionUnit,
    XLUExecutionUnit,
)
from .idu import InstructionDecode
from .ifu import InstructionFetch
from .instructions import (
    ALL_INSTRUCTION_SPECS,
    BType,
    DMAControlType,
    DMAType,
    DelayType,
    EmptyType,
    IType,
    Instruction,
    JType,
    MXUAccumulatorType,
    MXUMatmulAccType,
    MXUMatmulType,
    RType,
    ScaleImmType,
    ScaleMemType,
    SType,
    TensorMemType,
    UType,
    VectorImmType,
    VPUBinaryType,
    VPUUnaryType,
    WeightMemType,
    WeightTensorType,
    XLUUnaryType,
)
from .isa import SCALAR_LOAD_MNEMONICS
from .memory import Memory
from .tensor import (
    accum_tile_from_bytes,
    accum_tile_to_bytes,
    bf16_tile_pair_from_bytes,
    bf16_tile_pair_to_bytes,
    compute_accum_matmul,
    compute_bf16_row_reduce_max,
    compute_bf16_row_reduce_sum,
    compute_bf16_transpose,
    compute_bf16_vadd,
    compute_bf16_vredsum,
    compute_bf16_vexp,
    compute_bf16_vmax,
    compute_bf16_vmin,
    compute_bf16_vmov,
    compute_bf16_vrecip,
    compute_bf16_vrelu,
    compute_bf16_vsub,
    compute_bf16_vmul,
    export_accum_to_fp8,
    fp8_tile_from_bytes,
    compute_vector_immediate_fill,
)
from .uop import PipelineUop

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

_UNIT_LANES = {
    "salu": SALU_LANE,
    "dma": DMA_LANE,
    "tmem": TMEM_LANE,
    "mxu0": MXU0_LANE,
    "mxu1": MXU1_LANE,
    "vpu": VPU_LANE,
    "xlu": XLU_LANE,
}
_UNIT_CLASSES = {
    "salu": ScalarExecutionUnit,
    "dma": DmaExecutionUnit,
    "tmem": TensorMemoryExecutionUnit,
    "mxu0": MatrixExecutionUnit,
    "mxu1": MatrixExecutionUnit,
    "vpu": VectorExecutionUnit,
    "xlu": XLUExecutionUnit,
}


def _format_instruction(instruction: Instruction) -> str:
    params = instruction.params
    mnemonic = instruction.mnemonic

    if isinstance(params, RType):
        return f"{mnemonic} x{params.rd}, x{params.rs1}, x{params.rs2}"
    if isinstance(params, IType):
        if mnemonic in SCALAR_LOAD_MNEMONICS:
            return f"{mnemonic} x{params.rd}, {params.imm}(x{params.rs1})"
        if mnemonic == "jalr":
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
    if isinstance(params, DelayType):
        return f"{mnemonic} {params.cycles}"
    if isinstance(params, DMAType):
        return f"{mnemonic} x{params.rd}, x{params.rs1}, x{params.rs2}"
    if isinstance(params, DMAControlType):
        return f"{mnemonic} x{params.rs1}"
    if isinstance(params, ScaleImmType):
        return f"{mnemonic} e{params.ed}, {params.imm}"
    if isinstance(params, ScaleMemType):
        return f"{mnemonic} e{params.ed}, {params.imm}(x{params.rs1})"
    if isinstance(params, TensorMemType):
        return f"{mnemonic} m{params.mreg}, {params.imm}(x{params.rs1})"
    if isinstance(params, WeightMemType):
        return f"{mnemonic} w{params.slot}, {params.imm}(x{params.rs1})"
    if isinstance(params, WeightTensorType):
        return f"{mnemonic} w{params.slot}, m{params.ms}"
    if isinstance(params, MXUAccumulatorType):
        return f"{mnemonic} m{params.mreg}"
    if isinstance(params, MXUMatmulType):
        return f"{mnemonic} m{params.ms}, w{params.ws}"
    if isinstance(params, MXUMatmulAccType):
        return f"{mnemonic} m{params.ms}, w{params.ws}"
    if isinstance(params, VPUBinaryType):
        return f"{mnemonic} m{params.md}, m{params.ms1}, m{params.ms2}"
    if isinstance(params, VPUUnaryType):
        return f"{mnemonic} m{params.md}, m{params.ms}"
    if isinstance(params, VectorImmType):
        return f"{mnemonic} m{params.md}, {params.imm}"
    if isinstance(params, XLUUnaryType):
        return f"{mnemonic} m{params.md}, m{params.ms}"
    return f"{mnemonic} {params}"


def _execute_lane_for_instruction(instruction: Instruction) -> int:
    if instruction.mnemonic.startswith("dma."):
        return DMA_LANE
    if instruction.mnemonic in {"vload", "vstore"}:
        return TMEM_LANE
    if instruction.mnemonic.endswith(".mxu0"):
        return MXU0_LANE
    if instruction.mnemonic.endswith(".mxu1"):
        return MXU1_LANE
    if isinstance(instruction.params, (VPUBinaryType, VPUUnaryType)):
        return VPU_LANE
    if isinstance(instruction.params, XLUUnaryType):
        return XLU_LANE
    return SALU_LANE


def _unit_key_for_instruction(instruction: Instruction) -> str:
    lane = _execute_lane_for_instruction(instruction)
    for unit_key, unit_lane in _UNIT_LANES.items():
        if unit_lane == lane:
            return unit_key
    raise KeyError(f"No execution unit for lane {lane}")


def _dma_channel_for_instruction(instruction: Instruction) -> int | None:
    parts = instruction.mnemonic.split(".")
    if len(parts) != 3 or parts[0] != "dma":
        return None
    if not parts[2].startswith("ch"):
        return None
    return int(parts[2][2:])


def _is_dma_wait_instruction(instruction: Instruction) -> bool:
    return instruction.mnemonic.startswith("dma.wait.")


def _is_delay_instruction(instruction: Instruction) -> bool:
    return instruction.mnemonic == "delay"


def _is_decode_fence_instruction(instruction: Instruction) -> bool:
    return _is_dma_wait_instruction(instruction) or _is_delay_instruction(instruction)


def _dma_transfer_lane(channel: int) -> int:
    return DMA_TRANSFER_LANE_BASE + channel


def _is_async_tensor_lane(lane: int) -> bool:
    return lane in {MXU0_LANE, MXU1_LANE, VPU_LANE, XLU_LANE}


def _lane_occupancy_cycles(instruction: Instruction, execute_lane: int, total_latency: int) -> int:
    if execute_lane == DMA_LANE and not _is_dma_wait_instruction(instruction):
        return 1
    if instruction.mnemonic in {"vload", "vstore"}:
        return 1
    if instruction.mnemonic.startswith(
        (
            "vmatpush.weight.",
            "vload.weight.",
            "vmatpush.acc.fp8.",
            "vmatpush.acc.bf16.",
            "vmatpop.bf16.acc.",
            "vmatpop.fp8.acc.",
        )
    ):
        return 1
    if isinstance(instruction.params, VectorImmType):
        return 1
    if isinstance(instruction.params, VPUBinaryType):
        return 1
    if isinstance(instruction.params, VPUUnaryType):
        if instruction.mnemonic in {"vexp", "vrecip.bf16", "vrecip"}:
            return total_latency
        return 1
    if isinstance(instruction.params, XLUUnaryType):
        return 1
    if _is_async_tensor_lane(execute_lane):
        return total_latency
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
    if instruction.mnemonic.startswith("vmatpush.weight."):
        return state.config.vmatpush_weight_latency_cycles
    if instruction.mnemonic.startswith("vload.weight."):
        return state.config.vload_weight_latency_cycles
    if instruction.mnemonic.startswith("vmatpush.acc.bf16."):
        return state.config.vmatpush_acc_latency_cycles
    if instruction.mnemonic.startswith("vmatpush.acc.fp8."):
        return state.config.vmatpop_acc_fp8_latency_cycles
    if instruction.mnemonic.startswith("vmatpop.bf16.acc."):
        return state.config.vmatpop_acc_bf16_latency_cycles
    if instruction.mnemonic.startswith("vmatpop.fp8.acc."):
        return state.config.vmatpop_acc_fp8_latency_cycles
    if instruction.mnemonic.startswith("vmatmul"):
        return state.config.matmul_latency_cycles
    if isinstance(instruction.params, VectorImmType):
        return state.config.vpu_simple_op_latency_cycles
    if isinstance(instruction.params, VPUBinaryType):
        return state.config.vpu_simple_op_latency_cycles
    if isinstance(instruction.params, VPUUnaryType):
        if instruction.mnemonic in {"vexp", "vrecip.bf16", "vrecip"}:
            return state.config.vpu_non_pipelineable_op_latency_cycles
        return state.config.vpu_simple_op_latency_cycles
    if isinstance(instruction.params, XLUUnaryType):
        return state.config.xlu_transpose_latency_cycles
    return default_latency


@dataclass(slots=True)
class _ControlShadow:
    insn_id: int
    pc: int
    delay_slots_seen: int
    resolved: bool = False
    target: int | None = None


def _is_control_transfer_instruction(instruction: Instruction) -> bool:
    return isinstance(instruction.params, (BType, JType)) or instruction.mnemonic == "jalr"


class Core:
    """Cycle-accurate Penguin core built from explicit IFU/IDU/EXU blocks."""

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
        self._program_loaded = False
        self._program: Sequence[Instruction] = ()
        self._program_base = 0
        self._program_end = 0
        self._max_instructions: int | None = None
        self._start_count = 0
        self._step_limit_reached = False
        self._stop_logged = False
        self._control_shadows: list[_ControlShadow] = []

        self._ifu = InstructionFetch()
        self._idu = InstructionDecode(tuple(_UNIT_LANES))
        self._exus = {
            unit_key: _UNIT_CLASSES[unit_key](
                name=unit_key.upper(),
                lane_id=lane,
                logger=None,
            )
            for unit_key, lane in _UNIT_LANES.items()
        }

        self._reset_scoreboards(self.state.perf.cycles)
        self._reset_pipeline_state(self.state.perf.cycles)

    @property
    def config(self) -> PenguinCoreConfig:
        return self.state.config

    @property
    def memory(self) -> Memory:
        return self.state.vmem

    @property
    def perf(self) -> PerformanceCounters:
        return self.state.perf

    def _reset_pipeline_state(self, base_cycle: int) -> None:
        self._control_shadows = []
        self._step_limit_reached = False
        self._stop_logged = False
        self._unit_next_available_cycle = {unit_key: base_cycle for unit_key in _UNIT_LANES}
        self._dma_scheduled_ready_cycle = [0] * self.state.config.dma.channel_count
        self._dma_scheduled_valid = [False] * self.state.config.dma.channel_count
        self._ifu.reset()
        self._idu.reset()
        for exu in self._exus.values():
            exu.reset()

    def _reset_scoreboards(self, base_cycle: int) -> None:
        self._xreg_ready = [base_cycle] * self.state.config.scalar.xreg_count
        self._xreg_read_consumed = [base_cycle] * self.state.config.scalar.xreg_count
        self._ereg_ready = [base_cycle] * self.state.config.scale.num_ereg
        self._ereg_read_consumed = [base_cycle] * self.state.config.scale.num_ereg
        self._mreg_ready = [base_cycle] * self.state.config.tensor.num_mreg
        self._mreg_read_consumed = [base_cycle] * self.state.config.tensor.num_mreg
        self._accum_ready = [base_cycle] * self.state.config.tensor.mxu_count
        self._accum_read_consumed = [base_cycle] * self.state.config.tensor.mxu_count
        self._weight_ready = [
            [base_cycle] * self.state.config.tensor.weight_slots_per_mxu
            for _ in range(self.state.config.tensor.mxu_count)
        ]
        self._weight_read_consumed = [
            [base_cycle] * self.state.config.tensor.weight_slots_per_mxu
            for _ in range(self.state.config.tensor.mxu_count)
        ]
        self._vmem_ready_cycle = base_cycle
        self._xreg_ready[0] = base_cycle
        self._xreg_read_consumed[0] = base_cycle

    def _set_trace_logger(self, trace_logger: TraceLogger | None) -> None:
        self.state.trace_logger = trace_logger
        for exu in self._exus.values():
            exu.logger = trace_logger

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
            mxu_accum=self.state.mxu_accum,
            ereg=self.state.ereg,
        )
        self._program_loaded = False
        self._program = ()
        self._program_base = 0
        self._program_end = 0
        self._max_instructions = None
        self._start_count = 0
        self._reset_scoreboards(self.state.perf.cycles)
        self._reset_pipeline_state(self.state.perf.cycles)
        self._set_trace_logger(None)

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
        trace_ts = self._trace_cycle(trace_cycle)
        self.state.trace_start_cycle = trace_ts
        self.state.trace_end_cycle = trace_ts
        self.state.control_transfer_set = False
        return saved_pc

    def _restore_fetch_pc(self, saved_pc: int) -> None:
        del saved_pc
        self.state.pc = self._ifu.fetch_pc

    def _refresh_fetch_redirect(self) -> None:
        if not self._control_shadows:
            return
        youngest = self._control_shadows[-1]
        if not youngest.resolved or youngest.target is None:
            return
        if youngest.delay_slots_seen < self.state.config.scalar.control_flow_delay_slots:
            return
        buffered = self._ifu.output.peek()
        if (
            buffered is not None
            and buffered.pc > youngest.pc + self.state.config.scalar.control_flow_delay_slots
        ):
            self._ifu.output.reset()
        self._ifu.set_fetch_pc(youngest.target)
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

    def _on_fetch(self, uop: PipelineUop, cycle: int) -> None:
        for shadow in self._control_shadows:
            shadow.delay_slots_seen = min(
                shadow.delay_slots_seen + 1,
                self.state.config.scalar.control_flow_delay_slots,
            )
        if _is_control_transfer_instruction(uop.instruction):
            if any(
                shadow.delay_slots_seen <= self.state.config.scalar.control_flow_delay_slots
                for shadow in self._control_shadows
            ):
                self.state.stop(StopReason.ILLEGAL_INSTRUCTION)
                return
            self._control_shadows.append(
                _ControlShadow(insn_id=uop.insn_id, pc=uop.pc, delay_slots_seen=0)
            )

        logger = self.state.trace_logger
        if logger is None:
            return
        trace_cycle = self._trace_cycle(cycle)
        logger.log_insn(uop.insn_id, _format_instruction(uop.instruction))
        logger.log_arch_value("pc", 0, uop.pc, cycle=trace_cycle)
        logger.log_stage_start(uop.insn_id, "fetch", lane=FETCH_LANE, cycle=trace_cycle)

    def _on_fetch_stall(self, uop: PipelineUop, cycle: int) -> None:
        if not uop.fetch_stage_open or self.state.trace_logger is None:
            return
        self.state.trace_logger.log_stage_end(
            uop.insn_id,
            "fetch",
            lane=FETCH_LANE,
            cycle=self._trace_cycle(cycle),
        )
        uop.fetch_stage_open = False

    def _on_misaligned_fetch(self) -> None:
        self.state.stop(StopReason.INSTRUCTION_ADDRESS_MISALIGNED)

    def _on_fetch_pc_advanced(self, fetched_pc: int) -> None:
        del fetched_pc
        self._refresh_fetch_redirect()
        self.state.pc = self._ifu.fetch_pc

    def _on_claim_from_ifu(self, uop: PipelineUop, cycle: int) -> None:
        logger = self.state.trace_logger
        if logger is None:
            return
        trace_cycle = self._trace_cycle(cycle)
        if uop.fetch_stage_open:
            logger.log_stage_end(uop.insn_id, "fetch", lane=FETCH_LANE, cycle=trace_cycle)
            uop.fetch_stage_open = False
        logger.log_stage_start(uop.insn_id, "dispatch", lane=DISPATCH_LANE, cycle=trace_cycle)

    def _mark_step_limit_if_reached(self) -> None:
        self._step_limit_reached = (
            self._max_instructions is not None
            and self.state.perf.instructions - self._start_count >= self._max_instructions
        )

    def _squash_frontend_for_step_limit(self) -> None:
        self._ifu.output.reset()
        self._idu.current_uop = None

    def _instruction_operand_ready_cycle(self, instruction: Instruction) -> int:
        params = instruction.params
        ready_cycle = self.state.perf.cycles

        def wait_xreg(index: int) -> None:
            nonlocal ready_cycle
            ready_cycle = max(ready_cycle, self._xreg_ready[index])

        def wait_xreg_write(index: int) -> None:
            nonlocal ready_cycle
            if index != 0:
                ready_cycle = max(ready_cycle, self._xreg_read_consumed[index])

        def wait_ereg(index: int) -> None:
            nonlocal ready_cycle
            ready_cycle = max(ready_cycle, self._ereg_ready[index])

        def wait_ereg_write(index: int) -> None:
            nonlocal ready_cycle
            ready_cycle = max(ready_cycle, self._ereg_read_consumed[index])

        def wait_mreg(index: int) -> None:
            nonlocal ready_cycle
            if 0 <= index < len(self._mreg_ready):
                ready_cycle = max(ready_cycle, self._mreg_ready[index])

        def wait_mreg_write(index: int) -> None:
            nonlocal ready_cycle
            if 0 <= index < len(self._mreg_read_consumed):
                ready_cycle = max(ready_cycle, self._mreg_read_consumed[index])

        def wait_weight(mxu: int, slot: int) -> None:
            nonlocal ready_cycle
            ready_cycle = max(ready_cycle, self._weight_ready[mxu][slot])

        def wait_weight_write(mxu: int, slot: int) -> None:
            nonlocal ready_cycle
            ready_cycle = max(ready_cycle, self._weight_read_consumed[mxu][slot])

        def wait_accum(mxu: int) -> None:
            nonlocal ready_cycle
            ready_cycle = max(ready_cycle, self._accum_ready[mxu])

        def wait_accum_write(mxu: int) -> None:
            nonlocal ready_cycle
            ready_cycle = max(ready_cycle, self._accum_read_consumed[mxu])

        def wait_vmem() -> None:
            nonlocal ready_cycle
            ready_cycle = max(ready_cycle, self._vmem_ready_cycle)

        if isinstance(params, RType):
            wait_xreg(params.rs1)
            wait_xreg(params.rs2)
            wait_xreg_write(params.rd)
        elif isinstance(params, IType):
            wait_xreg(params.rs1)
            wait_xreg_write(params.rd)
            if instruction.mnemonic in SCALAR_LOAD_MNEMONICS:
                wait_vmem()
        elif isinstance(params, SType):
            wait_xreg(params.rs1)
            wait_xreg(params.rs2)
            wait_vmem()
        elif isinstance(params, BType):
            wait_xreg(params.rs1)
            wait_xreg(params.rs2)
        elif isinstance(params, JType):
            wait_xreg_write(params.rd)
        elif isinstance(params, UType):
            wait_xreg_write(params.rd)
        elif isinstance(params, ScaleMemType):
            wait_xreg(params.rs1)
            wait_vmem()
            wait_ereg_write(params.ed)
        elif isinstance(params, ScaleImmType):
            wait_ereg_write(params.ed)
        elif isinstance(params, TensorMemType):
            wait_xreg(params.rs1)
            wait_vmem()
            if instruction.mnemonic == "vstore":
                wait_mreg(params.mreg)
            else:
                wait_mreg_write(params.mreg)
        elif isinstance(params, WeightMemType):
            wait_xreg(params.rs1)
            wait_vmem()
            wait_weight_write(0 if instruction.mnemonic.endswith("mxu0") else 1, params.slot)
        elif isinstance(params, WeightTensorType):
            wait_mreg(params.ms)
            wait_weight_write(0 if instruction.mnemonic.endswith("mxu0") else 1, params.slot)
        elif isinstance(params, MXUAccumulatorType):
            mxu = 0 if instruction.mnemonic.endswith("mxu0") else 1
            wait_accum(mxu)
            if instruction.mnemonic.startswith(("vmatpush.acc.bf16.", "vmatpush.bf16.acc.")):
                wait_mreg(params.mreg)
                wait_mreg(params.mreg + 1)
                wait_accum_write(mxu)
            elif instruction.mnemonic.startswith("vmatpush.acc.fp8."):
                wait_mreg(params.mreg)
                wait_accum_write(mxu)
            elif instruction.mnemonic.startswith("vmatpop.bf16.acc."):
                wait_mreg_write(params.mreg)
                wait_mreg_write(params.mreg + 1)
            elif instruction.mnemonic.startswith("vmatpop.fp8.acc."):
                wait_mreg_write(params.mreg)
        elif isinstance(params, MXUMatmulType | MXUMatmulAccType):
            mxu = 0 if instruction.mnemonic.endswith("mxu0") else 1
            wait_mreg(params.ms)
            wait_weight(mxu, params.ws)
            wait_accum_write(mxu)
            if isinstance(params, MXUMatmulAccType):
                wait_accum(mxu)
        elif isinstance(params, VPUBinaryType):
            wait_mreg(params.ms1)
            wait_mreg(params.ms2)
            wait_mreg_write(params.md)
        elif isinstance(params, VPUUnaryType):
            wait_mreg(params.ms)
            wait_mreg_write(params.md)
        elif isinstance(params, VectorImmType):
            wait_mreg_write(params.md)
        elif isinstance(params, XLUUnaryType):
            wait_mreg(params.ms)
            wait_mreg_write(params.md)
        elif isinstance(params, DMAType):
            wait_xreg(params.rd)
            wait_xreg(params.rs1)
            wait_xreg(params.rs2)
            wait_vmem()
        elif isinstance(params, DMAControlType):
            wait_xreg(params.rs1)
            wait_vmem()

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

        def reserve_accum(mxu: int) -> None:
            self._accum_ready[mxu] = completion_cycle

        def reserve_vmem(ready_cycle: int) -> None:
            self._vmem_ready_cycle = max(self._vmem_ready_cycle, ready_cycle)

        if isinstance(params, RType | IType | UType | JType):
            reserve_xreg(params.rd)
            if isinstance(params, IType) and instruction.mnemonic in SCALAR_LOAD_MNEMONICS:
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
        elif isinstance(params, WeightTensorType):
            reserve_weight(0 if instruction.mnemonic.endswith("mxu0") else 1, params.slot)
        elif isinstance(params, MXUAccumulatorType):
            reserve_accum(0 if instruction.mnemonic.endswith("mxu0") else 1)
            if instruction.mnemonic.startswith("vmatpop.bf16.acc."):
                reserve_mreg(params.mreg)
                reserve_mreg(params.mreg + 1)
            elif instruction.mnemonic.startswith("vmatpop.fp8.acc."):
                reserve_mreg(params.mreg)
        elif isinstance(params, MXUMatmulType | MXUMatmulAccType):
            reserve_accum(0 if instruction.mnemonic.endswith("mxu0") else 1)
        elif isinstance(params, DMAType):
            reserve_vmem(completion_cycle)
        elif isinstance(params, DMAControlType):
            pass
        elif isinstance(params, VPUBinaryType | VPUUnaryType | VectorImmType | XLUUnaryType):
            reserve_mreg(params.md)

    def _reserve_instruction_reads(self, instruction: Instruction, execute_start_cycle: int) -> None:
        params = instruction.params

        def reserve_xreg(index: int) -> None:
            self._xreg_read_consumed[index] = max(self._xreg_read_consumed[index], execute_start_cycle)

        def reserve_ereg(index: int) -> None:
            self._ereg_read_consumed[index] = max(self._ereg_read_consumed[index], execute_start_cycle)

        def reserve_mreg(index: int) -> None:
            if 0 <= index < len(self._mreg_read_consumed):
                self._mreg_read_consumed[index] = max(
                    self._mreg_read_consumed[index], execute_start_cycle
                )

        def reserve_weight(mxu: int, slot: int) -> None:
            self._weight_read_consumed[mxu][slot] = max(
                self._weight_read_consumed[mxu][slot], execute_start_cycle
            )

        def reserve_accum(mxu: int) -> None:
            self._accum_read_consumed[mxu] = max(
                self._accum_read_consumed[mxu], execute_start_cycle
            )

        if isinstance(params, RType):
            reserve_xreg(params.rs1)
            reserve_xreg(params.rs2)
        elif isinstance(params, IType):
            reserve_xreg(params.rs1)
        elif isinstance(params, SType):
            reserve_xreg(params.rs1)
            reserve_xreg(params.rs2)
        elif isinstance(params, BType):
            reserve_xreg(params.rs1)
            reserve_xreg(params.rs2)
        elif isinstance(params, ScaleMemType):
            reserve_xreg(params.rs1)
        elif isinstance(params, TensorMemType):
            reserve_xreg(params.rs1)
            if instruction.mnemonic == "vstore":
                reserve_mreg(params.mreg)
        elif isinstance(params, WeightMemType):
            reserve_xreg(params.rs1)
        elif isinstance(params, WeightTensorType):
            reserve_mreg(params.ms)
        elif isinstance(params, MXUAccumulatorType):
            mxu = 0 if instruction.mnemonic.endswith("mxu0") else 1
            reserve_accum(mxu)
            if instruction.mnemonic.startswith(("vmatpush.acc.bf16.", "vmatpush.bf16.acc.")):
                reserve_mreg(params.mreg)
                reserve_mreg(params.mreg + 1)
            elif instruction.mnemonic.startswith("vmatpush.acc.fp8."):
                reserve_mreg(params.mreg)
        elif isinstance(params, MXUMatmulType | MXUMatmulAccType):
            mxu = 0 if instruction.mnemonic.endswith("mxu0") else 1
            reserve_mreg(params.ms)
            reserve_weight(mxu, params.ws)
            if isinstance(params, MXUMatmulAccType):
                reserve_accum(mxu)
        elif isinstance(params, VPUBinaryType):
            reserve_mreg(params.ms1)
            reserve_mreg(params.ms2)
        elif isinstance(params, VPUUnaryType):
            reserve_mreg(params.ms)
        elif isinstance(params, XLUUnaryType):
            reserve_mreg(params.ms)
        elif isinstance(params, DMAType):
            reserve_xreg(params.rd)
            reserve_xreg(params.rs1)
            reserve_xreg(params.rs2)
        elif isinstance(params, DMAControlType):
            reserve_xreg(params.rs1)

    def _build_async_callbacks(
        self,
        uop: PipelineUop,
        completion_cycle: int,
    ) -> tuple[Callable[[int], None], Callable[[int], None]]:
        instruction = uop.instruction
        params = instruction.params

        if isinstance(params, WeightTensorType):
            captured: dict[str, object] = {}
            mxu = 0 if instruction.mnemonic.endswith("mxu0") else 1

            def on_start(cycle: int) -> None:
                del cycle
                captured["payload"] = self.state.load_mreg(params.ms).clone()

            def on_complete(cycle: int) -> None:
                if "payload" not in captured:
                    return
                self.state.trace_end_cycle = self._trace_cycle(cycle)
                self.state.store_weight_slot(mxu, params.slot, captured["payload"])

            return on_start, on_complete

        if isinstance(params, WeightMemType):
            captured: dict[str, object] = {}
            mxu = 0 if instruction.mnemonic.endswith("mxu0") else 1

            def on_start(cycle: int) -> None:
                del cycle
                address = self.state.resolve_indirect_address(params.rs1, params.imm)
                if not self.state._check_tensor_alignment(address):
                    return
                captured["address"] = address
                captured["payload"] = self.state.vmem.read(
                    address,
                    self.state.config.weight_slot_bytes,
                ).clone()

            def on_complete(cycle: int) -> None:
                if "payload" not in captured:
                    return
                self.state.trace_end_cycle = self._trace_cycle(cycle)
                self.state.perf.bytes_read += self.state.config.weight_slot_bytes
                self.state.store_weight_slot(mxu, params.slot, captured["payload"])
                self.state._log_memory_access(
                    "vmem",
                    f"vload.weight.mxu{mxu}",
                    captured["address"],
                    params.slot,
                    size=self.state.config.weight_slot_bytes,
                )

            return on_start, on_complete

        if isinstance(params, MXUAccumulatorType):
            captured: dict[str, object] = {}
            mxu = 0 if instruction.mnemonic.endswith("mxu0") else 1

            def on_start(cycle: int) -> None:
                del cycle
                if instruction.mnemonic.startswith(("vmatpush.acc.bf16.", "vmatpush.bf16.acc.")):
                    if not self.state.check_even_mreg_pair_base(params.mreg):
                        return
                    captured["payload"] = (
                        self.state.load_mreg(params.mreg).clone(),
                        self.state.load_mreg(params.mreg + 1).clone(),
                    )
                elif instruction.mnemonic.startswith("vmatpush.acc.fp8."):
                    captured["payload"] = self.state.load_mreg(params.mreg).clone()
                elif instruction.mnemonic.startswith("vmatpop.bf16.acc."):
                    if not self.state.check_even_mreg_pair_base(params.mreg):
                        return
                    captured["payload"] = self.state.load_accum_buffer(mxu)
                else:
                    captured["payload"] = self.state.load_accum_buffer(mxu)

            def on_complete(cycle: int) -> None:
                if "payload" not in captured:
                    return
                self.state.trace_end_cycle = self._trace_cycle(cycle)
                if instruction.mnemonic.startswith(("vmatpush.acc.bf16.", "vmatpush.bf16.acc.")):
                    raw_lo, raw_hi = captured["payload"]
                    self.state.store_accum_buffer(
                        mxu,
                        accum_tile_to_bytes(
                            bf16_tile_pair_from_bytes(
                                raw_lo,
                                raw_hi,
                                config=self.state.config,
                            ),
                            config=self.state.config,
                        ),
                    )
                elif instruction.mnemonic.startswith("vmatpush.acc.fp8."):
                    self.state.store_accum_buffer(
                        mxu,
                        accum_tile_to_bytes(
                            fp8_tile_from_bytes(captured["payload"], config=self.state.config).to(
                                torch.bfloat16
                            ),
                            config=self.state.config,
                        ),
                    )
                elif instruction.mnemonic.startswith("vmatpop.bf16.acc."):
                    raw_lo, raw_hi = bf16_tile_pair_to_bytes(
                        accum_tile_from_bytes(captured["payload"], config=self.state.config),
                        config=self.state.config,
                    )
                    self.state.store_mreg(
                        params.mreg,
                        raw_lo,
                    )
                    self.state.store_mreg(
                        params.mreg + 1,
                        raw_hi,
                    )
                else:
                    self.state.store_mreg(
                        params.mreg,
                        export_accum_to_fp8(captured["payload"], config=self.state.config),
                    )

            return on_start, on_complete

        if isinstance(params, MXUMatmulType | MXUMatmulAccType):
            captured: dict[str, object] = {}
            mxu = 0 if instruction.mnemonic.endswith("mxu0") else 1
            accumulates = instruction.mnemonic.startswith("vmatmul.acc.")

            def on_start(cycle: int) -> None:
                del cycle
                captured["activation"] = self.state.load_mreg(params.ms).clone()
                captured["weights"] = self.state.load_weight_slot(mxu, params.ws).clone()
                if accumulates:
                    captured["accum"] = self.state.load_accum_buffer(mxu).clone()

            def on_complete(cycle: int) -> None:
                if "activation" not in captured:
                    return
                self.state.trace_end_cycle = self._trace_cycle(cycle)
                self.state.store_accum_buffer(
                    mxu,
                    compute_accum_matmul(
                        captured["activation"],
                        captured["weights"],
                        captured.get("accum"),
                        config=self.state.config,
                    ),
                )

            return on_start, on_complete

        if isinstance(params, VPUBinaryType):
            captured: dict[str, object] = {}
            op = {
                "vadd": compute_bf16_vadd,
                "vadd.bf16": compute_bf16_vadd,
                "vmul": compute_bf16_vmul,
                "vmul.bf16": compute_bf16_vmul,
                "vsub": compute_bf16_vsub,
                "vsub.bf16": compute_bf16_vsub,
                "vmax": compute_bf16_vmax,
                "vmax.bf16": compute_bf16_vmax,
                "vmin": compute_bf16_vmin,
                "vmin.bf16": compute_bf16_vmin,
            }[instruction.mnemonic]

            def on_start(cycle: int) -> None:
                del cycle
                captured["lhs"] = self.state.load_mreg(params.ms1).clone()
                captured["rhs"] = self.state.load_mreg(params.ms2).clone()

            def on_complete(cycle: int) -> None:
                if "lhs" not in captured:
                    return
                self.state.trace_end_cycle = self._trace_cycle(completion_cycle)
                self.state.store_mreg(
                    params.md,
                    op(captured["lhs"], captured["rhs"], config=self.state.config),
                )

            return on_start, on_complete

        if isinstance(params, VPUUnaryType):
            captured: dict[str, object] = {}
            op = {
                "vredsum.bf16": compute_bf16_vredsum,
                "vrelu": compute_bf16_vrelu,
                "vmov": compute_bf16_vmov,
                "vexp": compute_bf16_vexp,
                "vrecip": compute_bf16_vrecip,
                "vrecip.bf16": compute_bf16_vrecip,
            }[instruction.mnemonic]

            def on_start(cycle: int) -> None:
                del cycle
                captured["src"] = self.state.load_mreg(params.ms).clone()

            def on_complete(cycle: int) -> None:
                if "src" not in captured:
                    return
                self.state.trace_end_cycle = self._trace_cycle(completion_cycle)
                self.state.store_mreg(params.md, op(captured["src"], config=self.state.config))

            return on_start, on_complete

        if isinstance(params, VectorImmType):
            def on_start(cycle: int) -> None:
                del cycle

            def on_complete(cycle: int) -> None:
                self.state.trace_end_cycle = self._trace_cycle(completion_cycle)
                self.state.store_mreg(
                    params.md,
                    compute_vector_immediate_fill(params.imm, mode=instruction.mnemonic.split(".")[1], config=self.state.config),
                )

            return on_start, on_complete

        if isinstance(params, XLUUnaryType):
            captured: dict[str, object] = {}
            op = {
                "transpose.xlu": compute_bf16_transpose,
                "vtrpose.xlu": compute_bf16_transpose,
                "reduce.max.xlu": compute_bf16_row_reduce_max,
                "vreduce.max.xlu": compute_bf16_row_reduce_max,
                "reduce.sum.xlu": compute_bf16_row_reduce_sum,
                "vreduce.sum.xlu": compute_bf16_row_reduce_sum,
            }[instruction.mnemonic]

            def on_start(cycle: int) -> None:
                del cycle
                captured["src"] = self.state.load_mreg(params.ms).clone()

            def on_complete(cycle: int) -> None:
                if "src" not in captured:
                    return
                self.state.trace_end_cycle = self._trace_cycle(completion_cycle)
                self.state.store_mreg(params.md, op(captured["src"], config=self.state.config))

            return on_start, on_complete

        raise TypeError(f"Unsupported async instruction '{instruction.mnemonic}'")

    def _make_execute_callbacks(
        self,
        uop: PipelineUop,
        *,
        execute_start_cycle: int,
        completion_cycle: int,
    ) -> tuple[Callable[[int], None] | None, Callable[[int], None] | None]:
        instruction = uop.instruction
        params = instruction.params
        spec = ALL_INSTRUCTION_SPECS[instruction.mnemonic]
        execute_lane = _execute_lane_for_instruction(instruction)
        logger = self.state.trace_logger

        if instruction.mnemonic.startswith(("dma.load.", "dma.store.")):
            channel = _dma_channel_for_instruction(instruction)

            def on_start(cycle: int) -> None:
                saved_pc = self._set_execute_pc(uop.pc, trace_cycle=cycle)
                spec.semantics(self.state, instruction.params)
                self._restore_fetch_pc(saved_pc)
                if channel is not None:
                    transfer = self.state.dma_channels[channel].pending
                    if transfer is not None:
                        self._dma_scheduled_ready_cycle[channel] = transfer.ready_cycle
                        self._dma_scheduled_valid[channel] = True
                if logger is not None and self.state.stop_reason is None:
                    if channel is not None:
                        transfer = self.state.dma_channels[channel].pending
                        if transfer is not None:
                            logger.log_stage_start(
                                uop.insn_id,
                                "transfer",
                                lane=_dma_transfer_lane(channel),
                                cycle=self._trace_cycle(cycle + 1),
                            )
                            logger.log_stage_end(
                                uop.insn_id,
                                "transfer",
                                lane=_dma_transfer_lane(channel),
                                cycle=self._trace_cycle(transfer.ready_cycle),
                            )

            def on_complete(cycle: int) -> None:
                self.state.trace_end_cycle = self._trace_cycle(cycle)
                self.state.perf.record_instruction(instruction.mnemonic)
                self._mark_step_limit_if_reached()
                self._log_stop(cycle)

            return on_start, on_complete

        if instruction.mnemonic == "vload":
            assert isinstance(params, TensorMemType)
            captured: dict[str, object] = {}

            def on_start(cycle: int) -> None:
                del cycle
                address = (self.state.read_xreg(params.rs1) + params.imm) & 0xFFFF_FFFF
                if not self.state._check_tensor_alignment(address):
                    return
                captured["address"] = address
                captured["payload"] = self.state.vmem.read(
                    address,
                    self.state.config.mreg_bytes,
                ).clone()

            def on_complete(cycle: int) -> None:
                if "payload" not in captured:
                    return
                self.state.trace_end_cycle = self._trace_cycle(cycle)
                self.state.instruction_extra_cycles = (
                    self.state.config.vmem_transfer_cycles(self.state.config.mreg_bytes)
                    - self.state.config.vload_latency_cycles
                )
                self.state.perf.bytes_read += self.state.config.mreg_bytes
                self.state.store_mreg(params.mreg, captured["payload"])
                self.state._log_memory_access(
                    "vmem",
                    "vload",
                    int(captured["address"]),
                    0,
                    size=self.state.config.mreg_bytes,
                )
                self.state.perf.record_instruction(instruction.mnemonic)
                self._mark_step_limit_if_reached()
                self._log_stop(cycle)

            return on_start, on_complete

        if instruction.mnemonic == "vstore":
            assert isinstance(params, TensorMemType)
            captured: dict[str, object] = {}

            def on_start(cycle: int) -> None:
                del cycle
                address = (self.state.read_xreg(params.rs1) + params.imm) & 0xFFFF_FFFF
                if not self.state._check_tensor_alignment(address):
                    return
                captured["address"] = address
                captured["payload"] = self.state.load_mreg(params.mreg).clone()

            def on_complete(cycle: int) -> None:
                if "payload" not in captured:
                    return
                self.state.trace_end_cycle = self._trace_cycle(cycle)
                self.state.instruction_extra_cycles = (
                    self.state.config.vmem_transfer_cycles(self.state.config.mreg_bytes)
                    - self.state.config.vstore_latency_cycles
                )
                self.state.perf.bytes_written += self.state.config.mreg_bytes
                self.state.vmem.write(int(captured["address"]), captured["payload"])
                self.state._log_memory_access(
                    "vmem",
                    "vstore",
                    int(captured["address"]),
                    0,
                    size=self.state.config.mreg_bytes,
                )
                self.state.perf.record_instruction(instruction.mnemonic)
                self._mark_step_limit_if_reached()
                self._log_stop(cycle)

            return on_start, on_complete

        if _is_async_tensor_lane(execute_lane):
            async_start, async_complete = self._build_async_callbacks(uop, completion_cycle)

            def on_start(cycle: int) -> None:
                async_start(cycle)

            def on_complete(cycle: int) -> None:
                async_complete(cycle)
                self.state.perf.record_instruction(instruction.mnemonic)
                self._mark_step_limit_if_reached()
                self._log_stop(cycle)

            return on_start, on_complete

        def on_complete(cycle: int) -> None:
            self.state.trace_end_cycle = self._trace_cycle(cycle)
            saved_pc = self._set_execute_pc(uop.pc, trace_cycle=cycle)
            spec.semantics(self.state, instruction.params)
            self._consume_control_transfer(uop.insn_id)
            self._restore_fetch_pc(saved_pc)
            self.state.perf.record_instruction(instruction.mnemonic)
            self._mark_step_limit_if_reached()
            self._log_stop(cycle)

        return None, on_complete

    def _try_retire_decode_only(self, uop: PipelineUop, cycle: int) -> bool:
        instruction = uop.instruction
        if not _is_decode_fence_instruction(instruction):
            return False
        if uop.dispatch_start_cycle is None or cycle <= uop.dispatch_start_cycle:
            return False

        if _is_delay_instruction(instruction):
            retire_cycle = uop.dispatch_start_cycle + 1 + instruction.params.cycles
        else:
            wait_channel = _dma_channel_for_instruction(instruction)
            if wait_channel is None:
                raise ValueError(f"Malformed DMA wait mnemonic '{instruction.mnemonic}'")
            retire_cycle = self.state.dma_wait_completion_cycle(
                wait_channel, uop.dispatch_start_cycle
            )
            if self._dma_scheduled_valid[wait_channel]:
                retire_cycle = max(retire_cycle, self._dma_scheduled_ready_cycle[wait_channel])
        if cycle < retire_cycle:
            return False

        logger = self.state.trace_logger
        if logger is not None:
            trace_cycle = self._trace_cycle(cycle)
            logger.log_stage_end(uop.insn_id, "dispatch", lane=DISPATCH_LANE, cycle=trace_cycle)
            logger.log_retire(uop.insn_id, lane=DISPATCH_LANE, cycle=trace_cycle)
        self.state.trace_end_cycle = self._trace_cycle(cycle)
        self.state.perf.record_instruction(instruction.mnemonic)
        if _is_delay_instruction(instruction):
            pass
        else:
            self.state.retire_dma_wait(wait_channel, retire_cycle=cycle)
            self._dma_scheduled_valid[wait_channel] = False
            self._dma_scheduled_ready_cycle[wait_channel] = 0
        self._mark_step_limit_if_reached()
        self._log_stop(cycle)
        return True

    def _try_dispatch(self, uop: PipelineUop, cycle: int) -> str | None:
        instruction = uop.instruction
        if _is_decode_fence_instruction(instruction):
            return None
        if instruction.mnemonic not in ALL_INSTRUCTION_SPECS:
            self.state.stop(StopReason.ILLEGAL_INSTRUCTION)
            return None
        spec = ALL_INSTRUCTION_SPECS[instruction.mnemonic]
        if not isinstance(instruction.params, spec.params_type):
            raise TypeError(
                f"{instruction.mnemonic} expects {spec.params_type.__name__}, "
                f"got {type(instruction.params).__name__}"
            )
        if uop.dispatch_start_cycle is None or cycle < uop.dispatch_start_cycle:
            return None

        unit_key = _unit_key_for_instruction(instruction)
        execute_lane = _UNIT_LANES[unit_key]
        total_latency = _instruction_latency_cycles(self.state, instruction, spec.latency)
        lane_occupancy = _lane_occupancy_cycles(instruction, execute_lane, total_latency)
        execute_start_cycle = max(
            cycle + 1,
            self._unit_next_available_cycle[unit_key],
        )
        completion_cycle = execute_start_cycle + total_latency

        uop.unit_key = unit_key
        uop.dispatch_end_cycle = cycle + 1
        uop.execute_start_cycle = execute_start_cycle
        uop.completion_cycle = completion_cycle
        uop.on_execute_start, uop.on_execute_complete = self._make_execute_callbacks(
            uop,
            execute_start_cycle=execute_start_cycle,
            completion_cycle=completion_cycle,
        )
        if self.state.trace_logger is not None:
            self.state.trace_logger.log_stage_end(
                uop.insn_id,
                "dispatch",
                lane=DISPATCH_LANE,
                cycle=self._trace_cycle(uop.dispatch_end_cycle),
            )

        self._unit_next_available_cycle[unit_key] = execute_start_cycle + lane_occupancy
        if isinstance(instruction.params, DMAType):
            channel = _dma_channel_for_instruction(instruction)
            if channel is not None:
                predicted_size = self.state.read_xreg(instruction.params.rs2)
                predicted_ready_cycle = execute_start_cycle + self.state.config.dma_transfer_cycles(
                    predicted_size
                )
                self._dma_scheduled_ready_cycle[channel] = predicted_ready_cycle
                self._dma_scheduled_valid[channel] = True
        return unit_key

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

        self._program_loaded = True
        self._program = instructions
        self._program_base = program_base
        self._program_end = program_base + len(instructions)
        self._max_instructions = max_instructions
        self._start_count = self.state.perf.instructions
        self._reset_scoreboards(self.state.perf.cycles)
        self._reset_pipeline_state(self.state.perf.cycles)
        self._set_trace_logger(trace_logger)
        self.state.clear_stop()

        self._ifu.load_program(
            instructions,
            program_base=program_base,
            start_pc=start_pc,
        )
        self.state.pc = start_pc & 0xFFFF_FFFF
        if trace_logger is not None:
            trace_logger.set_pc_base_address(program_base)
            trace_logger.log_arch_value(
                "pc",
                0,
                self.state.pc,
                cycle=self._trace_cycle(self.state.perf.cycles),
            )

    def _pipeline_drained(self) -> bool:
        ifu_drained = self._ifu.is_finished() or (
            self._step_limit_reached and not self._ifu.output.is_valid()
        )
        return (
            ifu_drained
            and self._idu.is_finished()
            and all(not exu.has_in_flight for exu in self._exus.values())
        )

    def tick(self) -> bool:
        if not self._program_loaded:
            raise RuntimeError("No program loaded; call load_program() or execute() first")
        if self.state.stop_reason is not None:
            return False

        current_cycle = self.state.perf.cycles
        if self._max_instructions is not None:
            executed = self.state.perf.instructions - self._start_count
            if executed >= self._max_instructions:
                self._step_limit_reached = True
                self._squash_frontend_for_step_limit()

        if self._step_limit_reached and self._pipeline_drained():
            self.state.stop(StopReason.STEP_LIMIT)
            self._log_stop(current_cycle)
            return False
        if not self._step_limit_reached and self._pipeline_drained():
            self.state.stop(StopReason.PROGRAM_END)
            self._log_stop(current_cycle)
            return False

        active_cycle = current_cycle + 1
        self.state.perf.cycles = active_cycle

        for exu in self._exus.values():
            exu.complete_cycle(active_cycle)
        if self.state.stop_reason is not None:
            self._log_stop(active_cycle)
            return False
        for unit_key, exu in self._exus.items():
            exu.start_cycle(active_cycle)
        if self.state.stop_reason is not None:
            self._log_stop(active_cycle)
            return False
        if self._step_limit_reached:
            self._squash_frontend_for_step_limit()
        else:
            self._idu.tick(
                active_cycle,
                self._ifu.output,
                on_claim_from_ifu=self._on_claim_from_ifu,
                try_retire_decode_only=self._try_retire_decode_only,
                try_dispatch=self._try_dispatch,
            )
            for unit_key, queue in self._idu.outputs.items():
                exu = self._exus[unit_key]
                while queue:
                    exu.enqueue(queue.popleft())
        if self.state.stop_reason is not None:
            self._log_stop(active_cycle)
            return False
        if self._step_limit_reached:
            self._squash_frontend_for_step_limit()
        else:
            frontend_fenced = (
                self._idu.current_uop is not None
                and _is_decode_fence_instruction(self._idu.current_uop.instruction)
            )
            self._ifu.tick(
                active_cycle,
                allow_fetch=(not frontend_fenced) or (not self._ifu.output.is_valid()),
                on_fetch=self._on_fetch,
                on_fetch_stall=self._on_fetch_stall,
                on_misaligned_fetch=self._on_misaligned_fetch,
                on_fetch_pc_advanced=self._on_fetch_pc_advanced,
            )

        self.state.pc = self._ifu.fetch_pc
        if self.state.stop_reason is not None:
            self._log_stop(active_cycle)
            return False
        if self._step_limit_reached and self._pipeline_drained():
            self.state.stop(StopReason.STEP_LIMIT)
            self._log_stop(active_cycle)
            return False
        if not self._step_limit_reached and self._pipeline_drained():
            self.state.stop(StopReason.PROGRAM_END)
            self._log_stop(active_cycle)
            return False
        return True

    def is_finished(self) -> bool:
        return self._pipeline_drained()


from .simulation import Sim  # noqa: E402

__all__ = ["Core", "INSTRUCTION_LATENCY", "Sim"]
