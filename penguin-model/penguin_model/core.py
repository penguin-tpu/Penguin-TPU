"""Core execution model for the Penguin scalar integer subset."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING

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
    SType,
    TensorMemType,
    UType,
    VPUBinaryType,
    VPUUnaryType,
    WeightMemType,
)
from .memory import Memory

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
    if isinstance(params, TensorMemType):
        return f"{mnemonic} m{params.mreg}, {params.imm}(x{params.rs1})"
    if isinstance(params, WeightMemType):
        return f"{mnemonic} w{params.slot}, {params.imm}(x{params.rs1})"
    if isinstance(params, MXUMatmulType):
        return f"{mnemonic} m{params.md}, m{params.ms}, w{params.ws}"
    if isinstance(params, MXUMatmulAccType):
        return f"{mnemonic} m{params.md}, m{params.ms}, w{params.ws}, m{params.mp}"
    if isinstance(params, VPUBinaryType):
        return f"{mnemonic} m{params.md}, m{params.ms1}, m{params.ms2}"
    if isinstance(params, VPUUnaryType):
        return f"{mnemonic} m{params.md}, m{params.ms}"
    return f"{mnemonic} {params}"


def _execute_lane_for_instruction(instruction: Instruction) -> int:
    if instruction.mnemonic.startswith("dma."):
        return DMA_LANE
    if instruction.mnemonic in {"vload", "vstore"} or instruction.mnemonic.startswith("mxu.push."):
        return TMEM_LANE
    if isinstance(instruction.params, (VPUBinaryType, VPUUnaryType)):
        return VPU_LANE
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


class PenguinCore:
    """Executor for scalar instruction streams over an architectural state."""

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
        self._reset_trace_pipeline()

    @property
    def config(self) -> PenguinCoreConfig:
        return self.state.config

    def _reset_trace_pipeline(self) -> None:
        self._trace_ifu_ready = 0
        self._trace_idu_ready = 0
        self._trace_issue_ready = 0
        self._trace_exu_ready = {
            SALU_LANE: 0,
            DMA_LANE: 0,
            TMEM_LANE: 0,
            MXU0_LANE: 0,
            MXU1_LANE: 0,
            VPU_LANE: 0,
        }

    @property
    def memory(self) -> Memory:
        return self.state.vmem

    @property
    def perf(self) -> PerformanceCounters:
        return self.state.perf

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
            mem_base=self.state.mem_base,
        )
        self._reset_trace_pipeline()

    def execute_instruction(self, instruction: Instruction) -> None:
        if instruction.mnemonic not in ALL_INSTRUCTION_SPECS:
            raise KeyError(f"Unknown mnemonic '{instruction.mnemonic}'")
        spec = ALL_INSTRUCTION_SPECS[instruction.mnemonic]
        if not isinstance(instruction.params, spec.params_type):
            raise TypeError(
                f"{instruction.mnemonic} expects {spec.params_type.__name__}, "
                f"got {type(instruction.params).__name__}"
            )

        current_pc = self.state.pc
        start_cycle = self.state.perf.cycles
        insn_id = self.state.perf.instructions
        logger = self.state.trace_logger
        execute_lane = _execute_lane_for_instruction(instruction)
        # Capture trace transfer info for dma.wait so we can log transfer end in trace time (not cycle time).
        trace_transfer_end_insn_id: int | None = None
        trace_transfer_end_channel: int | None = None
        if instruction.mnemonic.startswith("dma.wait."):
            wait_ch = _dma_channel_for_instruction(instruction)
            if wait_ch is not None:
                ch = self.state.dma_channels[wait_ch]
                if ch.trace_transfer_insn_id is not None and ch.trace_transfer_start is not None:
                    trace_transfer_end_insn_id = ch.trace_transfer_insn_id
                    trace_transfer_end_channel = wait_ch
        trace_fetch_start = self._trace_ifu_ready
        trace_fetch_min_end = trace_fetch_start + 1
        trace_dispatch_start = max(trace_fetch_min_end, self._trace_idu_ready)
        trace_execute_start = max(
            trace_dispatch_start + 1,
            self._trace_issue_ready,
            self._trace_exu_ready[execute_lane],
        )

        self.state.instruction_extra_cycles = 0
        self.state.control_transfer_set = False
        self.state.trace_start_cycle = trace_fetch_start
        trace_ticks_per_cycle = self.state.config.trace.ticks_per_cycle
        self.state.trace_end_cycle = (start_cycle + spec.latency) * trace_ticks_per_cycle
        if logger is not None:
            logger.log_insn(insn_id, _format_instruction(instruction))
            # PC records the fetch-stage value so the instruction in IFU matches the logged PC.
            logger.log_arch_value("pc", 0, current_pc, cycle=trace_fetch_start)
            logger.log_stage_start(
                insn_id,
                "fetch",
                lane=FETCH_LANE,
                cycle=trace_fetch_start,
            )
            logger.log_stage_end(
                insn_id,
                "fetch",
                lane=FETCH_LANE,
                cycle=trace_dispatch_start,
            )
            logger.log_stage_start(
                insn_id,
                "dispatch",
                lane=DISPATCH_LANE,
                cycle=trace_dispatch_start,
            )
            logger.log_stage_end(
                insn_id,
                "dispatch",
                lane=DISPATCH_LANE,
                cycle=trace_execute_start,
            )
            logger.log_stage_start(
                insn_id,
                "execute",
                lane=execute_lane,
                cycle=trace_execute_start,
            )

        spec.semantics(self.state, instruction.params)
        total_latency = max(1, spec.latency + self.state.instruction_extra_cycles)
        execute_duration = max(1, total_latency * trace_ticks_per_cycle - 2)
        trace_execute_end = trace_execute_start + execute_duration
        self.state.trace_end_cycle = trace_execute_end
        self.state.perf.record_instruction(instruction.mnemonic, total_latency)
        self._trace_ifu_ready = trace_dispatch_start
        self._trace_idu_ready = trace_execute_start
        self._trace_issue_ready = trace_execute_end
        self._trace_exu_ready[execute_lane] = trace_execute_end

        if (
            logger is not None
            and instruction.mnemonic.startswith(("dma.load.", "dma.store."))
            and self.state.stop_reason is None
        ):
            channel = _dma_channel_for_instruction(instruction)
            if channel is not None:
                transfer = self.state.dma_channels[channel].pending
                if transfer is not None:
                    logger.log_stage_start(
                        insn_id,
                        "transfer",
                        lane=DMA_LANE,
                        cycle=trace_execute_end,
                    )
                    # End is logged when the matching dma.wait completes (so it uses trace time).
                    self.state.dma_channels[channel].trace_transfer_insn_id = insn_id
                    self.state.dma_channels[channel].trace_transfer_start = trace_execute_end
        if (
            logger is not None
            and trace_transfer_end_insn_id is not None
            and trace_transfer_end_channel is not None
        ):
            logger.log_stage_end(
                trace_transfer_end_insn_id,
                "transfer",
                lane=DMA_LANE,
                cycle=trace_execute_end,
            )
            self.state.dma_channels[trace_transfer_end_channel].trace_transfer_insn_id = None
            self.state.dma_channels[trace_transfer_end_channel].trace_transfer_start = None

        if self.state.stop_reason is None:
            if self.state.control_transfer_set:
                self.state.pc = (current_pc + 4) & 0xFFFF_FFFF
            elif self.state.delay_slots_remaining > 1:
                self.state.delay_slots_remaining -= 1
                self.state.pc = (current_pc + 4) & 0xFFFF_FFFF
            elif self.state.delay_slots_remaining == 1:
                self.state.delay_slots_remaining = 0
                assert self.state.next_pc is not None
                self.state.pc = self.state.next_pc
                self.state.next_pc = None
            else:
                self.state.pc = (current_pc + 4) & 0xFFFF_FFFF

        if logger is not None:
            logger.log_stage_end(
                insn_id,
                "execute",
                lane=execute_lane,
                cycle=trace_execute_end,
            )
            logger.log_retire(insn_id, lane=execute_lane, cycle=trace_execute_end)
            if self.state.stop_reason is not None:
                logger.log_stop(self.state.stop_reason.value, cycle=trace_execute_end)

        self.state.control_transfer_set = False

    def execute(
        self,
        program: Iterable[Instruction],
        *,
        start_pc: int | None = None,
        max_instructions: int | None = None,
        trace_logger: TraceLogger | None = None,
    ) -> PerformanceCounters:
        instructions = list(program) if not isinstance(program, Sequence) else program
        program_base = getattr(program, "base_address", 0)
        if start_pc is None:
            start_pc = program_base
        self.state.pc = start_pc & 0xFFFF_FFFF
        self.state.clear_stop()
        self.state.trace_logger = trace_logger
        self._reset_trace_pipeline()

        if trace_logger is not None:
            trace_logger.log_arch_value(
                "pc",
                0,
                self.state.pc,
                cycle=self.state.perf.cycles * self.state.config.trace.ticks_per_cycle,
            )

        if self.state.pc % 4 != 0:
            self.state.stop(StopReason.INSTRUCTION_ADDRESS_MISALIGNED)
            if trace_logger is not None:
                trace_logger.log_stop(
                    self.state.stop_reason.value,
                    cycle=self.state.perf.cycles * self.state.config.trace.ticks_per_cycle,
                )
            self.state.trace_logger = None
            return self.state.perf

        start_count = self.state.perf.instructions
        program_end = program_base + len(instructions) * 4

        while self.state.stop_reason is None:
            if max_instructions is not None:
                executed = self.state.perf.instructions - start_count
                if executed >= max_instructions:
                    self.state.stop(StopReason.STEP_LIMIT)
                    if trace_logger is not None:
                        trace_logger.log_stop(
                            self.state.stop_reason.value,
                            cycle=self.state.perf.cycles
                            * self.state.config.trace.ticks_per_cycle,
                        )
                    break

            if self.state.pc == program_end:
                self.state.stop(StopReason.PROGRAM_END)
                if trace_logger is not None:
                    trace_logger.log_stop(
                        self.state.stop_reason.value,
                        cycle=self.state.perf.cycles * self.state.config.trace.ticks_per_cycle,
                    )
                break

            if self.state.pc % 4 != 0:
                self.state.stop(StopReason.INSTRUCTION_ADDRESS_MISALIGNED)
                if trace_logger is not None:
                    trace_logger.log_stop(
                        self.state.stop_reason.value,
                        cycle=self.state.perf.cycles * self.state.config.trace.ticks_per_cycle,
                    )
                break

            if self.state.pc < program_base:
                self.state.stop(StopReason.PROGRAM_END)
                if trace_logger is not None:
                    trace_logger.log_stop(
                        self.state.stop_reason.value,
                        cycle=self.state.perf.cycles * self.state.config.trace.ticks_per_cycle,
                    )
                break

            if self.state.pc > program_end:
                self.state.stop(StopReason.PROGRAM_END)
                if trace_logger is not None:
                    trace_logger.log_stop(
                        self.state.stop_reason.value,
                        cycle=self.state.perf.cycles * self.state.config.trace.ticks_per_cycle,
                    )
                break

            instruction_index = (self.state.pc - program_base) // 4
            self.execute_instruction(instructions[instruction_index])

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

        with TraceLogger(TraceLoggerConfig(filename=str(trace_path))) as trace_logger:
            return self.execute(
                program,
                start_pc=start_pc,
                max_instructions=max_instructions,
                trace_logger=trace_logger,
            )


__all__ = [
    "ArchState",
    "INSTRUCTION_LATENCY",
    "PerformanceCounters",
    "PenguinCore",
    "StopReason",
]
