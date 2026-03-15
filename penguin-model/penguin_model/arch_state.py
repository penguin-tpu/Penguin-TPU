"""Architecture-visible machine state for the Penguin model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from .memory import (
    DMA_ALIGNMENT_BYTES,
    DRAM_BASE,
    DRAM_SIZE,
    IMEM_BASE,
    IMEM_SIZE,
    VMEM_BASE,
    VMEM_SIZE,
    DMAChannel,
    DMATransfer,
    Memory,
)

if TYPE_CHECKING:
    from .logging import TraceLogger

DRAM_LATENCY_CYCLES = 10
CONTROL_FLOW_DELAY_SLOTS = 2


class StopReason(str, Enum):
    """Execution stop reasons exposed by the functional model."""

    PROGRAM_END = "program_end"
    ECALL = "ecall"
    EBREAK = "ebreak"
    MISALIGNED_LOAD = "misaligned_load"
    MISALIGNED_STORE = "misaligned_store"
    INSTRUCTION_ADDRESS_MISALIGNED = "instruction_address_misaligned"
    DMA_CHANNEL_BUSY = "dma_channel_busy"
    DMA_MISALIGNED_ADDRESS = "dma_misaligned_address"
    DMA_MISALIGNED_SIZE = "dma_misaligned_size"
    STEP_LIMIT = "step_limit"


@dataclass(slots=True)
class PerformanceCounters:
    """Simple counters for the functional/performance model."""

    cycles: int = 0
    instructions: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    instructions_by_opcode: dict[str, int] = field(default_factory=dict)

    def record_instruction(self, mnemonic: str, latency: int) -> None:
        self.instructions += 1
        self.cycles += latency
        self.instructions_by_opcode[mnemonic] = (
            self.instructions_by_opcode.get(mnemonic, 0) + 1
        )


@dataclass(slots=True)
class ArchState:
    """Architecture-visible state consumed by instruction semantic functions."""

    dram: Memory
    vmem: Memory
    imem: Memory
    dma_channels: list[DMAChannel]
    xreg: list[int] = field(default_factory=lambda: [0] * 32)
    pc: int = 0
    perf: PerformanceCounters = field(default_factory=PerformanceCounters)
    stop_reason: StopReason | None = None
    next_pc: int | None = None
    trace_logger: TraceLogger | None = field(default=None, repr=False)
    trace_start_cycle: int = 0
    trace_end_cycle: int = 0
    instruction_extra_cycles: int = 0
    delay_slots_remaining: int = 0
    control_transfer_set: bool = False

    @classmethod
    def with_memory_sizes(
        cls,
        *,
        dram_size: int = DRAM_SIZE,
        vmem_size: int = VMEM_SIZE,
        imem_size: int = IMEM_SIZE,
    ) -> ArchState:
        return cls(
            dram=Memory(name="dram", size=dram_size, base=DRAM_BASE, paged=True),
            vmem=Memory(name="vmem", size=vmem_size, base=VMEM_BASE),
            imem=Memory(name="imem", size=imem_size, base=IMEM_BASE),
            dma_channels=[DMAChannel() for _ in range(8)],
        )

    def read_xreg(self, index: int) -> int:
        return 0 if index == 0 else self.xreg[index]

    def write_xreg(self, index: int, value: int) -> None:
        if index != 0:
            value &= 0xFFFF_FFFF
            self.xreg[index] = value
            if self.trace_logger is not None:
                self.trace_logger.log_arch_value(
                    "xrf",
                    index,
                    value,
                    cycle=self.trace_end_cycle,
                )

    def stop(self, reason: StopReason) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason

    def clear_stop(self) -> None:
        self.stop_reason = None
        self.next_pc = None
        self.instruction_extra_cycles = 0
        self.delay_slots_remaining = 0
        self.control_transfer_set = False

    def clear_dma_channels(self) -> None:
        for channel in self.dma_channels:
            channel.pending = None

    def set_next_pc(self, value: int) -> None:
        if value % 4 != 0:
            self.stop(StopReason.INSTRUCTION_ADDRESS_MISALIGNED)
            return
        self.next_pc = value & 0xFFFF_FFFF
        self.delay_slots_remaining = CONTROL_FLOW_DELAY_SLOTS
        self.control_transfer_set = True

    def _log_memory_access(
        self,
        region: str,
        access_type: str,
        address: int,
        value: int,
        *,
        size: int,
        cycle: int | None = None,
    ) -> None:
        if self.trace_logger is not None:
            self.trace_logger.log_memory_access(
                region,
                access_type,
                address,
                value,
                size=size,
                cycle=self.trace_end_cycle if cycle is None else cycle,
            )

    def load_vmem_u32(self, address: int) -> int | None:
        if address % 4 != 0:
            self.stop(StopReason.MISALIGNED_LOAD)
            return None
        value = self.vmem.load_u32(address)
        self.perf.bytes_read += 4
        self._log_memory_access("vmem", "load", address, value, size=4)
        return value

    def store_vmem_u32(self, address: int, value: int) -> bool:
        if address % 4 != 0:
            self.stop(StopReason.MISALIGNED_STORE)
            return False
        value &= 0xFFFF_FFFF
        self.perf.bytes_written += 4
        self.vmem.store_u32(address, value)
        self._log_memory_access("vmem", "store", address, value, size=4)
        return True

    def _issue_dma(
        self,
        channel: int,
        *,
        direction: str,
        dram_address: int,
        vmem_address: int,
        size: int,
    ) -> None:
        dma_channel = self.dma_channels[channel]
        if dma_channel.busy:
            self.stop(StopReason.DMA_CHANNEL_BUSY)
            return

        if dram_address % DMA_ALIGNMENT_BYTES != 0 or vmem_address % DMA_ALIGNMENT_BYTES != 0:
            self.stop(StopReason.DMA_MISALIGNED_ADDRESS)
            return

        if size % DMA_ALIGNMENT_BYTES != 0:
            self.stop(StopReason.DMA_MISALIGNED_SIZE)
            return

        if direction == "load":
            payload = self.dram.read(dram_address, size).clone()
        else:
            payload = self.vmem.read(vmem_address, size).clone()

        dma_channel.pending = DMATransfer(
            direction=direction,
            dram_address=dram_address,
            vmem_address=vmem_address,
            size=size,
            payload=payload,
            ready_cycle=self.perf.cycles + DRAM_LATENCY_CYCLES,
        )

    def issue_dma_load(self, channel: int, dram_address: int, vmem_address: int, size: int) -> None:
        self._issue_dma(
            channel,
            direction="load",
            dram_address=dram_address,
            vmem_address=vmem_address,
            size=size,
        )

    def issue_dma_store(
        self, channel: int, dram_address: int, vmem_address: int, size: int
    ) -> None:
        self._issue_dma(
            channel,
            direction="store",
            dram_address=dram_address,
            vmem_address=vmem_address,
            size=size,
        )

    def wait_dma_channel(self, channel: int) -> None:
        dma_channel: DMAChannel = self.dma_channels[channel]
        transfer = dma_channel.pending
        if transfer is None:
            return

        completion_cycle = max(self.perf.cycles + 1, transfer.ready_cycle)
        self.instruction_extra_cycles = max(
            self.instruction_extra_cycles,
            completion_cycle - (self.perf.cycles + 1),
        )
        completion_trace_cycle = completion_cycle * 3

        if transfer.direction == "load":
            self.vmem.write(transfer.vmem_address, transfer.payload)
            self._log_memory_access(
                "dram",
                "dma-read",
                transfer.dram_address,
                0,
                size=transfer.size,
                cycle=completion_trace_cycle,
            )
            self._log_memory_access(
                "vmem",
                "dma-write",
                transfer.vmem_address,
                0,
                size=transfer.size,
                cycle=completion_trace_cycle,
            )
        else:
            self.dram.write(transfer.dram_address, transfer.payload)
            self._log_memory_access(
                "vmem",
                "dma-read",
                transfer.vmem_address,
                0,
                size=transfer.size,
                cycle=completion_trace_cycle,
            )
            self._log_memory_access(
                "dram",
                "dma-write",
                transfer.dram_address,
                0,
                size=transfer.size,
                cycle=completion_trace_cycle,
            )

        self.perf.bytes_read += transfer.size
        self.perf.bytes_written += transfer.size
        dma_channel.pending = None

__all__ = [
    "ArchState",
    "CONTROL_FLOW_DELAY_SLOTS",
    "PerformanceCounters",
    "StopReason",
]
