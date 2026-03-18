"""Architecture-visible machine state for the Penguin model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import torch

from .core_config import DEFAULT_PENGUIN_CORE_CONFIG, PenguinCoreConfig
from .memory import (
    DMAChannel,
    DMATransfer,
    Memory,
    _mix_seed,
    _random_u8_tensor,
)
from .tensor import (
    accum_tile_from_bytes,
    accum_tile_to_bytes,
    bf16_tile_pair_from_bytes,
    bf16_tile_pair_to_bytes,
    export_accum_to_fp8,
    make_tensor_register_file_for_config,
    make_accum_buffer_file_for_config,
    make_weight_slot_file_for_config,
)

if TYPE_CHECKING:
    from .logging import TraceLogger

CONTROL_FLOW_DELAY_SLOTS = DEFAULT_PENGUIN_CORE_CONFIG.scalar.control_flow_delay_slots


class StopReason(str, Enum):
    """Execution stop reasons exposed by the functional model."""

    PROGRAM_END = "program_end"
    ECALL = "ecall"
    EBREAK = "ebreak"
    ILLEGAL_INSTRUCTION = "illegal_instruction"
    MISALIGNED_LOAD = "misaligned_load"
    MISALIGNED_STORE = "misaligned_store"
    INSTRUCTION_ADDRESS_MISALIGNED = "instruction_address_misaligned"
    DMA_CHANNEL_BUSY = "dma_channel_busy"
    DMA_MISALIGNED_ADDRESS = "dma_misaligned_address"
    DMA_MISALIGNED_SIZE = "dma_misaligned_size"
    TENSOR_MEMORY_MISALIGNED = "tensor_memory_misaligned"
    ILLEGAL_TENSOR_REGISTER_PAIR = "illegal_tensor_register_pair"
    STEP_LIMIT = "step_limit"


@dataclass(slots=True)
class PerformanceCounters:
    """Simple counters for the functional/performance model."""

    cycles: int = 0
    instructions: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    instructions_by_opcode: dict[str, int] = field(default_factory=dict)

    def record_instruction(self, mnemonic: str, latency: int = 0) -> None:
        self.instructions += 1
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
    config: PenguinCoreConfig = field(default_factory=lambda: DEFAULT_PENGUIN_CORE_CONFIG)
    mreg: torch.Tensor | None = None
    mxu_weight: torch.Tensor | None = None
    mxu_accum: torch.Tensor | None = None
    ereg: list[int] | None = None
    xreg: list[int] | None = None
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

    def __post_init__(self) -> None:
        if self.xreg is None:
            self.xreg = self._initial_xreg_file()
        elif len(self.xreg) != self.config.scalar.xreg_count:
            raise ValueError(
                f"xreg file expects {self.config.scalar.xreg_count} entries, got {len(self.xreg)}"
            )
        self.xreg[0] = 0
        if self.mreg is None:
            self.mreg = make_tensor_register_file_for_config(self.config)
        if self.mxu_weight is None:
            self.mxu_weight = make_weight_slot_file_for_config(self.config)
        if self.mxu_accum is None:
            self.mxu_accum = make_accum_buffer_file_for_config(self.config)
        if self.ereg is None:
            self.ereg = [0] * self.config.scale.num_ereg
        elif len(self.ereg) != self.config.scale.num_ereg:
            raise ValueError(
                f"ereg file expects {self.config.scale.num_ereg} entries, got {len(self.ereg)}"
            )

    def _initial_xreg_file(self) -> list[int]:
        if not self.config.initialization.randomize_scalar_registers:
            return [0] * self.config.scalar.xreg_count
        raw = _random_u8_tensor(
            self.config.scalar.xreg_count * 4,
            seed=_mix_seed(self.config.initialization.seed, 0x5852_4547),
        )
        xreg = [
            int.from_bytes(
                bytes(raw[index * 4 : (index + 1) * 4].tolist()),
                byteorder="little",
                signed=False,
            )
            for index in range(self.config.scalar.xreg_count)
        ]
        xreg[0] = 0
        return xreg

    @classmethod
    def from_config(
        cls,
        config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
    ) -> ArchState:
        return cls(
            dram=Memory(
                name="dram",
                size=config.memory_map.dram.size,
                base=config.memory_map.dram.base,
                paged=config.memory_backend.dram_paged,
                page_size=config.memory_backend.dram_page_size_bytes,
                randomize_contents=config.initialization.randomize_dram,
                init_seed=_mix_seed(config.initialization.seed, 0x4452_414D),
            ),
            vmem=Memory(
                name="vmem",
                size=config.memory_map.vmem.size,
                base=config.memory_map.vmem.base,
                randomize_contents=config.initialization.randomize_vmem,
                init_seed=_mix_seed(config.initialization.seed, 0x564D_454D),
            ),
            imem=Memory(
                name="imem",
                size=config.memory_map.imem.size,
                base=config.memory_map.imem.base,
                randomize_contents=False,
                init_seed=_mix_seed(config.initialization.seed, 0x494D_454D),
            ),
            dma_channels=[DMAChannel() for _ in range(config.dma.channel_count)],
            config=config,
        )

    @classmethod
    def with_memory_sizes(
        cls,
        *,
        dram_size: int | None = None,
        vmem_size: int | None = None,
        imem_size: int | None = None,
        config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
    ) -> ArchState:
        return cls.from_config(
            config.with_memory_sizes(
                dram_size=dram_size,
                vmem_size=vmem_size,
                imem_size=imem_size,
            )
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

    def read_ereg(self, index: int) -> int:
        assert self.ereg is not None
        return self.ereg[index] & 0xFF

    def write_ereg(self, index: int, value: int) -> None:
        assert self.ereg is not None
        value &= 0xFF
        self.ereg[index] = value
        if self.trace_logger is not None:
            self.trace_logger.log_arch_value(
                "erf",
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
        self.next_pc = value & 0xFFFF_FFFF
        self.delay_slots_remaining = self.config.scalar.control_flow_delay_slots
        self.control_transfer_set = True

    @property
    def bandwidth(self):
        """Access the bandwidth fragment of the core config."""

        return self.config.bandwidth

    def resolve_indirect_address(self, rs1: int, imm: int = 0) -> int:
        return (self.read_xreg(rs1) + imm) & 0xFFFF_FFFF

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

    def _check_vmem_alignment(self, address: int, *, align: int, reason: StopReason) -> bool:
        if address % align != 0:
            self.stop(reason)
            return False
        return True

    def load_vmem_u32(self, address: int) -> int | None:
        if not self._check_vmem_alignment(address, align=4, reason=StopReason.MISALIGNED_LOAD):
            return None
        value = self.vmem.load_u32(address)
        self.perf.bytes_read += 4
        self._log_memory_access("vmem", "load", address, value, size=4)
        return value

    def load_vmem_u8(self, address: int) -> int:
        value = self.vmem.load_u8(address)
        self.perf.bytes_read += 1
        self._log_memory_access("vmem", "load", address, value, size=1)
        return value

    def load_vmem_u16(self, address: int) -> int | None:
        if not self._check_vmem_alignment(address, align=2, reason=StopReason.MISALIGNED_LOAD):
            return None
        value = self.vmem.load_u16(address)
        self.perf.bytes_read += 2
        self._log_memory_access("vmem", "load", address, value, size=2)
        return value

    def store_vmem_u8(self, address: int, value: int) -> bool:
        value &= 0xFF
        self.perf.bytes_written += 1
        self.vmem.store_u8(address, value)
        self._log_memory_access("vmem", "store", address, value, size=1)
        return True

    def store_vmem_u16(self, address: int, value: int) -> bool:
        if not self._check_vmem_alignment(address, align=2, reason=StopReason.MISALIGNED_STORE):
            return False
        value &= 0xFFFF
        self.perf.bytes_written += 2
        self.vmem.store_u16(address, value)
        self._log_memory_access("vmem", "store", address, value, size=2)
        return True

    def store_vmem_u32(self, address: int, value: int) -> bool:
        if not self._check_vmem_alignment(address, align=4, reason=StopReason.MISALIGNED_STORE):
            return False
        value &= 0xFFFF_FFFF
        self.perf.bytes_written += 4
        self.vmem.store_u32(address, value)
        self._log_memory_access("vmem", "store", address, value, size=4)
        return True

    def load_mreg(self, index: int) -> torch.Tensor:
        assert self.mreg is not None
        return self.mreg[index].clone()

    def store_mreg(self, index: int, raw: torch.Tensor) -> None:
        if raw.numel() != self.config.mreg_bytes:
            raise ValueError(
                f"tensor register write expects {self.config.mreg_bytes} bytes, got {raw.numel()}"
            )
        assert self.mreg is not None
        self.mreg[index] = raw.reshape(self.config.mreg_bytes).to(torch.uint8)

    def load_weight_slot(self, mxu: int, slot: int) -> torch.Tensor:
        assert self.mxu_weight is not None
        return self.mxu_weight[mxu, slot].clone()

    def store_weight_slot(self, mxu: int, slot: int, raw: torch.Tensor) -> None:
        if raw.numel() != self.config.weight_slot_bytes:
            raise ValueError(
                f"weight-slot write expects {self.config.weight_slot_bytes} bytes, got {raw.numel()}"
            )
        assert self.mxu_weight is not None
        self.mxu_weight[mxu, slot] = raw.reshape(self.config.weight_slot_bytes).to(
            torch.uint8
        )

    def load_accum_buffer(self, mxu: int) -> torch.Tensor:
        assert self.mxu_accum is not None
        return self.mxu_accum[mxu].clone()

    def store_accum_buffer(self, mxu: int, raw: torch.Tensor) -> None:
        if raw.numel() != self.config.accum_buffer_bytes:
            raise ValueError(
                f"accumulator write expects {self.config.accum_buffer_bytes} bytes, got {raw.numel()}"
            )
        assert self.mxu_accum is not None
        self.mxu_accum[mxu] = raw.reshape(self.config.accum_buffer_bytes).to(torch.uint8)

    def check_mreg_pair_base(self, index: int) -> bool:
        if index < 0 or index + self.config.matmul_result_registers > self.config.tensor.num_mreg:
            self.stop(StopReason.ILLEGAL_TENSOR_REGISTER_PAIR)
            return False
        return True

    def _check_tensor_alignment(self, address: int) -> bool:
        if address % self.config.tensor.vmem_alignment_bytes != 0:
            self.stop(StopReason.TENSOR_MEMORY_MISALIGNED)
            return False
        return True

    def vload(self, mreg: int, address: int) -> None:
        if not self._check_tensor_alignment(address):
            return
        payload = self.vmem.read(address, self.config.mreg_bytes).clone()
        self.instruction_extra_cycles = (
            self.config.vmem_transfer_cycles(self.config.mreg_bytes)
            - self.config.vload_latency_cycles
        )
        self.perf.bytes_read += self.config.mreg_bytes
        self.store_mreg(mreg, payload)
        self._log_memory_access("vmem", "vload", address, 0, size=self.config.mreg_bytes)

    def vstore(self, mreg: int, address: int) -> None:
        if not self._check_tensor_alignment(address):
            return
        payload = self.load_mreg(mreg)
        self.instruction_extra_cycles = (
            self.config.vmem_transfer_cycles(self.config.mreg_bytes)
            - self.config.vstore_latency_cycles
        )
        self.perf.bytes_written += self.config.mreg_bytes
        self.vmem.write(address, payload)
        self._log_memory_access("vmem", "vstore", address, 0, size=self.config.mreg_bytes)

    def push_weight_slot_from_vmem(self, mxu: int, slot: int, address: int) -> None:
        if not self._check_tensor_alignment(address):
            return
        payload = self.vmem.read(address, self.config.weight_slot_bytes).clone()
        self.instruction_extra_cycles = (
            self.config.vmem_transfer_cycles(self.config.weight_slot_bytes)
            - self.config.vload_weight_latency_cycles
        )
        self.perf.bytes_read += self.config.weight_slot_bytes
        self.store_weight_slot(mxu, slot, payload)
        self._log_memory_access(
            "vmem",
            f"vload.weight.mxu{mxu}",
            address,
            slot,
            size=self.config.weight_slot_bytes,
        )

    def push_weight_slot_from_mreg(self, mxu: int, slot: int, mreg: int) -> None:
        payload = self.load_mreg(mreg)
        self.instruction_extra_cycles = (
            self.config.vmem_transfer_cycles(self.config.weight_slot_bytes)
            - self.config.vmatpush_weight_latency_cycles
        )
        self.store_weight_slot(mxu, slot, payload)

    def push_accum_from_mregs(self, mxu: int, base: int) -> None:
        if not self.check_mreg_pair_base(base):
            return
        tile = bf16_tile_pair_from_bytes(
            self.load_mreg(base),
            self.load_mreg(base + 1),
            config=self.config,
        )
        self.instruction_extra_cycles = (
            self.config.vmem_transfer_cycles(self.config.accum_buffer_bytes)
            - self.config.vmatpush_acc_latency_cycles
        )
        self.store_accum_buffer(mxu, accum_tile_to_bytes(tile, config=self.config))

    def pop_accum_to_mregs(self, mxu: int, base: int) -> None:
        if not self.check_mreg_pair_base(base):
            return
        raw_lo, raw_hi = bf16_tile_pair_to_bytes(
            accum_tile_from_bytes(self.load_accum_buffer(mxu), config=self.config),
            config=self.config,
        )
        self.instruction_extra_cycles = (
            self.config.vmem_transfer_cycles(self.config.accum_buffer_bytes)
            - self.config.vmatpop_acc_bf16_latency_cycles
        )
        self.store_mreg(base, raw_lo)
        self.store_mreg(base + 1, raw_hi)

    def pop_accum_to_fp8_mreg(self, mxu: int, mreg: int) -> None:
        self.instruction_extra_cycles = (
            self.config.vmem_transfer_cycles(self.config.mreg_bytes)
            - self.config.vmatpop_acc_fp8_latency_cycles
        )
        self.store_mreg(mreg, export_accum_to_fp8(self.load_accum_buffer(mxu), config=self.config))

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

        if (
            dram_address % self.config.dma.alignment_bytes != 0
            or vmem_address % self.config.dma.alignment_bytes != 0
        ):
            self.stop(StopReason.DMA_MISALIGNED_ADDRESS)
            return

        if size % self.config.dma.alignment_bytes != 0:
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
            ready_cycle=self.perf.cycles + self.config.dma_transfer_cycles(size),
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

    def dma_wait_completion_cycle(self, channel: int, issue_cycle: int) -> int:
        """Return the cycle when `dma.wait` may retire for the selected channel."""

        transfer = self.dma_channels[channel].pending
        if transfer is None:
            return issue_cycle + 1
        return max(issue_cycle + 1, transfer.ready_cycle)

    def retire_dma_wait(self, channel: int, *, retire_cycle: int) -> None:
        """Commit a completed DMA transfer when the matching wait retires."""

        dma_channel: DMAChannel = self.dma_channels[channel]
        transfer = dma_channel.pending
        if transfer is None:
            return
        if retire_cycle < transfer.ready_cycle:
            raise ValueError("dma.wait retired before the DMA transfer completed")

        completion_trace_cycle = retire_cycle * self.config.trace.ticks_per_cycle

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

    def wait_dma_channel(self, channel: int) -> None:
        completion_cycle = self.dma_wait_completion_cycle(channel, self.perf.cycles)
        self.instruction_extra_cycles = max(
            self.instruction_extra_cycles,
            completion_cycle - (self.perf.cycles + 1),
        )
        self.retire_dma_wait(channel, retire_cycle=completion_cycle)

__all__ = [
    "ArchState",
    "CONTROL_FLOW_DELAY_SLOTS",
    "PerformanceCounters",
    "StopReason",
]
