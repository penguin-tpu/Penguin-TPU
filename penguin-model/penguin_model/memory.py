"""Memory-region primitives for the Penguin functional model."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

from .core_config import DEFAULT_PENGUIN_CORE_CONFIG

IMEM_BASE = DEFAULT_PENGUIN_CORE_CONFIG.memory_map.imem.base
VMEM_BASE = DEFAULT_PENGUIN_CORE_CONFIG.memory_map.vmem.base
DRAM_BASE = DEFAULT_PENGUIN_CORE_CONFIG.memory_map.dram.base

IMEM_SIZE = DEFAULT_PENGUIN_CORE_CONFIG.memory_map.imem.size
VMEM_SIZE = DEFAULT_PENGUIN_CORE_CONFIG.memory_map.vmem.size
DRAM_SIZE = DEFAULT_PENGUIN_CORE_CONFIG.memory_map.dram.size
DMA_ALIGNMENT_BYTES = DEFAULT_PENGUIN_CORE_CONFIG.dma.alignment_bytes
DMA_CHANNEL_COUNT = DEFAULT_PENGUIN_CORE_CONFIG.dma.channel_count

DEFAULT_PAGE_SIZE = DEFAULT_PENGUIN_CORE_CONFIG.memory_backend.dram_page_size_bytes


class Memory:
    """Byte-addressed little-endian memory region with dense or paged backing."""

    def __init__(
        self,
        name: str,
        size: int,
        base: int = 0x0000_0000,
        verbose: bool = False,
        paged: bool = False,
        page_size: int = DEFAULT_PAGE_SIZE,
        randomize_contents: bool = False,
        init_seed: int = 0,
    ) -> None:
        self.name = name
        self.base = base
        self.size = size
        self.verbose = verbose
        self.paged = paged
        self.page_size = page_size
        self.randomize_contents = randomize_contents
        self.init_seed = init_seed
        self.mem: torch.Tensor | None = None
        self.pages: dict[int, torch.Tensor] = {}
        if self.paged:
            assert self.page_size > 0, "page_size must be positive"
        else:
            if self.randomize_contents:
                self.mem = _random_u8_tensor(self.size, seed=self.init_seed)
            else:
                self.mem = torch.zeros(self.size, dtype=torch.uint8)

    def _offset(self, address: int, size: int) -> int:
        offset = address - self.base
        assert offset >= 0, (
            f"Memory '{self.name}': address below base: 0x{address:08x} < 0x{self.base:08x}"
        )
        assert offset + size <= self.size, (
            f"Memory '{self.name}': access out of bounds: 0x{address:08x} + {size} bytes"
        )
        return offset

    def _page(self, page_index: int, *, create: bool) -> torch.Tensor | None:
        page = self.pages.get(page_index)
        if page is None and (create or self.randomize_contents):
            if self.randomize_contents:
                page = _random_u8_tensor(
                    self.page_size,
                    seed=_mix_seed(self.init_seed, page_index),
                )
            else:
                page = torch.zeros(self.page_size, dtype=torch.uint8)
            self.pages[page_index] = page
        return page

    def read(self, address: int, size: int) -> torch.Tensor:
        offset = self._offset(address, size)
        if self.verbose:
            print(
                f"\033[90m  Memory '{self.name}': read {size} bytes <- 0x{address:08x}\033[0m"
            )
        if not self.paged:
            assert self.mem is not None
            return self.mem[offset : offset + size]

        result = torch.zeros(size, dtype=torch.uint8)
        copied = 0
        while copied < size:
            absolute_offset = offset + copied
            page_index = absolute_offset // self.page_size
            page_offset = absolute_offset % self.page_size
            chunk_size = min(size - copied, self.page_size - page_offset)
            page = self._page(page_index, create=False)
            if page is not None:
                result[copied : copied + chunk_size] = page[page_offset : page_offset + chunk_size]
            copied += chunk_size
        return result

    def write(self, address: int, data: torch.Tensor) -> None:
        offset = self._offset(address, int(data.numel()))
        assert data.dtype == torch.uint8, (
            f"Memory '{self.name}': write data must be uint8, got {data.dtype}"
        )
        flattened = data.flatten()
        if not self.paged:
            assert self.mem is not None
            self.mem[offset : offset + flattened.numel()] = flattened
        else:
            written = 0
            size = int(flattened.numel())
            while written < size:
                absolute_offset = offset + written
                page_index = absolute_offset // self.page_size
                page_offset = absolute_offset % self.page_size
                chunk_size = min(size - written, self.page_size - page_offset)
                page = self._page(page_index, create=True)
                assert page is not None
                page[page_offset : page_offset + chunk_size] = flattened[
                    written : written + chunk_size
                ]
                written += chunk_size
        if self.verbose:
            print(
                f"\033[90m  Memory '{self.name}': wrote {data.numel()} bytes -> 0x{address:08x}\033[0m"
            )

    def copy_from(self, src: Memory, src_address: int, dst_address: int, size: int) -> None:
        self.write(dst_address, src.read(src_address, size).clone())

    def load_u32(self, address: int) -> int:
        data = self.read(address, 4)
        return int.from_bytes(bytes(data.tolist()), byteorder="little", signed=False)

    def store_u32(self, address: int, value: int) -> None:
        word = value & 0xFFFF_FFFF
        data = torch.tensor(
            [
                (word >> 0) & 0xFF,
                (word >> 8) & 0xFF,
                (word >> 16) & 0xFF,
                (word >> 24) & 0xFF,
            ],
            dtype=torch.uint8,
        )
        self.write(address, data)

    def load_f32(self, address: int) -> float:
        return struct.unpack("<f", struct.pack("<I", self.load_u32(address)))[0]

    def store_f32(self, address: int, value: float) -> None:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        self.store_u32(address, bits)


@dataclass(slots=True)
class DMATransfer:
    """Pending DMA transfer between DRAM and VMEM."""

    direction: str
    dram_address: int
    vmem_address: int
    size: int
    payload: torch.Tensor
    ready_cycle: int


@dataclass(slots=True)
class DMAChannel:
    """Single DMA channel state."""

    pending: DMATransfer | None = None
    # Trace-only: when a load logs transfer start, store so the matching wait can log end in trace time.
    trace_transfer_insn_id: int | None = None
    trace_transfer_start: int | None = None

    @property
    def busy(self) -> bool:
        return self.pending is not None


def _mix_seed(seed: int, *components: int) -> int:
    mixed = seed & 0xFFFF_FFFF_FFFF_FFFF
    for component in components:
        mixed ^= int(component) & 0xFFFF_FFFF_FFFF_FFFF
        mixed = (mixed * 0x9E37_79B9_7F4A_7C15) & 0xFFFF_FFFF_FFFF_FFFF
        mixed ^= mixed >> 33
    return mixed & 0x7FFF_FFFF_FFFF_FFFF


def _random_u8_tensor(numel: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
    return torch.randint(0, 256, (numel,), dtype=torch.int64, generator=generator).to(
        torch.uint8
    )


__all__ = [
    "DMA_ALIGNMENT_BYTES",
    "DMA_CHANNEL_COUNT",
    "DEFAULT_PAGE_SIZE",
    "DMAChannel",
    "DMATransfer",
    "DRAM_BASE",
    "DRAM_SIZE",
    "IMEM_BASE",
    "IMEM_SIZE",
    "Memory",
    "VMEM_BASE",
    "VMEM_SIZE",
    "_mix_seed",
    "_random_u8_tensor",
]
