"""Hierarchical configuration for the Penguin functional/performance model."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


def _ceil_div(dividend: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    if dividend <= 0:
        return 0
    return (dividend + divisor - 1) // divisor


@dataclass(frozen=True, slots=True)
class MemoryRegionConfig:
    """Address range configuration for one architectural memory region."""

    base: int
    """The base byte address of the memory region."""

    size: int
    """The capacity of the memory region in bytes."""


@dataclass(frozen=True, slots=True)
class ScalarCoreConfig:
    """Configuration of the scalar architectural core."""

    xreg_count: int = 32
    """The number of scalar architectural integer registers."""

    control_flow_delay_slots: int = 2
    """The number of architecturally required delay slots after control transfers."""


@dataclass(frozen=True, slots=True)
class ScaleConfig:
    """Configuration of the architectural scale-register file."""

    num_ereg: int = 32
    """The number of architectural exponent/scale registers."""

    ereg_bits: int = 8
    """The width of one scale register in bits."""


@dataclass(frozen=True, slots=True)
class MemoryMapConfig:
    """Top-level architectural memory map."""

    imem: MemoryRegionConfig = field(
        default_factory=lambda: MemoryRegionConfig(
            base=0x0002_0000,
            size=64 * 1024,
        )
    )
    """The on-chip instruction-memory address range."""

    vmem: MemoryRegionConfig = field(
        default_factory=lambda: MemoryRegionConfig(
            base=0x2000_0000,
            size=1 * 1024 * 1024,
        )
    )
    """The on-chip vector/tensor-memory address range."""

    dram: MemoryRegionConfig = field(
        default_factory=lambda: MemoryRegionConfig(
            base=0x8000_0000,
            size=16 * 1024 * 1024 * 1024,
        )
    )
    """The off-chip DRAM address range."""


@dataclass(frozen=True, slots=True)
class MemoryBackendConfig:
    """Python-model backing-store configuration."""

    dram_paged: bool = True
    """Whether DRAM uses sparse paged backing instead of one dense tensor."""

    dram_page_size_bytes: int = 4096
    """The page size of the sparse DRAM backing store in bytes."""


@dataclass(frozen=True, slots=True)
class InitializationConfig:
    """Power-on initialization behavior for architecturally visible state."""

    seed: int = 0x50E1_1234
    """The base seed used to deterministically generate pseudo-random initial contents."""

    randomize_dram: bool = True
    """Whether DRAM bytes are initialized to pseudo-random values before software writes them."""

    randomize_vmem: bool = True
    """Whether VMEM bytes are initialized to pseudo-random values before software writes them."""

    randomize_scalar_registers: bool = True
    """Whether scalar registers other than `x0` power up with pseudo-random values."""

    randomize_scale_registers: bool = True
    """Whether scale registers power up with pseudo-random values."""

    randomize_tensor_registers: bool = True
    """Whether tensor registers `m0..m63` power up with pseudo-random values."""

    randomize_weight_slots: bool = True
    """Whether MXU weight-slot state powers up with pseudo-random values."""

    randomize_accum_buffers: bool = True
    """Whether MXU accumulation buffers power up with pseudo-random values."""

    randomize_dma_base: bool = True
    """Whether `dma.base` powers up with a pseudo-random value."""


@dataclass(frozen=True, slots=True)
class DMAConfig:
    """DMA-engine configuration."""

    channel_count: int = 8
    """The number of architected DMA channels."""

    alignment_bytes: int = 32
    """The required alignment for DMA source, destination, and size in bytes."""

    offchip_command_words: int = 2
    """The number of serialized off-chip command words sent per DMA transfer."""


@dataclass(frozen=True, slots=True)
class TensorCoreConfig:
    """Tensor-register and MXU geometry."""

    num_mreg: int = 64
    """The number of architectural tensor registers."""

    mreg_rows: int = 64
    """The number of rows stored in one tensor register."""

    mreg_row_bytes: int = 64
    """The number of raw bytes stored in one tensor-register row."""

    mxu_count: int = 2
    """The number of architected MXU instances."""

    weight_slots_per_mxu: int = 2
    """The number of architected weight slots per MXU."""

    weight_tile_rows: int = 64
    """The number of rows in one MXU weight tile."""

    weight_tile_cols_fp8: int = 64
    """The number of FP8 columns stored in one MXU weight tile."""

    vmem_alignment_bytes: int = 32
    """The required VMEM alignment for tensor-register VMEM transfers."""

    matmul_latency_cycles: int = 64
    """The modeled latency of one MXU matmul launch in core cycles."""


@dataclass(frozen=True, slots=True)
class VPUConfig:
    """Vector-processing-unit timing configuration."""

    simple_op_latency_cycles: int = 2
    """The modeled latency of pipelineable VPU elementwise operations in core cycles."""

    non_pipelineable_op_latency_cycles: int = 8
    """The modeled latency of non-pipelineable VPU operations such as division in core cycles."""


@dataclass(frozen=True, slots=True)
class XLUConfig:
    """Cross-lane transpose-unit timing configuration."""

    transpose_latency_cycles: int = 4
    """The modeled latency of one whole-register transpose operation in core cycles."""


@dataclass(frozen=True, slots=True)
class BandwidthConfig:
    """Interconnect-width and serialization timing parameters."""

    offchip_link_width_bits: int = 32
    """The width of the off-chip bus connection in bits."""

    offchip_link_core_cycles_per_beat: int = 2
    """The number of core cycles required for one off-chip serialized beat."""

    vmem_bus_width_bits: int = 512
    """The width of the on-chip VMEM/system bus in bits."""

    vmem_bus_core_cycles_per_beat: int = 1
    """The number of core cycles required for one on-chip VMEM/system-bus beat."""

    @property
    def offchip_link_width_bytes(self) -> int:
        return self.offchip_link_width_bits // 8

    @property
    def vmem_bus_width_bytes(self) -> int:
        return self.vmem_bus_width_bits // 8

    def dma_offchip_cycles(self, payload_bytes: int, command_words: int = 2) -> int:
        total_bytes = payload_bytes + (command_words * self.offchip_link_width_bytes)
        beats = _ceil_div(total_bytes, self.offchip_link_width_bytes)
        return beats * self.offchip_link_core_cycles_per_beat

    def vmem_transfer_cycles(self, payload_bytes: int) -> int:
        beats = _ceil_div(payload_bytes, self.vmem_bus_width_bytes)
        return beats * self.vmem_bus_core_cycles_per_beat

    def dma_transfer_cycles(self, payload_bytes: int, command_words: int = 2) -> int:
        """Compute DMA completion time using the local bandwidth fragment only."""

        return max(
            self.dma_offchip_cycles(payload_bytes, command_words),
            self.vmem_transfer_cycles(payload_bytes),
        )


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """Trace-generation configuration."""

    ticks_per_cycle: int = 1
    """The number of trace time units emitted per modeled core cycle."""


@dataclass(frozen=True, slots=True)
class PenguinCoreConfig:
    """Single entry-point configuration for the Penguin chip model."""

    scalar: ScalarCoreConfig = field(default_factory=ScalarCoreConfig)
    """The scalar-core architectural configuration."""

    scale: ScaleConfig = field(default_factory=ScaleConfig)
    """The architectural scale-register configuration."""

    memory_map: MemoryMapConfig = field(default_factory=MemoryMapConfig)
    """The architectural memory-map configuration."""

    memory_backend: MemoryBackendConfig = field(default_factory=MemoryBackendConfig)
    """The Python-model backing-store configuration."""

    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    """The deterministic pseudo-random power-on initialization configuration."""

    dma: DMAConfig = field(default_factory=DMAConfig)
    """The DMA-engine configuration."""

    tensor: TensorCoreConfig = field(default_factory=TensorCoreConfig)
    """The tensor-register and MXU geometry configuration."""

    vpu: VPUConfig = field(default_factory=VPUConfig)
    """The VPU timing configuration."""

    xlu: XLUConfig = field(default_factory=XLUConfig)
    """The XLU timing configuration."""

    bandwidth: BandwidthConfig = field(default_factory=BandwidthConfig)
    """The interconnect bandwidth and serialization configuration."""

    trace: TraceConfig = field(default_factory=TraceConfig)
    """The trace-timestamp configuration."""

    def __post_init__(self) -> None:
        if self.scalar.xreg_count <= 0:
            raise ValueError("scalar.xreg_count must be positive")
        if self.scalar.control_flow_delay_slots < 0:
            raise ValueError("scalar.control_flow_delay_slots must be non-negative")
        if self.scale.num_ereg <= 0:
            raise ValueError("scale.num_ereg must be positive")
        if self.scale.ereg_bits <= 0:
            raise ValueError("scale.ereg_bits must be positive")
        if self.dma.channel_count <= 0:
            raise ValueError("dma.channel_count must be positive")
        if self.dma.alignment_bytes <= 0:
            raise ValueError("dma.alignment_bytes must be positive")
        if self.dma.offchip_command_words < 0:
            raise ValueError("dma.offchip_command_words must be non-negative")
        if self.tensor.num_mreg <= 0:
            raise ValueError("tensor.num_mreg must be positive")
        if self.tensor.mreg_rows <= 0 or self.tensor.mreg_row_bytes <= 0:
            raise ValueError("tensor register geometry must be positive")
        if self.tensor.mxu_count <= 0:
            raise ValueError("tensor.mxu_count must be positive")
        if self.tensor.weight_slots_per_mxu <= 0:
            raise ValueError("tensor.weight_slots_per_mxu must be positive")
        if self.tensor.weight_tile_rows <= 0 or self.tensor.weight_tile_cols_fp8 <= 0:
            raise ValueError("tensor weight-tile geometry must be positive")
        if self.tensor.vmem_alignment_bytes <= 0:
            raise ValueError("tensor.vmem_alignment_bytes must be positive")
        if self.tensor.matmul_latency_cycles <= 0:
            raise ValueError("tensor.matmul_latency_cycles must be positive")
        if self.vpu.simple_op_latency_cycles <= 0:
            raise ValueError("vpu.simple_op_latency_cycles must be positive")
        if self.vpu.non_pipelineable_op_latency_cycles <= 0:
            raise ValueError("vpu.non_pipelineable_op_latency_cycles must be positive")
        if self.xlu.transpose_latency_cycles <= 0:
            raise ValueError("xlu.transpose_latency_cycles must be positive")
        if self.bandwidth.offchip_link_width_bits <= 0 or self.bandwidth.offchip_link_width_bits % 8 != 0:
            raise ValueError(
                "bandwidth.offchip_link_width_bits must be a positive multiple of 8"
            )
        if self.bandwidth.offchip_link_core_cycles_per_beat <= 0:
            raise ValueError(
                "bandwidth.offchip_link_core_cycles_per_beat must be positive"
            )
        if self.bandwidth.vmem_bus_width_bits <= 0 or self.bandwidth.vmem_bus_width_bits % 8 != 0:
            raise ValueError(
                "bandwidth.vmem_bus_width_bits must be a positive multiple of 8"
            )
        if self.bandwidth.vmem_bus_core_cycles_per_beat <= 0:
            raise ValueError(
                "bandwidth.vmem_bus_core_cycles_per_beat must be positive"
            )
        if self.trace.ticks_per_cycle <= 0:
            raise ValueError("trace.ticks_per_cycle must be positive")
        for region_name, region in (
            ("imem", self.memory_map.imem),
            ("vmem", self.memory_map.vmem),
            ("dram", self.memory_map.dram),
        ):
            if region.size <= 0:
                raise ValueError(f"memory_map.{region_name}.size must be positive")
        if self.memory_backend.dram_page_size_bytes <= 0:
            raise ValueError("memory_backend.dram_page_size_bytes must be positive")
        if self.initialization.seed < 0:
            raise ValueError("initialization.seed must be non-negative")

    @property
    def mreg_bytes(self) -> int:
        """Total byte count stored in one tensor register."""

        return self.tensor.mreg_rows * self.tensor.mreg_row_bytes

    @property
    def mreg_fp8_cols(self) -> int:
        """Number of FP8 elements in one tensor-register row."""

        return self.tensor.mreg_row_bytes

    @property
    def mreg_bf16_cols(self) -> int:
        """Number of BF16 elements in one tensor-register row."""

        return self.tensor.mreg_row_bytes // 2

    @property
    def weight_slot_bytes(self) -> int:
        """Total byte count stored in one MXU weight slot."""

        return self.tensor.weight_tile_rows * self.tensor.weight_tile_cols_fp8

    @property
    def accum_buffer_bytes(self) -> int:
        """Total byte count stored in one MXU accumulation buffer."""

        return self.matmul_result_bytes

    @property
    def matmul_result_rows(self) -> int:
        """Number of rows in one MXU result tile."""

        return self.tensor.mreg_rows

    @property
    def matmul_result_cols(self) -> int:
        """Number of BF16 columns in one full MXU result tile."""

        return self.tensor.weight_tile_cols_fp8

    @property
    def matmul_result_bytes(self) -> int:
        """Total byte count in one full BF16 MXU result tile."""

        return self.matmul_result_rows * self.matmul_result_cols * 2

    @property
    def matmul_result_registers(self) -> int:
        """Number of architectural tensor registers used by one BF16 MXU result tile."""

        if self.matmul_result_bytes % self.mreg_bytes != 0:
            raise ValueError("matmul result does not map to an integer number of tensor registers")
        return self.matmul_result_bytes // self.mreg_bytes

    @property
    def vload_latency_cycles(self) -> int:
        """Modeled latency of one `vload` instruction in core cycles."""

        return self.bandwidth.vmem_transfer_cycles(self.mreg_bytes)

    @property
    def vstore_latency_cycles(self) -> int:
        """Modeled latency of one `vstore` instruction in core cycles."""

        return self.bandwidth.vmem_transfer_cycles(self.mreg_bytes)

    @property
    def vmatpush_weight_latency_cycles(self) -> int:
        """Modeled latency of one `vmatpush.weight.mxu*` tensor-to-weight transfer."""

        return self.bandwidth.vmem_transfer_cycles(self.weight_slot_bytes)

    @property
    def vmatpush_acc_latency_cycles(self) -> int:
        """Modeled latency of one BF16 tensor-pair to accumulator transfer."""

        return self.bandwidth.vmem_transfer_cycles(self.accum_buffer_bytes)

    @property
    def vmatpop_acc_bf16_latency_cycles(self) -> int:
        """Modeled latency of one BF16 accumulator export to a tensor-register pair."""

        return self.bandwidth.vmem_transfer_cycles(self.accum_buffer_bytes)

    @property
    def vmatpop_acc_fp8_latency_cycles(self) -> int:
        """Modeled latency of one FP8 accumulator export to a tensor register."""

        return self.bandwidth.vmem_transfer_cycles(self.mreg_bytes)

    @property
    def matmul_latency_cycles(self) -> int:
        """Modeled latency of one `vmatmul.*` instruction in core cycles."""

        return self.tensor.matmul_latency_cycles

    @property
    def vpu_simple_op_latency_cycles(self) -> int:
        """Modeled latency of one pipelineable VPU instruction in core cycles."""

        return self.vpu.simple_op_latency_cycles

    @property
    def vpu_non_pipelineable_op_latency_cycles(self) -> int:
        """Modeled latency of one non-pipelineable VPU instruction in core cycles."""

        return self.vpu.non_pipelineable_op_latency_cycles

    @property
    def xlu_transpose_latency_cycles(self) -> int:
        """Modeled latency of one `vtrpose.xlu` instruction in core cycles."""

        return self.xlu.transpose_latency_cycles

    def dma_offchip_cycles(self, payload_bytes: int) -> int:
        """Compute the off-chip serialized-link time for one DMA payload."""

        return self.bandwidth.dma_offchip_cycles(
            payload_bytes,
            self.dma.offchip_command_words,
        )

    def vmem_transfer_cycles(self, payload_bytes: int) -> int:
        """Compute the VMEM/system-bus time for one transfer payload."""

        return self.bandwidth.vmem_transfer_cycles(payload_bytes)

    def dma_transfer_cycles(self, payload_bytes: int) -> int:
        """Compute the modeled DMA completion time for one transfer payload."""

        return max(
            self.dma_offchip_cycles(payload_bytes),
            self.vmem_transfer_cycles(payload_bytes),
        )

    def with_memory_sizes(
        self,
        *,
        dram_size: int | None = None,
        vmem_size: int | None = None,
        imem_size: int | None = None,
    ) -> PenguinCoreConfig:
        """Return a copy of the config with selected memory capacities overridden."""

        return replace(
            self,
            memory_map=replace(
                self.memory_map,
                dram=replace(
                    self.memory_map.dram,
                    size=self.memory_map.dram.size if dram_size is None else dram_size,
                ),
                vmem=replace(
                    self.memory_map.vmem,
                    size=self.memory_map.vmem.size if vmem_size is None else vmem_size,
                ),
                imem=replace(
                    self.memory_map.imem,
                    size=self.memory_map.imem.size if imem_size is None else imem_size,
                ),
            ),
        )


DEFAULT_PENGUIN_CORE_CONFIG = PenguinCoreConfig()


__all__ = [
    "BandwidthConfig",
    "DEFAULT_PENGUIN_CORE_CONFIG",
    "DMAConfig",
    "InitializationConfig",
    "MemoryBackendConfig",
    "MemoryMapConfig",
    "MemoryRegionConfig",
    "PenguinCoreConfig",
    "ScaleConfig",
    "ScalarCoreConfig",
    "TensorCoreConfig",
    "TraceConfig",
    "XLUConfig",
    "VPUConfig",
]
