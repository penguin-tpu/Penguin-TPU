# Penguin Shared Configuration Parameters

Status: Working Baseline 0.2

## 1. Purpose

This document collects the configuration parameters that are shared across software,
functional modeling, RTL, and test collateral.

The goal is to keep the baseline machine shape explicit. These values are more stable
than local implementation details, but more concrete than the high-level architecture
document.

In the Python functional/performance model, these parameters are instantiated through one
hierarchical entry point, `PenguinCoreConfig`. That software-model contract is not a
separate architecture-visible register block, but it is the required way to construct a
concrete Penguin machine instance in the reference model.

## 2. Frozen Baseline Parameters

The following parameters are part of the current Penguin baseline.

| Parameter | Value | Unit | Meaning |
|---|---:|---|---|
| `INSN_WIDTH` | 32 | bits | Fixed instruction width |
| `INSN_ALIGN` | 4 | bytes | Instruction-fetch alignment |
| `SCALAR_WIDTH` | 32 | bits | Width of scalar general-purpose registers |
| `NUM_XREG` | 32 | - | Number of scalar general-purpose registers |
| `NUM_MREG` | 64 | - | Number of tensor registers |
| `MREG_ROWS` | 64 | rows | Number of rows per tensor register |
| `MREG_ROW_BYTES` | 32 | bytes | Bytes per tensor-register row |
| `MREG_BYTES` | 2048 | bytes | Total bytes per tensor register |
| `MXU_COUNT` | 2 | - | Number of architected MXUs |
| `XLU_COUNT` | 1 | - | Number of architected transpose units |
| `WEIGHT_SLOTS_PER_MXU` | 2 | - | Number of architected weight slots per MXU |
| `WEIGHT_TILE_ROWS` | 32 | rows | Rows in one MXU weight tile |
| `WEIGHT_TILE_COLS_FP8` | 16 | elements | FP8 columns in one MXU weight tile |
| `WEIGHT_SLOT_BYTES` | 512 | bytes | Bytes per MXU weight slot |
| `DMA_CHANNELS` | 8 | - | Number of architected DMA channels |
| `DMA_ALIGN` | 32 | bytes | Required DMA source/destination alignment |
| `OFFCHIP_LINK_WIDTH_BITS` | 32 | bits | Serialized DRAM link width |
| `OFFCHIP_LINK_CORE_CYCLES_PER_BEAT` | 2 | core cycles | Time for one off-chip serialized beat |
| `DMA_OFFCHIP_COMMAND_WORDS` | 2 | 32-bit words | DMA off-chip command overhead: op type + DRAM address |
| `VMEM_BUS_WIDTH_BITS` | 128 | bits | On-chip VMEM/system-bus width for tensor and DMA traffic |
| `VMEM_BUS_CORE_CYCLES_PER_BEAT` | 1 | core cycles | Time for one VMEM/system-bus beat |
| `VMEM_TENSOR_ALIGN` | 32 | bytes | Required alignment for `vload`, `vstore`, and `mxu.push.*` |
| `DRAM_BASE` | `0x8000_0000` | byte address | DRAM base address |
| `DRAM_SIZE` | `16 GiB` | bytes | DRAM capacity |
| `IMEM_BASE` | `0x0010_0000` | byte address | IMEM base address |
| `IMEM_SIZE` | `32 KiB` | bytes | IMEM capacity |
| `VMEM_BASE` | `0x0800_0000` | byte address | VMEM base address |
| `VMEM_SIZE` | `1 MiB` | bytes | VMEM capacity |

## 3. Tensor Register Views

Because the tensor register file is byte-defined rather than element-defined, the element
shape depends on the instruction-selected view.

| View | Elements per row | Tile shape |
|---|---:|---|
| `FP8_e4m3` | 32 | `64 x 32` |
| `BF16` | 16 | `64 x 16` |

These views are shared by software, the functional model, and RTL. If additional views
are introduced later, they should be added here only after their architecture contract is
frozen.

## 4. Software-Model Configuration Entry Point

The Python functional/performance model shall expose one top-level configuration object:
`PenguinCoreConfig`.

Required structure:

- `scalar`: scalar architectural quantities such as register count and delay-slot count
- `memory_map`: DRAM, IMEM, and VMEM base addresses and capacities
- `memory_backend`: Python-model backing-store choices such as paged DRAM
- `dma`: DMA channel count, alignment, and off-chip command-overhead parameters
- `tensor`: tensor-register geometry, weight-slot geometry, and MXU latency parameters
- `bandwidth`: off-chip-link and VMEM/system-bus width and beat-timing parameters
- `trace`: trace timestamp granularity for the performance model

Required derived quantities:

- `MREG_BYTES`
- `WEIGHT_SLOT_BYTES`
- `vload` / `vstore` latency from VMEM bandwidth
- `mxu.push.*` latency from VMEM bandwidth
- DMA completion time from the slower of:
  - off-chip serialized-link time
  - VMEM/system-bus time

Compatibility rule:

- module-level constants such as `DRAM_BASE`, `VMEM_BASE`, `MREG_BYTES`, and
  `DMA_CHANNEL_COUNT` may remain as aliases of the default configuration
- the active runtime behavior of a concrete model instance must flow from the
  `PenguinCoreConfig` bound to that instance, not from unrelated global state

## 5. Rationale For What Is Frozen

The baseline freezes quantities that shape program layout, register interpretation, or
memory images:

- instruction width and alignment affect IMEM contents and binary tooling
- tensor-register geometry affects compiler tiling, VMEM layout, and RTL storage
- weight-slot geometry affects MXU program shape and test-vector packing
- DMA channel count and alignment affect synchronization semantics and memory images
- off-chip link width and serialized beat timing affect DMA completion time
- VMEM bus width affects DMA drain/fill time and blocking VMEM-facing tensor transfers
- VMEM tensor-operation alignment affects instruction legality and blob layout
- memory-map constants affect program loading and runtime addressing

## 6. Deliberately Not Frozen Yet

The following classes of parameters are intentionally excluded from this document for now:

- clock frequency targets
- detailed VPU lane counts beyond what is implied by row size
- internal crossbar widths and arbitration policy
- exact long-chime pipeline depths

Those are still implementation parameters or pending design-space questions rather than
shared architectural constants.

## 7. Derived Transfer Formulas

The current performance model derives transfer timing from the shared width parameters.

- off-chip DMA serialization:
  `dma_offchip_cycles(bytes) = ceil((bytes + 4 * DMA_OFFCHIP_COMMAND_WORDS) / (OFFCHIP_LINK_WIDTH_BITS / 8)) * OFFCHIP_LINK_CORE_CYCLES_PER_BEAT`
- VMEM/system-bus transfer time:
  `vmem_transfer_cycles(bytes) = ceil(bytes / (VMEM_BUS_WIDTH_BITS / 8)) * VMEM_BUS_CORE_CYCLES_PER_BEAT`
- baseline DMA completion model:
  `dma_transfer_cycles(bytes) = max(dma_offchip_cycles(bytes), vmem_transfer_cycles(bytes))`

For the current baseline values:

- each 32-bit off-chip beat costs 2 core cycles
- each 128-bit VMEM beat costs 1 core cycle
- `vload` / `vstore` of one 2048-byte tensor register take 128 cycles
- `mxu.push.*` of one 512-byte weight tile takes 32 cycles
