# Penguin Shared Configuration Parameters

Status: Working Baseline 0.1

## 1. Purpose

This document collects the configuration parameters that are shared across software,
functional modeling, RTL, and test collateral.

The goal is to keep the baseline machine shape explicit. These values are more stable
than local implementation details, but more concrete than the high-level architecture
document.

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
| `WEIGHT_SLOTS_PER_MXU` | 2 | - | Number of architected weight slots per MXU |
| `WEIGHT_TILE_ROWS` | 32 | rows | Rows in one MXU weight tile |
| `WEIGHT_TILE_COLS_FP8` | 16 | elements | FP8 columns in one MXU weight tile |
| `WEIGHT_SLOT_BYTES` | 512 | bytes | Bytes per MXU weight slot |
| `DMA_CHANNELS` | 8 | - | Number of architected DMA channels |
| `DMA_ALIGN` | 32 | bytes | Required DMA source/destination alignment |
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

## 4. Rationale For What Is Frozen

The baseline freezes quantities that shape program layout, register interpretation, or
memory images:

- instruction width and alignment affect IMEM contents and binary tooling
- tensor-register geometry affects compiler tiling, VMEM layout, and RTL storage
- weight-slot geometry affects MXU program shape and test-vector packing
- DMA channel count and alignment affect synchronization semantics and memory images
- memory-map constants affect program loading and runtime addressing

## 5. Deliberately Not Frozen Yet

The following classes of parameters are intentionally excluded from this document for now:

- clock frequency targets
- off-chip bus protocol widths
- detailed VPU lane counts beyond what is implied by row size
- internal crossbar widths and arbitration policy
- exact long-chime pipeline depths

Those are still implementation parameters or pending design-space questions rather than
shared architectural constants.
