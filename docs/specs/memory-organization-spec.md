# Penguin Memory Organization Specification

Status: Working Baseline 0.3

## 1. Purpose

This document defines the first formal memory organization for Penguin.

The design goal is to keep the asynchronous boundary narrow and explicit:

- DRAM access is off-chip, low-bandwidth, and asynchronous
- on-chip execution remains deterministic
- tensor registers do not access DRAM directly
- DMA is the only architected path between DRAM and on-chip data memory

## 2. Finalized Memory Decisions And Rationale

This section records the memory-organization choices that are now treated as fixed for
the current baseline.

### 2.1 Three-level memory split

Decision:

- DRAM stores backing data
- IMEM stores instructions
- VMEM stores on-chip tensor/vector data

Rationale:

- separates instruction fetch from tensor data traffic
- creates one clear on-chip staging memory for tensor operations
- narrows the async boundary to DRAM-facing traffic

### 2.2 DMA as raw-byte transport

Decision:

- DMA moves unit-stride raw byte ranges only
- DMA does not interpret tensor shape, dtype, or layout

Rationale:

- keeps DMA simple and stable
- leaves tensor interpretation to software and VMEM-facing tensor instructions
- avoids coupling the DMA engine to evolving tensor formats

### 2.3 Async boundary at DRAM <-> VMEM

Decision:

- DMA between DRAM and VMEM is asynchronous
- DMA completion is fenced explicitly by channel
- VMEM-facing transfers are blocking

Rationale:

- concentrates async complexity in one place
- keeps on-chip transfers deterministic and easier to verify
- fits the statically scheduled execution model

### 2.4 Byte-addressed IMEM and VMEM

Decision:

- IMEM is byte-addressed
- VMEM is byte-addressed
- IMEM fetch is 4-byte aligned

Rationale:

- keeps the memory model conventional
- matches fixed-width 32-bit instruction fetch naturally
- leaves VMEM alignment optimization for a later pass

### 2.5 Scalar-register indirect addressing with one shared base CSR

Decision:

- DMA, `vload` / `vstore`, and `mxu.push.*` use scalar-register indirect addresses
- one shared memory-base CSR extends addressing beyond the 32-bit scalar range

Rationale:

- preserves the “one 32-bit word per instruction” rule
- avoids adding a second descriptor register file
- keeps the scalar core in charge of address generation

### 2.6 Whole-register tensor transfers

Decision:

- `vload` and `vstore` move whole `m` registers only
- `mxu.push.*` moves a whole weight tile from VMEM into a selected MXU weight slot

Rationale:

- matches the whole-tile tensor execution model
- avoids sub-window semantics in the first ISA cut
- simplifies VMEM transfer logic and compiler lowering

### 2.7 Explicit DMA visibility and ordering rules

Decision:

- each DMA channel exposes independent busy/completion state
- `dma.wait.chN` completes immediately if channel `N` is already idle
- completion order across different DMA channels is not architecturally ordered

Rationale:

- makes channelized async behavior precise enough for software and RTL
- avoids accidental reliance on issue order across independent channels
- keeps fencing local to the resource that software actually depends on

### 2.8 Host-populated IMEM

Decision:

- IMEM is populated before execution by host-side software or firmware
- the accelerator does not fetch executable instructions directly from DRAM in the
  baseline contract

Rationale:

- keeps program-loading semantics simple and explicit
- avoids inventing an instruction-fetch DMA path before the execution model is stable

## 3. Memory Structures

Penguin defines three primary memory structures:

- DRAM: off-chip main data storage
- IMEM: on-chip instruction memory
- VMEM: on-chip tensor/vector data memory

### 3.0 Fixed memory map

The baseline memory map is fixed as follows:

- IMEM base: `0x0010_0000`
- IMEM size: `32 KiB`
- VMEM base: `0x0800_0000`
- VMEM size: `1 MiB`
- DRAM base: `0x8000_0000`
- DRAM size: `16 GiB`

Unless a later revision states otherwise, software, the functional model, and RTL
should all treat these base addresses and capacities as the architectural baseline.

### 3.1 DRAM

DRAM is the backing store for program data.

Architectural properties:

- all persistent data resides in DRAM
- DRAM occupies the byte range starting at `0x8000_0000`
- DRAM capacity is `16 GiB`
- DRAM connectivity to the chip is bandwidth-limited relative to on-chip execution
- DRAM accesses are not cycle-deterministic at instruction granularity

### 3.2 IMEM

IMEM stores program instructions.

Architectural properties:

- the instruction fetch path reads from IMEM
- IMEM begins at `0x0010_0000`
- IMEM capacity is `32 KiB`
- IMEM is byte-addressed
- instructions are fixed-width 32-bit words
- all instruction classes fit within one 32-bit word
- compute instructions are not fetched directly from DRAM
- IMEM is populated before execution by host-side software or firmware
- IMEM instruction fetch requires 4-byte alignment

### 3.3 VMEM

VMEM stores tensor/vector data that has been staged on chip.

Architectural properties:

- VMEM is the only on-chip data memory directly visible to tensor load/store style
  operations
- VMEM begins at `0x0800_0000`
- VMEM capacity is `1 MiB`
- VMEM is byte-addressed
- tensor registers interact with VMEM, not DRAM
- partial matrix tiles are staged in VMEM before being consumed by the compute units
- alignment requirements for VMEM accesses are intentionally deferred to a later revision

## 4. Data Movement Paths

### 4.1 DRAM <-> VMEM

Movement between DRAM and VMEM occurs through DMA.

Architectural properties:

- DMA is the only architected path between DRAM and VMEM
- DMA operations are asynchronous
- software must use explicit synchronization before assuming DMA completion
- the async barrier for off-chip data access is placed at DMA
- DMA operates on raw byte ranges only
- DMA is unit-stride and does not interpret tensor shape, layout, or element type
- DMA addressing is register-indirect through scalar registers
- one shared memory-base CSR extends DMA addressing beyond the 32-bit scalar-register
  range
- DMA synchronization is channel-specific rather than global
- DMA source and destination addresses must be 32-byte aligned
- DMA transfer sizes must be integer multiples of 32 bytes

Initial intended direction:

- DMA operations are issued on named channels such as `dma.load.ch1`
- completion is fenced with matching wait operations such as `dma.wait.ch1`
- the first architecture revision exposes 8 DMA channels
- the 8 DMA channels are symmetric in capability
- `dma.wait.chN` returns immediately if `chN` is already idle
- completion order across different channels is not guaranteed by issue order

### 4.2 VMEM <-> Tensor Registers

Movement between VMEM and tensor registers occurs through explicit load/store-style
operations.

Architectural properties:

- `m` registers can only access VMEM
- data must first be brought from DRAM into VMEM before it can be loaded into `m`
  registers
- storing tensor results back to DRAM requires a two-step path: `m` registers to VMEM,
  then DMA from VMEM to DRAM
- VMEM-facing tensor transfers use scalar-register indirect addressing
- one shared memory-base CSR extends VMEM addressing beyond the 32-bit scalar-register
  range

Initial intended direction:

- `vload` moves data from VMEM into `m` registers
- `vstore` moves data from `m` registers into VMEM
- `vload` and `vstore` operate on whole `m` registers only
- each `vload` or `vstore` transfers exactly one 2048-byte tensor register image
- sub-tile windows and partial-row transfers are not part of the current contract

These operations are on-chip and block execution if the transfer is not ready in time.
They do not create a separate asynchronous completion model.

### 4.3 VMEM <-> MXU Weight Slots

Weight-stationary MXU operation introduces one additional on-chip path:

- `mxu.push.*` loads weight data from VMEM into an MXU-local weight slot

Architectural properties:

- `mxu.push.*` does not read directly from DRAM
- DRAM-sourced weights must first be staged into VMEM by DMA
- the async fence is still associated with DMA completion, not with on-chip `mxu.push.*`
- `mxu.push.*` uses scalar-register indirect VMEM addressing
- one shared memory-base CSR extends `mxu.push.*` addressing beyond the 32-bit
  scalar-register range
- each `mxu.push.*` transfers exactly one 512-byte weight tile into the selected slot

## 5. Determinism And Synchronization

Penguin uses two memory-completion models.

### 5.1 Asynchronous completion

Asynchronous completion applies to:

- DMA between DRAM and VMEM

Rule:

- software must use explicit resource-specific waits/fences before consuming data whose
  correctness depends on DMA completion
- DMA waits fence individual channels rather than all outstanding DMA traffic
- a successful `dma.wait.chN` guarantees that the destination bytes of the transfer
  previously issued on channel `N` are architecturally visible
- software must not assume any completion ordering between different DMA channels unless
  it has inserted the required waits

### 5.2 Blocking completion

Blocking completion applies to:

- tensor register loads from VMEM
- tensor register stores to VMEM
- on-chip weight-slot loads from VMEM into MXU-local weight storage

Rule:

- these operations may stall execution
- they do not require a separate asynchronous completion fence

## 6. Hazard-Ordering Rules

The baseline memory contract is software scheduled. Programs must explicitly sequence
memory hazards across DMA and on-chip tensor transfers.

Mandatory rules:

- a program must not issue `vload`, `vstore`, or `mxu.push.*` against VMEM bytes whose
  correctness depends on an outstanding DMA transfer until the matching `dma.wait.chN`
  has completed
- a program must not assume that two DMA channels become visible in issue order
- a program must not overlap producer and consumer operations on the same VMEM byte range
  without an explicit ordering point

If software violates these rules, behavior is architecturally undefined in this revision.

## 7. Architectural Access Rules

The following access rules are mandatory in this revision:

- scalar instruction fetch reads IMEM
- tensor registers do not read DRAM directly
- tensor registers do not write DRAM directly
- tensor registers access VMEM only
- VPU reads from and writes to `m` registers, not memory directly
- MXU weight slots are populated from VMEM, not DRAM directly

## 8. Example Dataflow

For an activation tile:

1. DMA activation data from DRAM to VMEM
2. wait for DMA completion
3. `vload` from VMEM into `m<src>`
4. compute using MXU or VPU
5. `vstore` result tile from `m<dest>` back to VMEM
6. DMA result tile from VMEM back to DRAM if needed
7. wait for DMA completion if software depends on the stored result

For a weight tile:

1. DMA weight data from DRAM to VMEM
2. wait for DMA completion
3. `mxu.push.*` from VMEM into `w0` or `w1`
4. launch `matmul.*`

## 9. Remaining Open Memory Items

This revision still leaves these items open:

- the exact `vload` / `vstore` instruction encodings
- VMEM alignment requirements
- whether VMEM is logically unified or internally partitioned by traffic class
