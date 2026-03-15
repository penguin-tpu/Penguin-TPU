# Penguin Microarchitecture Specification

Status: Working Baseline 0.3

## 1. Purpose

This document defines the intended microarchitectural direction for Penguin. It refines
the architecture-level goals in [architecture-spec.md](/home/tk/Desktop/Penguin-TPU/docs/specs/architecture-spec.md) into an implementation strategy that is still flexible enough for design-space exploration.

This is an initial design-iteration document. It captures the currently intended
organization and explicitly calls out unresolved choices.

## 2. Finalized Microarchitectural Decisions And Rationale

This section captures the microarchitectural decisions that are currently treated as the
 baseline implementation direction.

### 2.1 Narrow frontend

Decision:

- one fetch stream
- one decode stream
- one issue decision per cycle
- fixed-width 32-bit instruction fetch

Rationale:

- minimizes frontend complexity
- aligns with the non-VLIW architectural choice
- shifts throughput pressure to long-chime compute units instead of fetch bandwidth

### 2.2 Concurrent long-chime execution

Decision:

- `mxu0`, `mxu1`, and VPU may be active concurrently
- only one new instruction may be issued in a cycle

Rationale:

- preserves the single-issue frontend while still enabling overlap
- matches the compiler-scheduled latency-hiding model

### 2.3 Fully connected tensor access

Decision:

- a central crossbar connects tensor registers to all functional units
- every functional unit can reach every tensor register

Rationale:

- avoids exposing bank partitioning as an architectural rule
- simplifies software scheduling and operand naming
- leaves bandwidth/arbitration as a microarchitectural optimization problem

### 2.4 Distinct dual-MXU implementations

Decision:

- `mxu0` implements systolic-array accumulation
- `mxu1` implements inner-product-tree accumulation

Rationale:

- makes direct implementation comparison part of the design, not an afterthought
- keeps the software-visible contract identical across both engines

### 2.5 VMEM-centered data staging

Decision:

- VMEM is the sole on-chip data staging memory for tensor traffic
- `m` registers talk to VMEM only
- `mxu.push.*` also sources from VMEM

Rationale:

- isolates the async boundary at DRAM <-> VMEM
- keeps on-chip tensor dataflow deterministic and easier to model
- prevents compute instructions from inheriting DRAM variability

### 2.6 Blocking on-chip transfers, async off-chip transfers

Decision:

- DMA is asynchronous
- VMEM-facing `vload`, `vstore`, and `mxu.push.*` are blocking

Rationale:

- keeps async completion semantics narrow and explicit
- avoids polluting all data movement with fence logic
- simplifies both the functional model and the first RTL control path

### 2.7 Structural conflicts are absorbed by stalls, not partial retirement

Decision:

- bank conflicts, crossbar contention, or destination-port conflicts are resolved by
  stalling or arbitration
- these conflicts are not allowed to create architecturally visible partial-tile results

Rationale:

- keeps internal storage implementation hidden from software
- avoids fragile compiler rules around partially written destination tiles
- makes RTL and model comparison significantly cleaner

### 2.8 Minimal execution-control and observability plane

Decision:

- the microarchitecture shall expose enough control and status state to support host
  launch, halt, completion, and DMA-busy observation

Rationale:

- IMEM population, execution start, and completion need an explicit system contract
- this state is required for bring-up even before tensor execution is complete

## 3. High-Level Organization

The intended Penguin microarchitecture is organized around four major subsystems:

- a single-instruction scalar fetch, decode, and issue frontend
- a scalar execution path for control and orchestration
- a tensor register file with 64 architectural tile registers
- long-latency tensor functional units: `mxu0`, `mxu1`, and VPU

The scalar path is responsible for launching long-chime tensor work. The tensor units are
responsible for sustaining throughput over many cycles after a single issue event.

Multiple long-chime units may therefore be active concurrently even though only one new
instruction is issued in any given cycle.

## 4. Frontend And Issue

### 4.1 Frontend structure

The frontend is intentionally narrow:

- one fetch stream
- one decode stream
- one issue decision per cycle, subject to stalls
- at most one scalar or tensor instruction launch per cycle
- fixed-width 32-bit instruction fetch
- no multiword instruction fetch should be required by the ISA

This choice is intended to reduce instruction footprint and frontend complexity relative
to a TPU-style VLIW frontend.

### 4.2 Issue policy

Penguin is statically scheduled, but hardware still checks whether the destination
functional unit is currently able to accept a new instruction.

Microarchitectural requirements:

- issue logic shall track busy status for long-latency units
- if an instruction targets a busy unit, issue stalls until the unit becomes available
- if an instruction is legal and the targeted unit is ready, it may issue without dynamic
  rescheduling

This is not out-of-order execution. It is correctness-preserving issue gating around a
statically scheduled program.

The current design direction is:

- on-chip execution uses deterministic unit latency and busy tracking
- off-chip memory activity is asynchronous
- explicit synchronization is required only for off-chip memory completion
- synchronization should be resource-specific rather than one global wait
- `mxu0` and `mxu1` may execute concurrently, but they cannot both be issued in the same
  cycle
- memory-facing operations use scalar-register indirect addressing plus a CSR high-address
  base offset
- the CSR high-address base offset is shared across memory-like instructions
- DMA synchronization is channel-specific, with waits targeting individual DMA channels
- the first architecture revision exposes 8 DMA channels
- the 8 DMA channels are symmetric in capability

### 4.3 Frontend phase model

The baseline frontend is intended to be explainable in a small number of phases:

- IFG: instruction-address generation and IMEM request launch
- IFR: instruction return from IMEM
- ID: decode and dispatch
- EX: scalar execution or long-chime unit launch

This phase model is not meant to force a specific RTL partitioning, but it does capture
the timing intuition behind the 2-delay-slot branch model and the one-instruction-per-
cycle issue rule.

## 5. Tensor Register File

The tensor register file is the key datapath structure for accelerator execution.

Microarchitectural requirements:

- 64 architectural tensor registers
- tensor registers are addressed as `m0` through `m63`
- each register stores one 2D tile
- each tile has 64 rows
- each row stores 32 bytes
- the register file is physically shared across tensor data types
- instruction semantics define how a given operation interprets row contents
- both MXU and VPU consume tensor operands in row order
- a central crossbar connects tensor registers to the functional units
- every functional unit can access every tensor register through that crossbar
- the tensor register file must support row-wise readout at the throughput required by
  the target functional unit

The exact porting and arbitration structure behind the crossbar is not fixed in this
revision, but register reachability is intended to be fully connected.

Physical banking remains an implementation detail. If the chosen storage organization
cannot serve a requested access pattern in one cycle, the microarchitecture must stall or
arbitrate without exposing partial architectural completion.

## 6. Long-Chime Execution

Long-chime execution is central to Penguin.

### 6.1 Definition

A long-chime operation is a multi-cycle tensor instruction that sweeps over one or more
rows of one or more tensor registers after a single issue event.

### 6.2 Consequences

Microarchitectural support is expected for:

- unit-local sequencing across tile rows
- deterministic per-instruction total latency
- architectural busy indication while a long-chime operation is in flight
- scalar issue overlap when no architectural hazard exists

These determinism expectations apply to on-chip execution, not to off-chip memory
latency.

### 6.3 Scheduling intention

The compiler is expected to place scalar instructions into the useful slack created by
long-chime tensor instructions. The microarchitecture should support that overlap without
requiring multi-instruction issue.

## 7. Control-Transfer Timing

The baseline architecture now requires 2 branch/jump delay slots. The functional model
and the frontend microarchitecture must preserve that 2-slot rule.

Future work may revisit whether the frontend implementation should expose additional
internal control speculation, but it must still preserve the 2 architectural delay
slots visible to software.

## 8.1 Execution-control state and reset expectations

The microarchitecture shall maintain at least the following control and observability
state:

- execution enable / disable state
- current `pc`
- halt / done / trap outcome state
- DMA busy indication for all 8 channels

On reset, execution is disabled, DMA channels are idle, and any in-flight issue-side
state is cleared. Tensor data arrays and weight slots need not be zeroed by reset unless
some later integration requirement states otherwise.

## 9. Matrix Execution Units

### 9.1 Functional role

Penguin contains two distinct MXU instances:

- `mxu0`: systolic-array-based
- `mxu1`: inner-product-tree-based

Both MXUs perform low-precision floating-point MAC over tile operands.

Microarchitectural expectations:

- row-wise operand consumption from the tensor register file
- deterministic instruction latency
- support for long-chime operation
- architecturally visible completion only after the full tile sweep is complete
- MXU operand selection is whole-register at the architectural level
- weight-stationary scheduling is the intended MXU dataflow
- the MXU exposes two architecture-visible weight selectors, `w0` and `w1`
- `mxu0.w0/w1` and `mxu1.w0/w1` are distinct physical state
- processing elements may retain weights internally and reuse them across multiple cycles
- `w0` and `w1` select one of two PE-resident weight-buffer entries
- weight slots are loaded through explicit MXU push operations such as
  `mxu.push.mxu0 w0`
- MXU push operations source weight data from VMEM rather than from the `m0..m63`
  tensor register file
- MXU push operations are on-chip and blocking rather than asynchronously fenced
- the architecture-visible MXU contract allows accumulation of matmul partial sums across
  K-sweeps
- accumulation mode is expected to be explicit in the issued opcode rather than inferred
  from destination register state
- the architecture-visible MXU contract excludes fused tensor postprocessing such as bias
  add inside the MXU

### 9.2 Dual-MXU design goal

The project explicitly intends to compare two coexisting MXU organizations:

1. `mxu0`: systolic-array-style accumulation
2. `mxu1`: parallel inner-product-tree accumulation

The reason for carrying both options is to enable direct comparison of:

- area
- energy efficiency
- throughput implications
- control complexity

The architecture should remain stable across both engines so that the comparison is
meaningful.

### 9.3 Low-precision MAC

The MXU uses low-precision floating-point MACs.

Current intended arithmetic contract:

- multiplicands use `FP8_e4m3`
- accumulation uses BF16
- an optional writeback mode may quantize results back to FP8
- BF16-to-FP8 quantization uses round-to-nearest-even
- BF16-to-FP8 quantization saturates on overflow
- a scalar output scale factor is applied per workload-level matmul
- scaling is applied on the output path rather than on MXU input fetch

## 10. Vector Processing Unit

The VPU is a row-wise tensor unit that shares the tensor-register abstraction with the
MXU.

Microarchitectural expectations:

- row-wise operand readout
- VPU operands are read directly from the tensor register file through the shared crossbar
- VPU results are written back to the tensor register file rather than directly to memory
- VPU operand selection is whole-register at the architectural level
- deterministic latency
- long-chime execution model
- first implementation target is elementwise operations
- initial intended opcode floor is `vadd`, `vmul`, `vmax`, `vmin`, `vrelu`, and `vmov`
- later scope may include additional row-wise or reduction-adjacent operations

The VPU exists to handle tensor work that is not best mapped onto the MXU.

## 11. DMA And Memory Stall Behavior

The first microarchitecture should make DMA behavior explicit enough for model and RTL
alignment.

Microarchitectural expectations:

- 8 architected DMA channels each expose a logical busy state
- a `dma.wait.chN` instruction may stall the frontend until channel `N` becomes idle
- while a `dma.wait.chN` stalls issue, older in-flight long-chime operations may continue
  draining
- completion order across different DMA channels need not match issue order
- request queues, transaction counters, and request/response bookkeeping are
  microarchitectural choices rather than architecture-visible contracts

The important invariant is that a channel-specific wait only releases once the
architecture-visible destination bytes are valid for the transfer issued on that channel.

## 12. Memory-System Direction

The memory structure is an explicit intermediate milestone between scalar-core bring-up
and matrix acceleration.

Microarchitectural implication:

- the team should not optimize matrix execution in isolation before the memory path is
  well-defined

The future memory structure must support:

- scalar load/store behavior already defined by the scalar spec
- byte-addressed IMEM for instruction fetch
- fixed-width 32-bit instruction storage in IMEM
- movement of tile data between DRAM and VMEM through DMA
- movement of tile data into and out of the tensor register file through VMEM-facing
  load/store operations
- raw-byte, unit-stride DMA transport without tensor-shape interpretation in the DMA path
- whole-register VMEM-facing tensor transfers in the first design cut
- byte-addressed VMEM
- scalar-register indirect memory addressing for DMA and tensor-memory operations
- CSR-based high-address extension for memory regions beyond the 32-bit scalar range
- one shared memory-base CSR rather than separate per-memory-space base CSRs
- deterministic behavior suitable for static scheduling
- concurrent MXU and VPU operation when bandwidth and structural resources permit
- asynchronous off-chip transfers
- explicit completion synchronization for off-chip transfers
- channel-specific DMA completion waits
- blocking on-chip transfers between VMEM and compute-local storage

Software should still schedule with structural resource limits in mind, but tensor
register reachability itself should not be restricted by static register partitioning.

Formal details are further defined in
[memory-organization-spec.md](/home/tk/Desktop/Penguin-TPU/docs/specs/memory-organization-spec.md).

## 13. Performance And Comparison Goals

The microarchitecture is intended to support disciplined comparison rather than ad hoc
tuning.

At minimum, the design flow should eventually compare `mxu0` and `mxu1` on:

- cycle count
- utilization
- LUT, FF, BRAM, and DSP cost
- achievable clock frequency
- energy or power proxies when available

## 14. Remaining Open Microarchitectural Items

This revision still leaves these items open:

- tensor crossbar bandwidth and read-port structure
- exact long-chime pipeline depth per unit
- exact busy-bit or scoreboard structure
- exact scalar-to-tensor hazard rules
- exact DMA instruction shapes
- exact VMEM-facing tensor load/store instruction shapes
- exact VMEM alignment requirements

## 15. Immediate Next Spec Work

The next design-spec iterations should define:

- the memory structure formally
- the tensor instruction set
- the executable-package format
- the tile data-layout rules
- the criteria for comparing `mxu0` versus `mxu1`
