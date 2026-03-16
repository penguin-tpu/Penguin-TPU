# Penguin Architecture Specification

Status: Working Baseline 0.4

## 1. Purpose

This document defines the architecture-level direction for Penguin as a TPU-like
accelerator with a scalar control path and a tile-oriented compute datapath.

The target end state is not a general CPU with an attached matrix coprocessor. Penguin is
an accelerator-first design with:

- a scalar control path used for orchestration, address generation, and control flow
- a tensor register file holding two-dimensional tiles
- two matrix execution units (MXUs) for dense matrix-style arithmetic
- a vector processing unit (VPU) for row-wise elementwise or reduction-style work
- a cross-lane transpose unit (XLU) for transpose work
- a single-instruction fetch and issue frontend rather than a VLIW issue bundle

This specification captures architecture-visible behavior. Implementation tradeoffs and
candidate execution structures are defined separately in the microarchitecture
specification.

## 2. Architectural Design Goals

Penguin shall provide the following architecture-level properties:

- statically scheduled execution
- deterministic arithmetic latencies
- long-chime instructions that launch multi-cycle tile sweeps
- tile-oriented matrix and vector execution
- low instruction fetch bandwidth relative to a VLIW TPU frontend

The architectural intent is that compile-time scheduling should do most of the latency
hiding work. On-chip operations are intended to be cycle-deterministic. Off-chip memory
operations are intended to be asynchronous and require explicit synchronization.

## 3. Finalized Architectural Decisions And Rationale

This section records the decisions that are considered frozen for the current design
baseline, together with the reasoning behind them.

### 3.1 Single-issue frontend

Decision:

- Penguin fetches and issues one fixed-width 32-bit instruction at a time

Rationale:

- reduces frontend complexity and instruction bandwidth pressure
- keeps decode and scheduling logic explainable
- relies on long-chime tensor execution rather than multi-issue width for throughput

### 3.2 Flat tensor register file

Decision:

- Penguin exposes 64 tensor registers, `m0` through `m63`
- each register is `64 rows x 32 bytes`
- all tensor data types share the same architectural storage

Rationale:

- keeps the programming model simple
- avoids proliferating type-specific tensor storage classes
- makes tensor interpretation an instruction-level concern rather than a storage-layout
  concern

### 3.3 Whole-register tensor operations

Decision:

- VPU, MXU, `vload`, and `vstore` operate on whole tensor registers only

Rationale:

- simplifies the first ISA cut
- avoids introducing sub-tile window semantics too early
- matches the long-chime, tile-oriented execution model

### 3.4 Dual-MXU architecture

Decision:

- Penguin exposes two architecturally visible MXUs
- `mxu0` is systolic-array-based
- `mxu1` is inner-product-tree-based

Rationale:

- makes the area/energy/performance comparison a first-class architectural goal
- keeps both engines under the same software contract
- allows future benchmarking and implementation comparison without redefining the ISA

### 3.5 Pure matmul MXU contract

Decision:

- MXU performs matrix multiply/accumulate only
- MXU may accumulate partial sums across K-sweeps
- MXU does not fuse bias, residual, activation, or other tensor postprocessing

Rationale:

- preserves the TPU-like split between matrix and vector/tensor postprocessing work
- simplifies the MXU interface and comparison between `mxu0` and `mxu1`
- keeps fusion policy in software and VPU scheduling rather than inside matrix hardware

### 3.6 Weight-stationary execution with explicit weight slots

Decision:

- each MXU has two architecturally visible weight selectors, `w0` and `w1`
- `mxu0.w0/w1` and `mxu1.w0/w1` are distinct state
- weights are loaded explicitly with `mxu.push.*`

Rationale:

- makes weight residency explicit to software
- matches the intended PE-local reuse pattern
- keeps matrix instruction operands compact enough to fit in one 32-bit word

### 3.7 MXU numeric contract

Decision:

- MXU computes `FP8_e4m3 x FP8_e4m3 -> BF16`
- output-only scaling is applied per workload-level matmul
- optional BF16-to-FP8 writeback uses round-to-nearest-even with saturation on overflow

Rationale:

- low-precision multiplicands keep matrix hardware efficient
- BF16 accumulation provides a cleaner first numerical target than ultra-low-precision
  accumulation
- output-only scaling keeps the ISA and dataflow simpler than per-input scaling

### 3.8 Resource-specific async synchronization

Decision:

- off-chip memory completion is fenced explicitly
- synchronization is resource-specific rather than global
- DMA is channelized and fenced by channel

Rationale:

- preserves overlap between unrelated async transfers
- gives the compiler precise control over dependence points
- fits a statically scheduled machine better than global barriers

### 3.9 VPU as a register-to-register tensor unit

Decision:

- VPU reads from `m` registers and writes back to `m` registers only
- VPU has no MXU-style local operand buffers in the current contract
- first VPU opcode floor is `vadd`, `vmul`, `vmax`, `vmin`, `vrelu`, and `vmov`
- the initial floating-point VPU view is BF16 over the `64 x 16` interpretation of one
  whole tensor register

Rationale:

- keeps memory movement separate from tensor compute
- avoids creating a second hidden data-movement path
- gives a useful first elementwise floor without overcommitting to a large VPU ISA

### 3.10 XLU as a register-to-register transpose unit

Decision:

- Penguin includes one architecturally visible XLU
- XLU reads tensor registers and writes tensor registers only
- XLU is intended for transpose work only in the baseline design

Rationale:

- makes transpose-style data movement a first-class accelerator operation
- avoids forcing awkward transpose flows through VMEM or elementwise VPU sequences
- keeps tensor reordering separate from arithmetic units and DMA
- avoids overcommitting to a broader cross-lane function set before the workload requires
  it

### 3.11 Explicit execution-state contract

Decision:

- Penguin defines a finite set of architecture-visible execution state
- that state includes scalar registers, `pc`, tensor registers, MXU weight slots,
  execution-control state, DMA busy state, and the shared memory-base CSR

Rationale:

- makes software, functional model, RTL, and board bring-up agree on the same machine
  state
- keeps host control and accelerator-local execution semantics from being left implicit
- avoids burying architecturally relevant status in implementation-only notes

### 3.12 No partial architectural retirement on structural conflict

Decision:

- structural conflicts may stall issue or local execution progress
- structural conflicts must not expose partial-row or partial-tile architectural results
- younger instructions do not preempt older instructions mid-writeback in an
  architecture-visible way

Rationale:

- keeps correctness independent of internal banking and arbitration choices
- makes compiler scheduling and formal verification tractable
- deliberately avoids architecting partial-completion behavior as a software-visible rule

## 4. Architectural Components

### 4.1 Scalar control path

The scalar path is integer-only in the current direction of the project. The scalar ISA
subset is defined in [scalar-functional-subset.md](/home/tk/Desktop/Penguin-TPU/docs/specs/scalar-functional-subset.md).

In the Python functional/performance model, a concrete machine instance is created
through one hierarchical software configuration entry point, `PenguinCoreConfig`, whose
fields are constrained by the shared-parameter specification.

The scalar path is responsible for:

- control flow
- address generation
- scalar bookkeeping
- launching long-chime matrix and vector operations

### 4.1.1 Architectural execution state

The current baseline requires the following architecture-visible state:

- scalar integer registers `x0` through `x31`
- a 32-bit program counter `pc`
- tensor registers `m0` through `m63`
- per-engine weight slots `mxu0.w0`, `mxu0.w1`, `mxu1.w0`, and `mxu1.w1`
- one shared memory-base CSR used to extend memory-like addressing beyond the raw
  32-bit scalar-register range
- execution-control and execution-status state sufficient to start, stop, and observe the
  accelerator
- DMA channel busy state for the 8 architected DMA channels

This section defines the required logical state only. The exact CSR encodings and CSR
region base address are intentionally left to the SoC-integration specification.

The baseline does, however, require one host-visible CSR region containing a consecutive
logical block of at least:

- `CTRL`
- `STATUS`
- `PC`
- `MEM_BASE`
- `DMA_BUSY`
- scalar architectural registers `x0` through `x31`

### 4.1.2 Host launch and completion model

Penguin is intended to be launched by a host-side software or firmware environment.

Architecture-visible requirements:

- the host populates IMEM before accelerator execution begins
- the host arranges any required DRAM and VMEM initialization before the relevant program
  phase depends on it
- software must initialize any architecturally visible data it consumes; unread DRAM or
  VMEM contents are undefined by the architecture
- software must clear scalar registers at program entry before relying on register values
  not explicitly written by host-side setup or earlier instructions
- the accelerator begins fetching only after host-visible execution-control state enables
  execution
- completion or error-halt outcome must be observable through host-visible status state

The exact MMIO map is not frozen here, but the launch model is: the accelerator is not
self-booting and does not fetch its program directly from DRAM in the baseline design.

In the early bring-up model:

- the host is responsible for initializing CSR state such as `MEM_BASE`
- Penguin does not yet require a dedicated scalar CSR-manipulation ISA
- future revisions may allow Penguin scalar code to access the same CSR region through
  MMIO-style loads and stores or direct hardware connections where appropriate

### 4.1.3 Scalar binary encoding baseline

The first scalar binary encoding baseline is now defined.

Architecture-visible requirements:

- scalar instructions remain fixed-width 32-bit words
- scalar instructions use standard RV32I field layouts:
  - R-type
  - I-type
  - S-type
  - B-type
  - U-type
  - J-type
- the first scalar binary layer reuses the corresponding RV32I opcode, `funct3`, and
  `funct7` placements for the integer subset
- Penguin-specific scalar mnemonics remain the software-visible assembly contract, but
  the underlying binary layout is RV32I-compatible
- `sld` reuses the standard `lw` binary encoding shape
- `sst` reuses the standard `sw` binary encoding shape
- `sfence` reuses the standard `fence` encoding shape
- `secall` and `sebreak` reuse the standard system-encoding forms corresponding to
  `ecall` and `ebreak`

This preserves a conservative scalar binary contract while still allowing Penguin to
define architecture-visible semantic differences such as:

- VMEM-only scalar loads and stores
- 2 architecturally visible delay slots on branches and jumps
- halt-on-error behavior rather than a general trap-and-restart model

The architecture also reserves the standard RISC-V custom major opcodes for future
Penguin accelerator instructions rather than spending them on scalar bring-up:

- `custom-0` (`0001011`)
- `custom-1` (`0101011`)
- `custom-2` (`1011011`)
- `custom-3` (`1111011`)

Those opcode families are intended for future DMA, tensor-memory, MXU, VPU, XLU, and
other accelerator-specific instruction forms.

### 4.2 Tensor register file

Penguin shall expose 64 architectural tensor registers.

Architecture-visible requirements:

- tensor registers are named `m0` through `m63`
- each tensor register stores one two-dimensional tile
- each tile has 64 rows
- each row stores exactly 32 bytes
- all tensor data types share the same flat architectural register file
- the same tensor register may be viewed using different element types as defined by the
  instruction semantics
- tensor registers are the main source operands for MXU, VPU, and XLU instructions
- tensor registers are read in row order by MXU, VPU, and XLU execution
- every functional unit may access every tensor register through a shared interconnect
- tensor register contents persist until overwritten by an architectural instruction

The architecture freezes storage by bytes rather than by element count. Therefore:

- an FP8 tile stores `64 x 32` elements
- a BF16 tile stores `64 x 16` elements
- narrower element types pack more elements into the same 32-byte row
- wider element types pack fewer elements into the same 32-byte row

Each tensor register therefore stores 2048 bytes of architectural tile state.

The 32-byte row is also the fundamental row quantum for tensor-side execution:

- FP8 view: 32 elements per row
- BF16 view: 16 elements per row
- all whole-register tensor instructions are defined as deterministic sweeps over 64 such
  rows

### 4.2.1 MXU weight-source operand class

Each MXU also consumes a weight operand that is distinct from the `m0..m63` tensor
register file.

Architecture-visible requirements:

- MXU weight operands are named `w0` and `w1`
- `w<src>` selects one of two weight-buffer entries resident in the MXU/PE array
- `mxu0.w0/w1` and `mxu1.w0/w1` are distinct storage
- the weight operand provides matrix operand `B`
- the tensor operand `m<src>` provides activation operand `A`
- weight-buffer contents are populated through explicit MXU push instructions
- each weight slot stores one FP8 tile of shape `32 x 16`, or 512 bytes

This operand class is intentionally narrow. It is not a second general-purpose tensor
register file. It is an architecture-visible selector for one of two MXU-resident weight
slots used by weight-stationary execution.

Initial intended weight-load form:

- `mxu.push.mxu0 w0`
- `mxu.push.mxu0 w1`
- `mxu.push.mxu1 w0`
- `mxu.push.mxu1 w1`

Weight push instructions pull weight data from memory rather than from `m` tensor
registers. In the current memory organization, that source memory is VMEM, not DRAM.
Weight data must first be staged from DRAM into VMEM by DMA. The exact VMEM-addressing
form still needs to be defined by the tensor/memory instruction set, but weight
residency itself is architecturally explicit.

Weight push behavior:

- `mxu.push.mxu0` is an on-chip transfer from VMEM into an MXU-local weight slot
- completion is blocking rather than asynchronously fenced
- execution may stall if the push cannot complete on time

### 4.3 Matrix execution units

Penguin exposes two architecturally named matrix execution units:

- `mxu0`: systolic-array-based
- `mxu1`: inner-product-tree-based

Both MXUs execute long-chime matrix arithmetic over tensor-register tiles.

Architecture-visible requirements:

- MXU instructions launch multi-cycle operations rather than completing in one cycle
- MXU operations consume tile rows in a deterministic sequence
- MXU is architected as pure matrix multiply/accumulate rather than fused tensor
  postprocessing such as bias add
- MXU instructions operate on whole tensor registers as operands and destinations
- sub-tile MXU operand windows are not part of the current architectural contract
- MXU operations use `FP8_e4m3 x FP8_e4m3 -> BF16` multiply-accumulate arithmetic
- MXU accumulation is architecturally BF16
- MXU accumulation across repeated K-sweeps is architecturally explicit
- the architecture distinguishes a fresh matmul launch from a matmul-accumulate launch
- MXU may optionally quantize results down to FP8 as part of a writeback mode defined by
  the instruction set
- BF16-to-FP8 writeback uses round-to-nearest-even
- BF16-to-FP8 writeback saturates on overflow
- each workload-level matmul includes a scalar output scaling factor
- output scaling is applied on the result path rather than on the stored input tensors
- MXU results are written back to architectural destinations defined by the instruction

The internal accumulation organization is not architecture-visible.

Architecturally, the MXU may accumulate matmul partial sums, but post-matmul fusion such
as bias add, residual add, or activation should be expressed using separate instructions
rather than a fused MXU instruction form.

Initial intended instruction shape:

- `matmul.mxu0 m<dest>, m<src>, w<src>`
- `matmul.add.mxu0 m<dest>, m<src>, w<src>, m<partial>`
- `matmul.mxu1 m<dest>, m<src>, w<src>`
- `matmul.add.mxu1 m<dest>, m<src>, w<src>, m<partial>`

Intent:

- `matmul.mxu0 m<dest>, m<src>, w<src>` computes `C = A @ B`
- `matmul.add.mxu0 m<dest>, m<src>, w<src>, m<partial>` computes
  `C = (A @ B) + partial`
- `matmul.mxu1 m<dest>, m<src>, w<src>` computes `C = A @ B`
- `matmul.add.mxu1 m<dest>, m<src>, w<src>, m<partial>` computes
  `C = (A @ B) + partial`
- `m<src>` denotes the source activation tile `A`
- `w<src>` denotes which of the two selected-engine weight entries supplies `B`
- `m<partial>` denotes the previous partial-sum tile for explicit accumulation
- `m<dest>` receives the result tile `C`

Exact operand roles still need a full tensor-ISA document, but accumulation is explicit
in the opcode rather than implicit in destination state.

### 4.4 Vector processing unit

The VPU executes long-chime row-wise operations over tensor-register tiles.

Architecture-visible requirements:

- VPU instructions launch multi-cycle operations
- VPU reads tensor tiles row by row
- VPU sources operands directly from `m0..m63`
- VPU writes results only to `m0..m63`
- VPU instructions operate on whole tensor registers as operands and destinations
- sub-tile VPU operand windows are not part of the current architectural contract
- VPU latency is deterministic for a fixed instruction form and configuration
- initial pipelineable VPU elementwise operations use a 2-cycle latency class in the
  performance model
- future non-pipelineable VPU operations such as division use an 8-cycle latency class
- VPU operations may be scheduled in parallel with scalar instructions when architectural
  hazards permit
- VPU is intended to grow toward a broad tensor-function set, but the first architected
  VPU scope should be elementwise operations

Initial intended VPU opcode floor:

- `vadd`
- `vmul`
- `vmax`
- `vmin`
- `vrelu`
- `vmov`

The VPU does not currently expose a separate local operand-buffer abstraction comparable
to the MXU weight slots.
The VPU also does not directly write to memory in the current architecture direction.

The exact supported vector opcode set beyond the initial elementwise subset is not frozen
in this revision.

### 4.5 Cross-lane transpose unit

The XLU executes long-chime transpose operations over tensor-register tiles.

Architecture-visible requirements:

- XLU instructions launch multi-cycle operations
- XLU reads source data from `m0..m63`
- XLU writes results only to `m0..m63`
- XLU instructions operate on whole tensor registers as operands and destinations
- XLU does not directly access VMEM or DRAM
- XLU latency is deterministic for a fixed instruction form and configuration
- XLU exists to implement transpose operations that are not a natural fit for MXU or VPU

Baseline design direction:

- the XLU may reuse the shift-based transpose structure described in the upstream
  `npu_model` reference design
- the architecture only commits to transpose support, not a broad family of rotate,
  reduction, or pack-unpack operations

The exact XLU instruction encoding and detailed multi-instruction transpose sequencing are
not frozen in this revision.

## 5. Execution Model

### 5.1 Single-instruction frontend

Penguin is not a VLIW architecture. The frontend fetches and issues one instruction at a
time.

Architectural rationale:

- reduces instruction footprint
- simplifies the frontend
- relies on long-chime operations to amortize instruction bandwidth
- only one instruction may be launched per cycle
- instructions are fixed-width 32-bit words
- all architected instruction classes must fit in one 32-bit word
- operations needing more parameters must use register-indirect operands or decompose into
  multiple instructions
- memory-facing tensor and DMA instructions use scalar-register indirection for addresses
- one shared CSR-provided base offset extends addressing beyond the 32-bit scalar
  register width

### 5.2 Statically scheduled execution

Programs are expected to be statically scheduled by the compiler or assembler flow.

Architectural consequences:

- arithmetic unit latencies are deterministic
- the compiler is expected to arrange instructions to hide latency
- the hardware may stall issue when the targeted functional unit is busy
- such stalls preserve correctness but should be uncommon in optimized programs
- `mxu0` and `mxu1` cannot both be launched in the same cycle because issue is single
  width
- `mxu0` and `mxu1` may still execute concurrently after being launched on different
  cycles
- off-chip memory operations are not implicitly retired into architectural visibility
- software must use explicit synchronization before consuming results of asynchronous
  off-chip memory operations

### 5.3 Long-chime instructions

Long-chime instructions are a first-class architectural concept.

A long-chime instruction:

- launches a multi-cycle operation
- sweeps through one or more tensor-register tiles
- occupies its target functional unit for a deterministic duration
- does not require one instruction per row or one instruction per element

The purpose of long-chime execution is to:

- lower instruction throughput demand
- allow scalar work to overlap with long-latency tensor work
- make compile-time scheduling worthwhile

### 5.4 Control-transfer timing

Branches and jumps have exactly 2 architecturally visible delay slots in the
current functional architecture contract.

Current rule:

- the functional model must honor 2 delay slots for branches and jumps
- software may schedule useful work into those 2 delay slots
- this 2-delay-slot rule is part of the committed baseline architecture contract

### 5.5 Structural conflicts and completion atomicity

The architecture requires all-or-nothing visibility at the granularity of an issued
instruction.

Current rule:

- if a structural conflict prevents forward progress, the machine stalls rather than
  exposing a partially completed architectural writeback
- tensor instructions do not retire partially visible rows because of crossbar, banking,
  or destination-port conflicts
- if an instruction is architecturally visible as complete, then its full defined
  destination effect is visible

This rule applies even if the underlying implementation uses banked SRAMs, shared buses,
or other bandwidth-limited structures.

### 5.6 Reset and initialization semantics

The baseline architecture now defines the following reset expectations:

- `pc` resets to the IMEM base address
- execution-control state resets to disabled
- DMA channels reset to idle
- any pending control-transfer redirection state resets to empty
- `x0` remains zero by definition
- DRAM and VMEM contents are architecturally unspecified after reset unless software or
  the host environment initializes them
- other scalar registers, tensor registers, and MXU weight slots are architecturally
  unspecified after reset unless software or the host environment initializes them

Software must not rely on reset-time contents of architectural data state other than
values explicitly defined above.

In the Python functional/performance model, the default instantiation of these
architecturally unspecified contents is deterministic pseudo-random data controlled by
`PenguinCoreConfig.initialization`.

### 5.7 Error handling model

Penguin does not support general trap or interrupt recovery in the baseline design.

Current rule:

- if execution encounters an architecturally defined error condition, the accelerator
  halts
- the halt reason is reported through host-visible status state
- the machine does not attempt recovery, trap handling, or restart from an exception
  handler
- environment instructions such as `secall` and `sebreak` are modeled as terminal halt
  conditions rather than recoverable exceptions

## 6. Determinism Requirements

For a fixed hardware configuration, the following shall be deterministic:

- scalar instruction latency
- MXU instruction latency
- VPU instruction latency
- XLU instruction latency
- architectural read and write ordering at instruction granularity

Off-chip memory operations are explicitly excluded from single-latency determinism. Their
completion is governed by asynchronous memory behavior plus explicit synchronization.

This determinism is required so that compile-time scheduling remains meaningful.

## 6.1 Tensor layout and padding responsibility

The architecture assumes that software provides tensor data in the tiled layout expected
by the issued program.

Baseline rule:

- the compiler or host-side packer is responsible for arranging tensor blobs in the
  correct tiled layout
- if a logical tensor shape does not fill a complete architectural tile, software must
  apply zero padding at tile granularity
- hardware does not synthesize masked tails automatically in the baseline design

The OpenXLA tiled-layout material is a useful non-normative reference for this software
packing problem, but Penguin will still need its own formal tensor-layout specification.

## 6.2 Floating-point corner semantics

The baseline tensor numeric model is intentionally narrow.

Current rule:

- subnormal inputs are treated as zero
- operations do not intentionally produce IEEE infinities; values that overflow the
  supported destination range clamp to the largest-magnitude representable value
- BF16-to-FP8 conversion uses round-to-nearest
- NaNs are not expected in valid software-generated workloads
- if NaNs are encountered anyway, they propagate through the computation

## 7. Architectural Differences From Google TPU

Penguin deliberately diverges from classic TPU execution in one major way:

- Penguin uses a single-instruction fetch frontend rather than issuing multiple
  instructions per cycle through a VLIW bundle

Penguin intentionally preserves these TPU-like characteristics:

- tile-oriented execution
- explicit matrix and vector units
- a dedicated transpose unit
- long-lived tensor operands in a dedicated register file
- compiler-visible deterministic execution timing

## 8. Memory-System Direction

The memory-system direction is now defined at a high level:

- DRAM is off-chip main storage
- DRAM base is `0x8000_0000` and DRAM size is `16 GiB`
- IMEM stores program instructions
- IMEM base is `0x0010_0000` and IMEM size is `32 KiB`
- IMEM is byte-addressed
- VMEM stores on-chip tensor/vector data
- VMEM base is `0x0800_0000` and VMEM size is `1 MiB`
- VMEM is byte-addressed
- DMA moves data between DRAM and VMEM
- DMA transports unit-stride raw bytes rather than typed tensor objects
- tensor registers access VMEM only
- `vload` and `vstore` connect tensor registers to VMEM
- `vload` and `vstore` operate on whole tensor registers rather than sub-tile windows
- DMA is asynchronous and requires explicit synchronization
- DMA synchronization is channel-specific rather than global
- the first architecture revision exposes 8 DMA channels
- the 8 DMA channels are symmetric in capability
- on-chip VMEM-facing transfers are blocking rather than asynchronously fenced
- address indirection is performed through scalar registers
- one shared CSR base-offset mechanism exists for accessing higher memory regions beyond
  the raw 32-bit scalar address space

The intended memory-model split is:

- on-chip execution and on-chip dataflow are cycle-deterministic
- off-chip transfers are asynchronous
- explicit synchronization is required before software may assume off-chip transfer
  completion
- DMA waits apply to specific channels such as `dma.wait.ch1`

VMEM-facing tensor operations require 32-byte alignment in the current baseline
direction. IMEM fetch requires 4-byte alignment.

## 9. Remaining Open Architectural Items

The following items remain open in this revision:

- exact VPU opcode set
- exact XLU transpose instruction encoding
- exact matrix instruction set
- executable-package format for program plus tensor data
- exact host-side CSR-region base address and launch MMIO protocol

## 10. Required Companion Specifications

This document depends on future companion documents for:

- full architecture specification of tensor instructions
- microarchitecture specification
- memory-map and memory-structure specification
- shared configuration-parameter specification
- executable-package specification
- tensor layout and packing specification
