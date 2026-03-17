# Penguin Microarchitecture Specification

Status: Baseline 1.0

## 1. Scope

This document defines the baseline implementation direction for Penguin-TPU.

It is the authoritative specification for:

- the intended block-level organization
- pipeline and issue behavior
- functional-unit timing classes
- shared model and RTL parameters
- memory-system implementation direction
- arbitration and overlap rules
- reset / initialization strategy used by the software model
- trace and observability requirements

This document refines the architecture specification. If this document conflicts with the
architecture specification on an architecture-visible behavior, the architecture
specification takes precedence.

## 2. Design Objectives

The baseline microarchitecture shall optimize for:

- a narrow, explainable single-issue frontend
- deterministic on-chip timing for a fixed configuration
- explicit overlap of long-chime execution units
- a single asynchronous boundary at `DRAM <-> VMEM`
- straightforward software-model, RTL, and trace alignment

The baseline does not attempt to optimize for:

- superscalar issue
- out-of-order execution
- speculative memory disambiguation
- hidden hardware-managed tensor caches
- generalized trap recovery

## 3. Frozen Shared Parameters

The following parameters are frozen for the current baseline implementation.

### 3.1 Architectural shape

| Parameter | Value | Meaning |
|---|---:|---|
| `INSN_WIDTH` | `32` bits | Fixed instruction width |
| `INSN_ALIGN` | `4` bytes | Instruction alignment |
| `NUM_XREG` | `32` | Scalar register count |
| `CONTROL_FLOW_DELAY_SLOTS` | `2` | Required branch/jump delay slots |
| `NUM_EREG` | `32` | Scale register count |
| `EREG_BITS` | `8` | Bits per `e` register (`FP8_E8M0`) |
| `NUM_MREG` | `64` | Tensor register count |
| `MREG_ROWS` | `64` | Rows per tensor register |
| `MREG_ROW_BYTES` | `64` bytes | Bytes per tensor-register row |
| `MREG_BYTES` | `4096` bytes | Bytes per tensor register |
| `MXU_COUNT` | `2` | Architected MXU count |
| `WEIGHT_SLOTS_PER_MXU` | `2` | Weight slots per MXU |
| `WEIGHT_TILE_ROWS` | `64` | Rows per MXU weight tile |
| `WEIGHT_TILE_COLS_FP8` | `64` | FP8 columns per MXU weight tile |
| `WEIGHT_SLOT_BYTES` | `4096` bytes | Bytes per MXU weight slot |
| `MXU_ARRAY_ROWS` | `64` | Rows per MXU array |
| `MXU_ARRAY_COLS` | `64` | Columns per MXU array |
| `VPU_LANES_BF16` | `32` | BF16 lanes in the baseline VPU |
| `DMA_CHANNELS` | `8` | Architected DMA channels |
| `DMA_ALIGN` | `32` bytes | DMA alignment and granularity |
| `IMEM_BASE` | `0x0010_0000` | IMEM base address |
| `IMEM_SIZE` | `32 KiB` | IMEM capacity |
| `VMEM_BASE` | `0x0800_0000` | VMEM base address |
| `VMEM_SIZE` | `1 MiB` | VMEM capacity |
| `DRAM_BASE` | `0x8000_0000` | DRAM base address |
| `DRAM_SIZE` | `16 GiB` | DRAM capacity |

### 3.2 Timing classes and bandwidth fragments

| Parameter | Value | Meaning |
|---|---:|---|
| `MATMUL_LATENCY_CYCLES` | `64` | One MXU matmul launch latency class |
| `VPU_SIMPLE_OP_LATENCY_CYCLES` | `2` | Pipelineable VPU latency class |
| `VPU_NON_PIPELINEABLE_OP_LATENCY_CYCLES` | `8` | Non-pipelineable VPU latency class |
| `XLU_TRANSPOSE_LATENCY_CYCLES` | `4` | Whole-register transpose latency class |
| `OFFCHIP_LINK_WIDTH_BITS` | `32` | DRAM-link beat width |
| `OFFCHIP_LINK_CORE_CYCLES_PER_BEAT` | `2` | Off-chip serialized beat time |
| `DMA_OFFCHIP_COMMAND_WORDS` | `2` | DRAM-side DMA command overhead |
| `VMEM_BUS_WIDTH_BITS` | `128` | VMEM/system-bus beat width |
| `VMEM_BUS_CORE_CYCLES_PER_BEAT` | `1` | VMEM/system-bus beat time |
| `VMEM_TENSOR_ALIGN` | `32` bytes | VMEM alignment for `vload`, `vstore`, `mxu.push.*` |
| `TRACE_TICKS_PER_CYCLE` | `3` | Trace timestamp granularity |

### 3.3 Software-model initialization parameters

| Parameter | Value | Meaning |
|---|---:|---|
| `INIT_SEED` | `0x50E11234` | Deterministic pseudo-random initialization seed |
| `RANDOMIZE_DRAM` | `true` | Randomize DRAM at power-on in the Python model |
| `RANDOMIZE_VMEM` | `true` | Randomize VMEM at power-on in the Python model |
| `RANDOMIZE_SCALAR_REGISTERS` | `true` | Randomize scalar registers except `x0` |
| `RANDOMIZE_TENSOR_REGISTERS` | `true` | Randomize tensor registers |
| `RANDOMIZE_WEIGHT_SLOTS` | `true` | Randomize MXU weight-slot state |

## 4. Top-Level Organization

The baseline Penguin microarchitecture is organized around the following major
subsystems:

- an instruction frontend
- a scalar execution path
- a scale register file
- a tensor register file and tensor interconnect
- two MXUs
- one VPU
- one XLU
- an instruction memory path
- a VMEM subsystem
- a DMA engine complex connecting `DRAM` and `VMEM`
- a host-visible control and status block

### 4.1 Frontend

The frontend baseline shall be:

- single-stream instruction fetch
- single instruction decode
- single issue decision per cycle
- fixed-width 32-bit fetch from `IMEM`

The frontend phase model used by the performance model and trace infrastructure is:

- `IFG`: instruction-address generation and request launch
- `IFR`: instruction return
- `IDU`: decode and dispatch
- `EXU`: execution-unit launch or scalar execution

### 4.2 Execution-unit overlap model

The baseline implementation shall support concurrent long-chime unit activity.

Requirements:

- `mxu0`, `mxu1`, `vpu`, `xlu`, and DMA transfers may be active concurrently
- only one new instruction may issue in a cycle
- issue stalls when the targeted unit cannot accept a new instruction
- issue does not perform dynamic reordering to bypass stalled older instructions

### 4.3 Tensor access organization

The baseline tensor-access model shall be fully connected at the architectural boundary:

- every functional unit may access every tensor register
- internal arbitration may stall or serialize traffic
- internal banking shall not be exposed as a software-visible rule

The baseline implementation direction is a central tensor crossbar or equivalent shared
interconnect.

## 5. Scalar Frontend and Scalar Execution Path

### 5.1 Scalar decode

The scalar decode path shall recognize the RV32I-compatible binary baseline defined by
the architecture specification.

Minimum decode outputs:

- instruction class
- `rd`
- `rs1`
- `rs2`
- sign-extended immediate
- target execution unit
- legality classification

The decode stage should classify standard RISC-V custom major opcodes distinctly from
fully illegal encodings to preserve future accelerator-extension space.

The accelerator decode path shall recognize the baseline custom-opcode allocation defined
by the architecture specification:

- `custom-0` for scalar-side auxiliary loads, DMA, and VMEM-facing tensor transfers
- `custom-1` for MXU launch instructions
- `custom-2` for VPU whole-register instructions
- `custom-3` for XLU whole-register instructions

Minimum accelerator decode outputs:

- accelerator family
- reconstructed `6`-bit tensor-register operands where present
- reconstructed `5`-bit scale-register operands where present
- reconstructed weight-slot selector where present
- reconstructed DMA channel selector where present
- reconstructed scaled VMEM immediate where present
- legality classification for reserved-bit and reserved-subopcode violations

The baseline decode implementation shall treat any nonzero reserved field in the
accelerator encodings as illegal rather than silently ignored.

### 5.2 Scalar execution blocks

The intended first scalar implementation slice is partitioned into:

- scalar decoder
- scalar register file
- scale-register load path for `seli` and `seld`
- scalar ALU / compare datapath
- branch and jump target unit
- VMEM-facing scalar load/store unit
- scalar control block that owns `pc`, delay-slot tracking, and halt status

### 5.3 Delay-slot handling

The microarchitecture shall preserve the architectural two-delay-slot rule without
speculation.

The baseline implementation direction is:

- the control block records a pending redirect and the remaining delay-slot count
- sequential fetch continues through the two delay-slot instructions
- once the delay-slot count reaches zero, the redirect is applied
- a younger branch or jump in a delay slot overwrites the older pending redirect

## 6. Scale Register File, Tensor Register File, and Local Interconnect

### 6.1 Scale register file

The scale register file shall hold:

- `32` registers
- `8` bits per register
- one whole-tensor `FP8_E8M0` scale per register

Baseline implementation direction:

- `seli` writes `e` registers through a scalar-side immediate path
- `seld` writes `e` registers through a scalar-controlled VMEM byte-load path
- MXUs read `e` registers as side metadata for scaled matmul launch

Rationale:

- scale values are too small and too structured to justify burning full `m` registers
- scale values are not ordinary scalar integers and should not consume general `x`
  register bandwidth and lifetime

### 6.2 Tensor register file

The tensor register file shall hold:

- `64` registers
- `4096` bytes per register
- whole-register read and write access for tensor instructions

The baseline whole-register interpretations are:

- `64 x 32` BF16 for VPU, XLU, and MXU half-result traffic
- `64 x 64` FP8 source tiles across the full physical row width for MXU input traffic

The baseline MXU result contract writes one `64 x 64` BF16 tile into two consecutive
tensor registers:

- destination base register carries BF16 columns `[31:0]`
- destination base plus one carries BF16 columns `[63:32]`
- accumulate-form partial sums follow that same paired-register convention

The baseline does not require architectural bank visibility.

### 6.3 Weight slots

Each MXU shall contain two distinct weight-slot storage entries.

The baseline implementation direction is weight-stationary:

- weight data is pushed into the selected slot from `VMEM`
- the slot remains resident until overwritten
- matmul launch selects between `w0` and `w1`
- each weight slot stores one `64 x 64` FP8 tile

### 6.4 Structural-conflict handling

Structural conflicts shall be handled by stalls or arbitration.

They shall not create:

- partial architectural row retirement
- partial architectural tile retirement
- architecturally visible younger-over-older preemption

## 7. Memory-System Implementation Direction

### 7.1 High-level rule

The baseline memory system shall keep the asynchronous boundary narrow:

- `IMEM` fetch is local and deterministic
- `VMEM` is the sole on-chip tensor staging memory
- `DRAM` access is off-chip and asynchronous
- DMA is the only `DRAM <-> VMEM` path

### 7.2 Blocking on-chip tensor transfers

The following instructions are modeled and implemented as blocking:

- `vload`
- `vstore`
- `mxu.push.*`
- `seld`
- scalar `sld`
- scalar `sst`

The baseline intent is to keep on-chip movement deterministic and simpler to verify.

### 7.3 Asynchronous DMA

DMA shall be channelized and asynchronous.

Each channel supports:

- at most one outstanding transfer
- independent busy/completion state
- `dma.wait.chN` synchronization

The microarchitecture is free to implement the DMA channels with shared internal data
paths or arbitration, provided the architecture-visible channel behavior is preserved.

### 7.4 Baseline transfer formulas

The performance model and timing expectations shall use the following formulas.

Definitions:

- `OFFCHIP_BYTES_PER_BEAT = OFFCHIP_LINK_WIDTH_BITS / 8 = 4`
- `VMEM_BYTES_PER_BEAT = VMEM_BUS_WIDTH_BITS / 8 = 16`

Required formulas:

- `dma_offchip_cycles(bytes) = ceil((bytes + 4 * DMA_OFFCHIP_COMMAND_WORDS) / OFFCHIP_BYTES_PER_BEAT) * OFFCHIP_LINK_CORE_CYCLES_PER_BEAT`
- `vmem_transfer_cycles(bytes) = ceil(bytes / VMEM_BYTES_PER_BEAT) * VMEM_BUS_CORE_CYCLES_PER_BEAT`
- `dma_transfer_cycles(bytes) = max(dma_offchip_cycles(bytes), vmem_transfer_cycles(bytes))`

For the frozen baseline values:

- one off-chip beat costs `2` core cycles
- one VMEM beat costs `1` core cycle
- one `vload` / `vstore` of a `4096`-byte tensor register takes `256` cycles
- one `mxu.push.*` of a `4096`-byte weight tile takes `256` cycles

## 8. Functional Units

### 8.1 MXU0 and MXU1

The two MXUs share the same architectural interface but intentionally differ internally.

Baseline intent:

- `mxu0`: systolic-array accumulation
- `mxu1`: inner-product-tree accumulation

Shared microarchitectural requirements:

- deterministic latency class of `64` cycles per matmul launch
- whole-register activation source
- selected weight-slot source
- selected activation-scale and weight-scale source
- BF16 architectural accumulation
- square `64 x 64` array geometry for both `mxu0` and `mxu1`
- paired-register BF16 writeback for each full MXU result tile
- ability to overlap with scalar and other long-chime units, subject to issue policy

### 8.2 VPU

The VPU baseline shall implement the initial BF16 elementwise floor:

- `vadd`
- `vsub`
- `vmul`
- `vmax`
- `vmin`
- `vrelu`
- `vmov`
- `vexp`
- `vrecip`

Timing requirements:

- pipelineable elementwise operations use the `2`-cycle latency class
- non-pipelineable elementwise operations such as exponent and reciprocal use the
  `8`-cycle latency class
- the baseline lane count is `32` BF16 lanes

### 8.3 XLU

The XLU baseline shall implement whole-register transpose and row-reduction broadcasts
over the baseline `64 x 32` BF16 view.

Timing requirement:

- `transpose.xlu` uses the `4`-cycle latency class
- `reduce.max.xlu` uses the `4`-cycle latency class
- `reduce.sum.xlu` uses the `4`-cycle latency class

## 9. Reset, Initialization, and Model Contract

### 9.1 Reset behavior

The hardware shall reset control state, but the architecture does not require zeroed
data memories or registers.

The reference model shall instantiate the unspecified state deterministically using the
frozen initialization seed and randomization controls.

### 9.2 `PenguinCoreConfig`

The Python functional / performance model shall expose one top-level configuration object
that binds the frozen parameter set to a concrete machine instance.

Required logical fragments:

- scalar
- scale
- memory map
- memory backend
- initialization
- DMA
- tensor
- VPU
- XLU
- bandwidth
- trace

The active runtime behavior of a model instance shall flow from the bound configuration,
not from unrelated global constants.

## 10. Trace and Observability

The baseline model and trace infrastructure shall distinguish:

- instruction fetch / frontend occupancy
- decode / dispatch occupancy
- scalar execution
- DMA execution
- memory-region-visible traffic

The current trace timestamp granularity is `3` ticks per core cycle.

The baseline trace lane split is:

- `IFU`
- `IDU`
- `EXU.SALU`
- `EXU.DMA`
- additional tensor-unit lanes as needed by the model

## 11. Performance-Model Interpretation

The current roofline and throughput model shall interpret the frozen parameters in
normalized `ops/cycle` and `bytes/cycle` units unless an explicit clock frequency is
provided externally.

Required derived roofs:

- DRAM roof from `2 B/cycle`
- VMEM roof from `16 B/cycle`
- total MXU peak from two `64`-cycle matmul engines

The performance model may scale those normalized values linearly into `ops/s` and
`bytes/s` when a frequency target is supplied, but clock frequency is not frozen by this
specification.

## 12. Implementation Freedom

The following remain implementation choices provided the architecture-visible behavior is
preserved:

- exact RTL partitioning of frontend phases
- bank and port structure inside the tensor register file
- specific arbitration policy for internal tensor and VMEM paths
- detailed systolic-array / reduction-tree internal datapath organization
- detailed DMA datapath sharing between channels
- physical placement, clock gating, and local buffering strategy

The following are not implementation freedoms in this baseline:

- instruction width and scalar binary compatibility
- two control-flow delay slots
- the three-region memory map
- DMA channelization and fence-by-channel behavior
- the frozen timing-class and bandwidth parameters above
