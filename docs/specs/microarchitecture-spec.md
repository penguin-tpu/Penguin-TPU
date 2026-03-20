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
- out-of-order frontend issue
- speculative memory disambiguation
- hidden hardware-managed tensor caches
- generalized trap recovery

## 3. Frozen Shared Parameters

### 3.1 Architectural shape

| Parameter | Value | Meaning |
|---|---:|---|
| `INSN_WIDTH` | `32` bits | Fixed instruction width |
| `INSN_ALIGN` | `4` bytes | Instruction alignment |
| `NUM_XREG` | `32` | Scalar register count |
| `CONTROL_FLOW_DELAY_SLOTS` | `2` | Required branch / jump delay slots |
| `NUM_EREG` | `32` | Scale register count |
| `EREG_BITS` | `8` | Bits per `e` register |
| `NUM_MREG` | `64` | Tensor register count |
| `MREG_ROWS` | `64` | Rows per tensor register |
| `MREG_ROW_BYTES` | `64` bytes | Bytes per tensor-register row |
| `MREG_BYTES` | `4096` bytes | Bytes per tensor register |
| `MXU_COUNT` | `2` | Architected MXU count |
| `WEIGHT_SLOTS_PER_MXU` | `2` | Weight slots per MXU |
| `WEIGHT_SLOT_BYTES` | `4096` bytes | Bytes per MXU weight slot |
| `ACCUM_BUFFER_ROWS` | `64` | Rows per MXU accumulation buffer |
| `ACCUM_BUFFER_COLS_BF16` | `64` | BF16 columns per MXU accumulation buffer |
| `ACCUM_BUFFER_BYTES` | `8192` bytes | Bytes per MXU accumulation buffer |
| `MXU_ARRAY_ROWS` | `64` | Rows per MXU array |
| `MXU_ARRAY_COLS` | `64` | Columns per MXU array |
| `VPU_LANES_BF16` | `32` | BF16 lanes in the baseline VPU |
| `DMA_CHANNELS` | `8` | Architected DMA channels |
| `DMA_ALIGN` | `32` bytes | DMA alignment and granularity |
| `IMEM_BASE` | `0x0002_0000` | IMEM base address |
| `IMEM_SIZE` | `64 KiB` | IMEM capacity |
| `VMEM_BASE` | `0x2000_0000` | VMEM base address |
| `VMEM_SIZE` | `1 MiB` | VMEM capacity |
| `DRAM_BASE` | `0x8000_0000` | DRAM base address |
| `DRAM_SIZE` | `16 GiB` | DRAM capacity |

### 3.2 Timing classes and bandwidth fragments

| Parameter | Value | Meaning |
|---|---:|---|
| `MATMUL_LATENCY_CYCLES` | `64` | One MXU matmul launch latency class |
| `VPU_SIMPLE_OP_LATENCY_CYCLES` | `2` | Pipelineable VPU latency class |
| `VPU_NON_PIPELINEABLE_OP_LATENCY_CYCLES` | `8` | Non-pipelineable VPU latency class |
| `XLU_TRANSFORM_LATENCY_CYCLES` | `4` | Whole-register XLU latency class |
| `OFFCHIP_LINK_WIDTH_BITS` | `32` | DRAM-link beat width |
| `OFFCHIP_LINK_CORE_CYCLES_PER_BEAT` | `2` | Off-chip serialized beat time |
| `DMA_OFFCHIP_COMMAND_WORDS` | `2` | DRAM-side DMA command overhead |
| `VMEM_BUS_WIDTH_BITS` | `512` | VMEM / system-bus beat width |
| `VMEM_BUS_CORE_CYCLES_PER_BEAT` | `1` | VMEM / system-bus beat time |
| `VMEM_TENSOR_ALIGN` | `32` bytes | VMEM alignment for `vload`, `vstore`, and `vload.weight.*` |
| `TRACE_TICKS_PER_CYCLE` | `1` | Trace timestamp granularity |

### 3.3 Software-model initialization parameters

| Parameter | Value | Meaning |
|---|---:|---|
| `INIT_SEED` | `0x50E11234` | Deterministic pseudo-random initialization seed |
| `RANDOMIZE_DRAM` | `true` | Randomize DRAM at power-on in the Python model |
| `RANDOMIZE_VMEM` | `true` | Randomize VMEM at power-on in the Python model |
| `RANDOMIZE_SCALAR_REGISTERS` | `true` | Randomize scalar registers except `x0` |
| `RANDOMIZE_SCALE_REGISTERS` | `true` | Randomize scale registers |
| `RANDOMIZE_TENSOR_REGISTERS` | `true` | Randomize tensor registers |
| `RANDOMIZE_WEIGHT_SLOTS` | `true` | Randomize MXU weight-slot state |
| `RANDOMIZE_ACCUM_BUFFERS` | `true` | Randomize MXU accumulation-buffer state |
| `RANDOMIZE_DMA_BASE` | `true` | Randomize `dma.base` at power-on in the Python model |

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
- an instruction-memory path
- a VMEM subsystem
- a DMA engine complex connecting `DRAM` and `VMEM`
- a host-visible control and status block

### 4.1 Frontend

The frontend baseline shall be:

- single-stream instruction fetch
- single instruction decode
- single issue decision per cycle
- fixed-width `32`-bit fetch from `IMEM`

The frontend phase model used by the performance model and trace infrastructure is:

- `IFU`: fetched-instruction stage
- `IDU`: decode and issue stage
- `EXU`: execution-unit stage

### 4.2 Execution-unit overlap model

The baseline implementation shall support concurrent long-chime unit activity.

Requirements:

- `mxu0`, `mxu1`, `vpu`, `xlu`, and DMA transfers may be active concurrently
- only one new instruction may issue in a cycle
- issue stalls when the targeted unit cannot accept a new instruction
- issue does not perform dynamic reordering to bypass stalled older instructions
- the baseline issue model does not perform architectural register dependency checks or
  register-ready scoreboarding
- execution timing is instead determined by frontend ordering, unit availability,
  architecturally defined blocking instructions, and fixed instruction latency classes

The intended performance-model interpretation is:

- scalar fetch and decode advance one cycle at a time
- an older long-chime instruction may remain busy in its target unit after it leaves
  `IDU`
- while that unit remains busy, younger instructions may still be fetched, decoded,
  issued, and completed on other available units
- a decode-resident fence such as `delay` or `dma.wait.chN` blocks younger issue until it
  retires

### 4.3 Tensor access organization

The baseline tensor-access model shall be fully connected at the architectural boundary:

- every functional unit may access every tensor register
- internal arbitration may stall or serialize traffic
- internal banking shall not be exposed as a software-visible rule

The baseline implementation direction is a central tensor crossbar or equivalent shared
interconnect.

## 5. Frontend Decode and Control

### 5.1 Decode responsibilities

The decode path shall recognize:

- scalar `R`, `I`, `S`, `SB`, `U`, and `UJ` instructions
- Penguin `VLS`, `VR`, and `VI` instructions
- Penguin scalar `delay`
- DMA transfer and DMA control families

Minimum decode outputs:

- instruction class
- `rd`
- `rs1`
- `rs2`
- sign-extended scalar immediate where applicable
- reconstructed `vd`, `vs1`, and `vs2` where applicable
- reconstructed weight-slot selector where applicable
- reconstructed DMA channel selector where applicable
- target execution unit
- legality classification

The decode control record shall additionally expose at least:

- `valid`
- `illegal`
- `format_class`
- `scalar_op_class`
- `tensor_op_class`
- `writes_xrd`
- `writes_ereg`
- `writes_mreg`
- `reads_rs1`
- `reads_rs2`
- `reads_vs1`
- `reads_vs2`
- `is_branch`
- `is_jump`
- `is_delay`
- `is_dma_wait`
- `is_scalar_load`
- `is_scalar_store`
- `is_tensor_move`
- `is_long_chime`

### 5.2 Opcode map recognized by decode

The decode stage shall recognize the frozen architecture opcode allocation:

| Opcode bits `[6:0]` | Family |
|---|---|
| `0000011` | scalar loads, `seld`, `seli` |
| `0000111` | `VLS` tensor transfer family |
| `0001111` | `fence` |
| `0010011` | scalar immediate ALU |
| `0010111` | `auipc` |
| `0100011` | scalar stores |
| `0110011` | scalar register ALU |
| `0110111` | `lui` |
| `1010111` | `VR` VPU family |
| `1011111` | `VI` vector-immediate family |
| `1100011` | scalar branches |
| `1100111` | `jalr` and `delay` |
| `1101011` | `VR` XLU family |
| `1101111` | `jal` |
| `1110011` | `ecall` and `ebreak` |
| `1110111` | `VR` MXU family |
| `1111011` | DMA transfer family |
| `1111111` | DMA control family |

The decode stage shall classify any other major opcode as illegal in the current
baseline.

### 5.3 Family-specific decode rules

Required family-specific decode behavior:

- opcode `0x03`, `funct3=110` decodes as `seld` and writes `e[rd]`
- opcode `0x03`, `funct3=111` decodes as `seli` and writes `e[rd]`; nonzero `rs1` is
  illegal
- opcode `0x67`, `funct3=000` decodes as `jalr`
- opcode `0x67`, `funct3=001` decodes as `delay`; nonzero `rd` or `rs1` is illegal
- opcode `0x07` decodes as `VLS` and interprets `f2` as subopcode
- opcode `0x57`, `0x6B`, and `0x77` decode as `VR` and interpret `funct7` as subopcode
- opcode `0x5F` decodes as `VI` and interprets `f3` as subopcode
- opcode `0x7B` interprets `funct3` as DMA channel selector and `funct7` as load/store
  selector
- opcode `0x7F` interprets `funct3` as DMA channel selector and `imm[6:0]` as control
  subopcode

The decode implementation shall also enforce the architectural reserved-zero rules:

- `vload.weight.*`: `vd[5:1] = 0`
- `VPU` unary operations: `vs2 = 0`
- `XLU` operations: `vs2 = 0`
- `vmatpush.weight.*`: `vs2 = 0` and `vd[5:1] = 0`
- `vmatpush.acc.fp8.*`: `vs2 = 0` and `vd[5:1] = 0`
- `vmatpush.acc.bf16.*`: `vs2 = 0` and `vd[5:1] = 0`
- `vmatpop.*`: `vs1 = 0` and `vs2[5:1] = 0`
- `vmatmul.*`: `vd[5:1] = 0` and `vs2[5:1] = 0`
- `dma.config.chN` and `dma.wait.chN`: `rd = x0`
- `dma.wait.chN`: `rs1 = x0`
- DMA control `imm[11:7] = 0`

Any reserved-field violation shall decode as illegal.

### 5.4 Delay-slot handling

The microarchitecture shall preserve the architectural two-delay-slot rule without
speculation.

The baseline implementation direction is:

- the control block tracks unresolved control-flow shadows in program order
- sequential fetch continues until the youngest resolved shadow has observed its required
  two delay-slot fetches
- once the youngest resolved shadow has consumed its two delay slots, its redirect is
  applied
- a branch or jump decoded in a delay-slot position is illegal and shall terminate
  execution before any younger redirect is applied

### 5.5 Scalar implementation slice

The intended first scalar implementation slice is partitioned into:

- scalar decoder
- scalar register file
- scale-register write path for `seld` and `seli`
- scalar ALU / compare datapath
- branch and jump target unit
- VMEM-facing scalar load/store unit supporting byte, halfword, and word accesses
- scalar control block that owns `pc`, delay-slot tracking, halt status, and `dma.base`

Recommended block responsibilities:

- `penguin_scalar_defs.vh`: opcode, `funct3`, `funct7`, halt-reason, and ALU-function
  constants
- `penguin_scalar_decoder.v`: binary field extraction, immediate generation, and legality
  classification
- `penguin_scalar_regfile.v`: `32 x 32` scalar register storage with hardwired `x0`
- `penguin_scalar_eregfile.v` or equivalent: `32 x 8` scale-register storage
- `penguin_scalar_alu.v`: scalar ALU and compare datapath
- `penguin_scalar_branch_unit.v`: branch / jump target generation plus alignment checks
- `penguin_scalar_lsu.v`: blocking VMEM byte / halfword / word load-store path with
  alignment checks
- `penguin_scalar_controller.v`: `pc` sequencing, delay-slot bookkeeping, `dma.base`,
  and halt-status generation
- `penguin_scalar_core.v`: top-level scalar integration of fetch, decode, execute, and
  memory interfaces

## 6. Register Files, MXU Local State, and Interconnect

### 6.1 Scale register file

The scale register file shall hold:

- `32` registers
- `8` bits per register
- one whole-tensor scale payload per register

Baseline implementation direction:

- `seli` writes `e` registers through a scalar-side immediate path
- `seld` writes `e` registers through a scalar-controlled VMEM byte-load path
- `e` registers are read only by future or current tensor instructions that explicitly
  name them; no implicit scalar aliasing exists

### 6.2 Tensor register file

The tensor register file shall hold:

- `64` registers
- `4096` bytes per register
- whole-register read and write access for tensor instructions

The baseline whole-register interpretations are:

- `64 x 32 BF16` for `vadd.bf16`, `vsub.bf16`, `vmul.bf16`, `vmin.bf16`,
  `vmax.bf16`, `vredsum.bf16`, `vrecip.bf16`, `vexp`, `vrelu`, `vmov`, `vtrpose.xlu`,
  `vreduce.max.xlu`, and `vreduce.sum.xlu`
- `64 x 64 FP8` for MXU activation and weight traffic

When `BF16` data is materialized in the tensor register file, one `64 x 64 BF16` tile
uses two consecutive tensor registers:

- destination base register carries columns `[31:0]`
- destination base plus one carries columns `[63:32]`

The baseline does not require architectural bank visibility.

### 6.3 Weight slots

Each MXU shall contain two distinct weight-slot storage entries.

The baseline implementation direction is weight-stationary:

- weight data is loaded into the selected slot either from `VMEM` by
  `vload.weight.mxuN` or from one tensor register by `vmatpush.weight.mxuN`
- the slot remains resident until overwritten
- each weight slot stores one `64 x 64 FP8` tile

### 6.4 Accumulation buffers

Each MXU shall contain two local `BF16` accumulation buffers, `acc0` and `acc1`.

Baseline requirements:

- each accumulation buffer stores one complete `64 x 64 BF16` tile
- MXU matmul launch writes its result into one selected local accumulation buffer, not directly
  into the tensor register file
- `vmatpush.acc.fp8.*` loads one selected accumulation buffer from one `FP8` tensor register
- `vmatpush.acc.bf16.*` loads one selected accumulation buffer from one `BF16` tensor-register
  pair
- `vmatpop.bf16.acc.*` stores one selected accumulation buffer into one `BF16` tensor-register
  pair
- `vmatpop.fp8.acc.*` stores a quantized `FP8` view of one selected accumulation buffer into one
  tensor register
- there is no direct architectural `VMEM` path into or out of the accumulation buffer

### 6.5 DMA local state

The DMA subsystem shall hold:

- one shared `dma.base` register
- one busy / idle state bit per channel
- per-channel in-flight transfer metadata sufficient to resume progress across cycles

The baseline architecture does not require per-channel base registers.

### 6.6 Structural-conflict handling

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

### 7.2 Blocking on-chip transfers

The following instructions are modeled and implemented as blocking:

- scalar `lb`, `lh`, `lw`, `lbu`, `lhu`, `sb`, `sh`, `sw`
- `seld`
- `vload`
- `vstore`
- `vload.weight.mxu0`
- `vload.weight.mxu1`
- `vmatpush.weight.mxu0`
- `vmatpush.weight.mxu1`
- `vmatpush.acc.fp8.mxu0`
- `vmatpush.acc.fp8.mxu1`
- `vmatpush.acc.bf16.mxu0`
- `vmatpush.acc.bf16.mxu1`
- `vmatpop.fp8.acc.mxu0`
- `vmatpop.fp8.acc.mxu1`
- `vmatpop.bf16.acc.mxu0`
- `vmatpop.bf16.acc.mxu1`

The baseline intent is to keep on-chip movement deterministic and simple to verify.

The cycle-accurate model shall also preserve VMEM ordering across units:

- a VMEM reader shall not observe data older than the most recent completed VMEM writer
- `dma.store.chN` shall not complete before older blocking VMEM writes make their data
  architecturally visible
- this ordering is modeled with fixed completion timing and program order, not with a
  general architectural dependency scoreboard

### 7.3 Asynchronous DMA

DMA shall be channelized and asynchronous.

Each channel supports:

- at most one outstanding transfer
- independent busy / completion state
- `dma.wait.chN` synchronization

The microarchitecture is free to implement the DMA channels with shared internal data
paths or arbitration, provided the architecture-visible channel behavior is preserved.

`dma.wait.chN` and scalar `delay` shall behave as frontend fences in the decode / issue
machinery:

- neither instruction allocates a normal execute-stage slot while it is holding decode
- if channel `N` is already done when `dma.wait.chN` reaches decode, the instruction
  shall spend that cycle in decode and retire directly
- if channel `N` is not yet done, decode shall remain occupied until the transfer
  completes, then the instruction shall retire directly from decode
- younger instructions shall not issue past that decode fence until the wait retires
- `delay N` shall also remain resident in decode for exactly `N` additional core cycles
  after its decode cycle, then retire directly from decode
- `delay 0` shall spend only its decode cycle in `IDU` and retire directly

### 7.4 Baseline transfer formulas

The performance model and timing expectations shall use the following formulas.

Definitions:

- `OFFCHIP_BYTES_PER_BEAT = OFFCHIP_LINK_WIDTH_BITS / 8 = 4`
- `VMEM_BYTES_PER_BEAT = VMEM_BUS_WIDTH_BITS / 8 = 64`

Required formulas:

- `dma_offchip_cycles(bytes) = ceil((bytes + 4 * DMA_OFFCHIP_COMMAND_WORDS) / OFFCHIP_BYTES_PER_BEAT) * OFFCHIP_LINK_CORE_CYCLES_PER_BEAT`
- `vmem_transfer_cycles(bytes) = ceil(bytes / VMEM_BYTES_PER_BEAT) * VMEM_BUS_CORE_CYCLES_PER_BEAT`
- `dma_transfer_cycles(bytes) = max(dma_offchip_cycles(bytes), vmem_transfer_cycles(bytes))`

For the frozen baseline values:

- one off-chip beat costs `2` core cycles
- one VMEM beat costs `1` core cycle
- one `vload` or `vstore` of a `4096`-byte tensor register takes `64` cycles
- one `vload.weight.*` or `vmatpush.weight.*` of a `4096`-byte weight tile takes `64`
  cycles
- one `vmatpush.acc.fp8.*` or `vmatpop.fp8.acc.*` of a `4096`-byte `FP8` tile takes
  `64` cycles
- one `vmatpush.acc.bf16.*` or `vmatpop.bf16.acc.*` of an `8192`-byte `BF16` tile takes
  `128` cycles

## 8. Functional Units

### 8.1 MXU0 and MXU1

The two MXUs share the same architectural interface but intentionally differ internally.

Baseline intent:

- `mxu0`: systolic-array accumulation
- `mxu1`: inner-product-tree accumulation

Shared microarchitectural requirements:

- deterministic latency class of `64` cycles per `vmatmul.*` launch
- whole-register activation source
- resident local weight-slot source
- `BF16` architectural accumulation
- square `64 x 64` array geometry for both `mxu0` and `mxu1`
- one local `64 x 64 BF16` accumulation buffer per MXU
- tensor-register-only accumulator preload and spill path
- local quantization path for `vmatpop.fp8.acc.*`
- ability to overlap with scalar and other long-chime units, subject to issue policy

### 8.2 VPU

The VPU baseline shall implement the following whole-register operations:

- `vadd.bf16`
- `vredsum.bf16`
- `vsub.bf16`
- `vmin.bf16`
- `vmax.bf16`
- `vmul.bf16`
- `vmov`
- `vrecip.bf16`
- `vexp`
- `vrelu`

Timing requirements:

- pipelineable elementwise operations use the `2`-cycle latency class
- non-pipelineable operations such as `vexp` and `vrecip.bf16` use the `8`-cycle latency
  class
- the baseline lane count is `32 BF16` lanes

### 8.3 XLU

The XLU baseline shall implement whole-register transform and reduction operations over
the baseline `64 x 32 BF16` view.

Supported operations:

- `vtrpose.xlu`
- `vreduce.max.xlu`
- `vreduce.sum.xlu`

Timing requirement:

- each baseline `XLU` operation uses the `4`-cycle latency class

## 9. Reset, Initialization, and Model Contract

### 9.1 Reset behavior

Hardware shall reset control state, but the architecture does not require zeroed data
memories or registers.

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
- tensor-execution-unit occupancy
- memory-region-visible traffic

The baseline trace timestamp granularity is `1` trace time unit per modeled core cycle.

For the current visualization convention used by `penguin-model`, one modeled cycle is
annotated as `1 us` of trace time.

The JSON trace shall also expose a free-running `cycle` counter stream:

- the `cycle` trace counter increments by `1` on every simulator core cycle
- it is emitted at timestamps spaced by `ticks_per_cycle`
- it does not stall during frontend bubbles, fences, or long-latency execution

Frontend trace annotations shall follow the cycle-driven model:

- instruction fetch is annotated on every cycle where the frontend launches a new
  instruction
- fetch occupies exactly one modeled cycle in the trace
- if an `IFU` output is not claimed on the next cycle, the fetch stage ends on that first
  blocked cycle and the remaining stall interval is shown as a gap before decode begins
- `dma.wait.chN` and scalar `delay` are the baseline cases that may hold decode and
  therefore create a fetch gap for younger instructions
- a decode-resident `dma.wait.chN` may still leave one younger instruction buffered in
  the `IFU`; no further fetches occur until the wait retires and the buffered output is
  claimed
- a decode-resident `delay` follows that same `IFU` buffering rule

The baseline trace lane split is:

- `IFU`
- `IDU`
- `EXU.SALU`
- `EXU.DMA`
- `EXU.MXU0`
- `EXU.MXU1`
- `EXU.VPU`
- `EXU.XLU`

## 11. Performance-Model Interpretation

The current throughput model shall interpret the frozen parameters in normalized
`ops/cycle` and `bytes/cycle` units unless an explicit clock frequency is provided
externally.

Required derived roofs:

- DRAM roof from `2 B/cycle`
- VMEM roof from `64 B/cycle`
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

- instruction width and frozen opcode map
- two control-flow delay slots
- the three-region memory map
- DMA channelization and fence-by-channel behavior
- the frozen timing-class and bandwidth parameters above
