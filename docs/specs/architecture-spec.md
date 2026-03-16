# Penguin Architecture Specification

Status: Baseline 1.0

## 1. Scope

This document is the normative architecture-visible specification for Penguin-TPU.

It defines:

- the architectural execution model
- the visible machine state
- the architectural memory map
- the scalar and tensor instruction-set contract
- architecturally visible data formats
- architecturally visible error and halt behavior
- frozen architectural constants shared by software, the functional model, RTL, and
  system integration

This document does not define a specific RTL pipeline, arbitration network, buffer
implementation, or physical datapath layout. Those subjects are defined in the
microarchitecture specification.

## 2. Normative Language

The key words `shall`, `must`, `must not`, `should`, and `may` are to be interpreted as
requirements for any conforming Penguin implementation or model.

Conformance to this specification means conformance of the following software-visible
surfaces:

- assembly source and binary encoding
- runtime memory images
- machine-visible execution state
- architecturally visible instruction results
- architecturally visible halt and error behavior

## 3. Architectural Overview

Penguin is a statically scheduled accelerator-oriented machine with a scalar control path
and a tile-oriented tensor datapath.

The baseline machine contains:

- one scalar integer control path
- one architectural tensor register file of `64` registers, `m0` through `m63`
- two architecturally visible matrix execution units, `mxu0` and `mxu1`
- one vector processing unit, `vpu`
- one transpose unit, `xlu`
- one instruction memory, `IMEM`
- one on-chip tensor/vector memory, `VMEM`
- one off-chip backing memory, `DRAM`
- eight architected DMA channels between `DRAM` and `VMEM`

The machine is intentionally narrow in the frontend:

- one fixed-width 32-bit instruction stream
- one fetch stream
- one dispatch decision per cycle

The machine is intentionally explicit at the asynchronous boundary:

- on-chip execution is deterministic for a fixed instruction sequence and configuration
- `DRAM <-> VMEM` transfer is asynchronous and channelized
- all other architected tensor and scalar transfers are blocking

## 4. Architectural Execution Model

### 4.1 General rules

The baseline Penguin execution model shall satisfy the following rules:

- instructions are fixed-width `32`-bit words
- instructions conceptually retire in program order
- the scalar frontend issues at most one new instruction per cycle
- long-chime tensor instructions may remain active for multiple cycles after issue
- different execution units may be active concurrently
- architectural completion order remains defined by the instruction semantics, not by
  internal speculation or out-of-order retirement

### 4.2 Control-flow delay slots

Branches and jumps shall have exactly `2` architecturally visible delay slots.

For a control-transfer instruction at address `pc`:

- the instructions at `pc + 4` and `pc + 8` shall retire before the resolved redirect is
  applied
- if the control transfer is not taken, those same delay-slot instructions still retire
- if a younger branch or jump executes in a delay slot, the younger unresolved redirect
  shall replace the older pending redirect

This rule applies to:

- `sjal`
- `sjalr`
- `sbeq`
- `sbne`
- `sblt`
- `sbge`
- `sbltu`
- `sbgeu`

### 4.3 Halt model

The baseline architecture shall support explicit host-visible completion and error halt
observation.

Penguin shall not provide a general trap-and-resume architectural model in this revision.
Illegal or misaligned architectural behavior stops execution and reports halt status.

Architecturally meaningful stop classes are:

- normal end-of-program completion
- `secall`
- `sebreak`
- illegal instruction
- instruction-address misaligned
- misaligned scalar memory access
- illegal DMA issue
- other architecturally defined fatal model errors

## 5. Architectural State

### 5.1 Scalar state

The scalar architectural state shall include:

- `32` general-purpose integer registers, `x0` through `x31`
- one `32`-bit program counter `pc`

Requirements:

- each scalar register stores one `32`-bit value
- `x0` is hardwired to zero
- `pc` is a byte address
- `pc` shall remain `4`-byte aligned

### 5.2 Tensor state

The tensor architectural state shall include:

- `64` tensor registers, `m0` through `m63`
- two weight slots per MXU: `mxu0.w0`, `mxu0.w1`, `mxu1.w0`, and `mxu1.w1`

Tensor-register requirements:

- each tensor register stores `64` rows
- each row stores exactly `32` bytes
- each tensor register therefore stores `2048` raw bytes
- the tensor register file is flat and type-agnostic
- tensor element interpretation is selected by the instruction semantics, not by the
  storage class

### 5.3 Control and status state

The architecture-visible execution-control plane shall include at least:

- execution enable / halt control
- execution status / stop reason
- `pc` visibility
- one shared high-address extension CSR named `MEM_BASE`
- DMA busy state for the eight architected DMA channels

The exact MMIO encoding is left to system integration, but the state itself is
architecturally required.

## 6. Architectural Memory Organization

### 6.1 Memory regions

Penguin shall expose three disjoint architectural memory regions.

| Region | Base Address | Size | Role |
|---|---:|---:|---|
| `IMEM` | `0x0010_0000` | `32 KiB` | Instruction memory |
| `VMEM` | `0x0800_0000` | `1 MiB` | On-chip tensor/vector data memory |
| `DRAM` | `0x8000_0000` | `16 GiB` | Off-chip backing data memory |

Memory-region rules:

- `IMEM`, `VMEM`, and `DRAM` are byte-addressed
- instruction fetch conceptually reads `IMEM`
- scalar data load/store access `VMEM` only
- tensor register load/store and MXU weight push access `VMEM` only
- DMA is the only architected path between `DRAM` and `VMEM`

### 6.2 Alignment rules

The following alignment rules are architectural:

- instruction fetch address alignment: `4` bytes
- scalar `sld` / `sst` alignment: `4` bytes
- DMA source address alignment: `32` bytes
- DMA destination address alignment: `32` bytes
- DMA size granularity: multiple of `32` bytes
- `vload`, `vstore`, and `mxu.push.*` VMEM address alignment: `32` bytes

### 6.3 Initialization rules

The architecture does not define deterministic reset contents for general data storage.

Unless software or host setup explicitly initializes them:

- `DRAM` contents are architecturally undefined
- `VMEM` contents are architecturally undefined
- scalar registers other than `x0` are architecturally undefined
- tensor registers and MXU weight slots are architecturally undefined

The host shall populate `IMEM` before enabling accelerator execution.

## 7. Scalar ISA

### 7.1 Naming convention

Scalar integer mnemonics are derived from RV32I mnemonics by prefixing `s`, except that
scalar memory access uses:

- `sld`
- `sst`

The baseline scalar subset intentionally excludes:

- floating-point scalar state
- byte and halfword scalar loads and stores
- unaligned scalar load/store support

### 7.2 Binary encoding baseline

The scalar binary encoding baseline shall remain RV32I-compatible:

- fixed-width `32`-bit instruction words
- standard RV32I field layouts for `R`, `I`, `S`, `B`, `U`, and `J` formats
- standard RV32I opcode / `funct3` / `funct7` placement for the supported scalar subset

Required binary compatibility points:

- `sld` reuses the `lw` encoding shape
- `sst` reuses the `sw` encoding shape
- `sfence` reuses the `fence` encoding shape
- `secall` and `sebreak` reuse the standard system encodings

Penguin accelerator instructions shall use the standard RISC-V custom major opcodes.
Their allocation is defined in Section 8.

### 7.3 Scalar instruction set

#### 7.3.1 Upper-immediate instructions

| Mnemonic | Semantics |
|---|---|
| `slui rd, imm20` | `x[rd] <- imm20 << 12` |
| `sauipc rd, imm20` | `x[rd] <- pc + (imm20 << 12)` |

#### 7.3.2 Jumps

| Mnemonic | Semantics |
|---|---|
| `sjal rd, imm` | `x[rd] <- pc + 4`; redirect to `pc + imm` after 2 delay slots |
| `sjalr rd, rs1, imm` | `target <- (x[rs1] + imm) & ~1`; `x[rd] <- pc + 4`; redirect after 2 delay slots |

#### 7.3.3 Branches

| Mnemonic | Branch condition |
|---|---|
| `sbeq rs1, rs2, imm` | `x[rs1] == x[rs2]` |
| `sbne rs1, rs2, imm` | `x[rs1] != x[rs2]` |
| `sblt rs1, rs2, imm` | `signed(x[rs1]) < signed(x[rs2])` |
| `sbge rs1, rs2, imm` | `signed(x[rs1]) >= signed(x[rs2])` |
| `sbltu rs1, rs2, imm` | `unsigned(x[rs1]) < unsigned(x[rs2])` |
| `sbgeu rs1, rs2, imm` | `unsigned(x[rs1]) >= unsigned(x[rs2])` |

If the branch is taken, the redirect target is `pc + imm` after the two required delay
slots. If the branch is not taken, execution continues sequentially after those same
delay-slot instructions retire.

#### 7.3.4 Immediate ALU operations

| Mnemonic | Semantics |
|---|---|
| `saddi rd, rs1, imm` | `x[rd] <- x[rs1] + imm` |
| `sslti rd, rs1, imm` | `x[rd] <- 1` if `signed(x[rs1]) < signed(imm)` else `0` |
| `ssltiu rd, rs1, imm` | `x[rd] <- 1` if `unsigned(x[rs1]) < unsigned(sign_extend(imm))` else `0` |
| `sxori rd, rs1, imm` | `x[rd] <- x[rs1] xor sign_extend(imm)` |
| `sori rd, rs1, imm` | `x[rd] <- x[rs1] or sign_extend(imm)` |
| `sandi rd, rs1, imm` | `x[rd] <- x[rs1] and sign_extend(imm)` |
| `sslli rd, rs1, shamt` | `x[rd] <- x[rs1] << (shamt & 0x1f)` |
| `ssrli rd, rs1, shamt` | `x[rd] <- unsigned(x[rs1]) >> (shamt & 0x1f)` |
| `ssrai rd, rs1, shamt` | `x[rd] <- signed(x[rs1]) >>> (shamt & 0x1f)` |

#### 7.3.5 Register-register ALU operations

| Mnemonic | Semantics |
|---|---|
| `sadd rd, rs1, rs2` | `x[rd] <- x[rs1] + x[rs2]` |
| `ssub rd, rs1, rs2` | `x[rd] <- x[rs1] - x[rs2]` |
| `ssll rd, rs1, rs2` | `x[rd] <- x[rs1] << (x[rs2] & 0x1f)` |
| `sslt rd, rs1, rs2` | `x[rd] <- 1` if `signed(x[rs1]) < signed(x[rs2])` else `0` |
| `ssltu rd, rs1, rs2` | `x[rd] <- 1` if `unsigned(x[rs1]) < unsigned(x[rs2])` else `0` |
| `sxor rd, rs1, rs2` | `x[rd] <- x[rs1] xor x[rs2]` |
| `ssrl rd, rs1, rs2` | `x[rd] <- unsigned(x[rs1]) >> (x[rs2] & 0x1f)` |
| `ssra rd, rs1, rs2` | `x[rd] <- signed(x[rs1]) >>> (x[rs2] & 0x1f)` |
| `sor rd, rs1, rs2` | `x[rd] <- x[rs1] or x[rs2]` |
| `sand rd, rs1, rs2` | `x[rd] <- x[rs1] and x[rs2]` |

All scalar integer arithmetic is modulo `2^32` unless signed comparison semantics are
explicitly stated.

#### 7.3.6 Scalar memory operations

| Mnemonic | Semantics |
|---|---|
| `sld rd, imm(rs1)` | `x[rd] <- VMEM32[x[rs1] + imm]` |
| `sst rs2, imm(rs1)` | `VMEM32[x[rs1] + imm] <- x[rs2]` |

Scalar memory requirements:

- exactly one aligned `32`-bit word is transferred
- the effective address must be `4`-byte aligned
- no partial-word scalar memory operation exists in the baseline architecture

#### 7.3.7 Ordering and environment operations

| Mnemonic | Semantics |
|---|---|
| `sfence` | architecturally legal; baseline no-op |
| `secall` | terminate execution with environment-call halt status |
| `sebreak` | terminate execution with breakpoint halt status |

## 8. Tensor and Accelerator ISA

### 8.1 Tensor data views

The tensor register file shall support the following architecturally visible views.

| View | Elements per row | Logical tile shape |
|---|---:|---|
| `FP8_e4m3` | `32` | `64 x 32` |
| `BF16` | `16` | `64 x 16` |

### 8.2 Accelerator binary encoding overview

All accelerator instructions shall use fixed-width `32`-bit words and one of the four
standard RISC-V custom major opcodes.

| Major opcode | Bits `[6:0]` | Family | Baseline instruction families |
|---|---|---|---|
| `custom-0` | `0001011` | Memory movement | `dma.*`, `vload`, `vstore`, `mxu.push.*` |
| `custom-1` | `0101011` | MXU launch | `matmul.*` |
| `custom-2` | `1011011` | VPU whole-register ops | `vadd`, `vsub`, `vmul`, `vmax`, `vmin`, `vrelu`, `vmov`, `vexp`, `vrecip` |
| `custom-3` | `1111011` | XLU whole-register ops | `transpose.xlu`, `reduce.max.xlu`, `reduce.sum.xlu` |

### 8.3 Common accelerator field conventions

The accelerator encodings shall follow these common rules:

- scalar `x` registers remain encoded in standard `5`-bit RISC-V register positions
- tensor `m` registers are architecturally `6` bits wide and therefore use one dedicated
  high-bit field plus one standard `5`-bit low field
- reserved bits shall be written as zero by software and shall be treated as illegal if
  observed nonzero by a conforming implementation

Tensor-register reconstruction rules:

- `m = {m_hi, m_lo}`
- `m_hi` is the most-significant tensor-register index bit
- `m_lo` occupies the standard `rd`, `rs1`, or `rs2` field position used by the format

Scaled-immediate rule for VMEM-facing tensor transfers:

- `imm11_32` is a signed `11`-bit immediate encoded in units of `32` bytes
- the effective byte offset is `sign_extend(imm11_32) << 5`
- the representable byte-offset range is `[-32768, 32736]`
- software shall use scalar address-generation instructions when a larger offset is
  required

Weight-selector rule:

- the baseline architecture has `2` weight slots per MXU
- `w0` encodes as `0`
- `w1` encodes as `1`

### 8.4 `custom-0`: DMA and VMEM-facing transfer encodings

`custom-0` is split into two formats:

- `TMEM-I` for `vload`, `vstore`, and `mxu.push.*`
- `TDMA-R` for `dma.*`

#### 8.4.1 `TMEM-I` format

| Bits | Field | Meaning |
|---|---|---|
| `[31]` | `t_hi` | High bit of tensor register index for `vload` / `vstore`; reserved for `mxu.push.*` |
| `[30:20]` | `imm11_32` | Signed `11`-bit VMEM offset in units of `32` bytes |
| `[19:15]` | `rs1` | Scalar base register |
| `[14:12]` | `funct3` | Transfer sub-opcode |
| `[11:7]` | `t_lo` | Low bits of tensor register index or encoded weight-slot selector |
| `[6:0]` | `opcode` | `custom-0` = `0001011` |

`TMEM-I` `funct3` assignments:

| `funct3` | Mnemonic family | Operand interpretation |
|---|---|---|
| `000` | `vload` | `m = {t_hi, t_lo}` |
| `001` | `vstore` | `m = {t_hi, t_lo}` |
| `010` | `mxu.push.mxu0` | `wslot = t_lo[0]`; `t_hi = 0`; `t_lo[4:1] = 0` |
| `011` | `mxu.push.mxu1` | `wslot = t_lo[0]`; `t_hi = 0`; `t_lo[4:1] = 0` |
| `100` to `111` | Reserved | Illegal in the baseline |

Architectural rules:

- `vload m<dest>, imm(x<rs1>)` transfers one whole tensor register between `VMEM` and
  `m<dest>`
- `vstore m<src>, imm(x<rs1>)` transfers one whole tensor register between `m<src>` and
  `VMEM`
- `mxu.push.* w<slot>, imm(x<rs1>)` transfers one whole weight tile from `VMEM` into the
  selected MXU weight slot
- the effective address is `x[rs1] + (sign_extend(imm11_32) << 5)`
- these instructions are blocking
- the effective VMEM address shall satisfy the `32`-byte alignment rule

#### 8.4.2 `TDMA-R` format

| Bits | Field | Meaning |
|---|---|---|
| `[31:29]` | `ch` | DMA channel index |
| `[28:27]` | `dma_op` | DMA sub-opcode |
| `[26:25]` | `0` | Reserved, shall be zero |
| `[24:20]` | `rs_size` | Scalar register holding transfer size in bytes |
| `[19:15]` | `rs_vmem` | Scalar register holding VMEM byte address |
| `[14:12]` | `funct3` | `100` for `TDMA-R` |
| `[11:7]` | `rs_dram` | Scalar register holding DRAM byte address |
| `[6:0]` | `opcode` | `custom-0` = `0001011` |

`TDMA-R` `dma_op` assignments:

| `dma_op` | Mnemonic family | Operand interpretation |
|---|---|---|
| `00` | `dma.load.chN` | `N = ch`; DRAM to VMEM |
| `01` | `dma.store.chN` | `N = ch`; VMEM to DRAM |
| `10` | `dma.wait.chN` | `N = ch`; `rs_dram = rs_vmem = rs_size = 0` |
| `11` | Reserved | Illegal in the baseline |

Architectural rules:

- `N` shall be one of `0` through `7`
- `dma.load.chN rs_dram, rs_vmem, rs_size` launches a `DRAM -> VMEM` transfer
- `dma.store.chN rs_dram, rs_vmem, rs_size` launches a `VMEM -> DRAM` transfer
- `dma.wait.chN` waits only on the selected channel
- the channel must be idle before a new `dma.load` or `dma.store` is issued
- DMA moves raw unit-stride bytes only
- completion order across channels is not ordered by issue order
- `dma.wait.chN` completes immediately if the channel is already idle

### 8.5 `custom-1`: MXU launch encodings

`custom-1` carries the MXU launch family in one `TMXU-R` format.

#### 8.5.1 `TMXU-R` format

| Bits | Field | Meaning |
|---|---|---|
| `[31]` | `md_hi` | High bit of destination tensor register |
| `[30]` | `ms_hi` | High bit of activation-source tensor register |
| `[29]` | `mp_hi` | High bit of partial-sum tensor register for accumulate form; zero otherwise |
| `[28]` | `mxu` | MXU selector: `0` for `mxu0`, `1` for `mxu1` |
| `[27]` | `wslot` | Weight-slot selector: `0` for `w0`, `1` for `w1` |
| `[26:25]` | `0` | Reserved, shall be zero |
| `[24:20]` | `mp_lo` | Low bits of partial-sum tensor register for accumulate form; zero otherwise |
| `[19:15]` | `ms_lo` | Low bits of activation-source tensor register |
| `[14:12]` | `funct3` | MXU sub-opcode |
| `[11:7]` | `md_lo` | Low bits of destination tensor register |
| `[6:0]` | `opcode` | `custom-1` = `0101011` |

`TMXU-R` `funct3` assignments:

| `funct3` | Meaning | Architectural mnemonic |
|---|---|---|
| `000` | Fresh matmul; `mp_hi = 0`, `mp_lo = 0` | `matmul.mxu<mxu> m<dest>, m<src>, w<wslot>` |
| `001` | Matmul with explicit partial accumulation | `matmul.add.mxu<mxu> m<dest>, m<src>, w<wslot>, m<partial>` |
| `010` to `111` | Reserved | Illegal in the baseline |

Architectural rules:

- `m<src>` supplies activation operand `A`
- `w<wslot>` selects weight operand `B`
- `m<partial>` supplies the prior partial-sum tile when the accumulate form is used
- the fresh form computes `C = A @ B`
- the accumulate form computes `C = (A @ B) + partial`
- MXU arithmetic is `FP8_e4m3 x FP8_e4m3 -> BF16`
- BF16 accumulation is architecturally visible
- MXU is pure matmul/accumulate only; bias, residual, activation, and other tensor
  postprocessing are separate instructions

### 8.6 `custom-2`: VPU whole-register encodings

`custom-2` carries the VPU family in one `TVEC-R` format.

#### 8.6.1 `TVEC-R` format

| Bits | Field | Meaning |
|---|---|---|
| `[31]` | `md_hi` | High bit of destination tensor register |
| `[30]` | `ms1_hi` | High bit of first source tensor register |
| `[29]` | `ms2_hi` | High bit of second source tensor register for binary form; zero otherwise |
| `[28:25]` | `vec_op` | VPU operation selector |
| `[24:20]` | `ms2_lo` | Low bits of second source tensor register for binary form; zero otherwise |
| `[19:15]` | `ms1_lo` | Low bits of first source tensor register |
| `[14:12]` | `funct3` | VPU form selector |
| `[11:7]` | `md_lo` | Low bits of destination tensor register |
| `[6:0]` | `opcode` | `custom-2` = `1011011` |

`TVEC-R` `funct3` assignments:

| `funct3` | Meaning |
|---|---|
| `000` | Binary whole-register VPU operation |
| `001` | Unary whole-register VPU operation |
| `010` to `111` | Reserved and illegal in the baseline |

Binary `vec_op` assignments (`funct3 = 000`):

| `vec_op` | Mnemonic |
|---|---|
| `0000` | `vadd` |
| `0001` | `vsub` |
| `0010` | `vmul` |
| `0011` | `vmax` |
| `0100` | `vmin` |
| `0101` to `1111` | Reserved |

Unary `vec_op` assignments (`funct3 = 001`):

| `vec_op` | Mnemonic |
|---|---|
| `0000` | `vmov` |
| `0001` | `vrelu` |
| `0010` | `vexp` |
| `0011` | `vrecip` |
| `0100` to `1111` | Reserved |

Architectural rules:

- VPU reads tensor operands from `m` registers
- VPU writes tensor results to `m` registers only
- VPU operates on whole tensor registers only
- the initial floating-point VPU view is BF16 over the `64 x 16` interpretation
- unary forms shall encode `ms2_hi = 0` and `ms2_lo = 0`

### 8.7 `custom-3`: XLU whole-register encodings

`custom-3` carries the XLU family in one `TXLU-R` format.

#### 8.7.1 `TXLU-R` format

| Bits | Field | Meaning |
|---|---|---|
| `[31]` | `md_hi` | High bit of destination tensor register |
| `[30]` | `ms_hi` | High bit of source tensor register |
| `[29:25]` | `xlu_op` | XLU operation selector |
| `[24:20]` | `0` | Reserved, shall be zero |
| `[19:15]` | `ms_lo` | Low bits of source tensor register |
| `[14:12]` | `funct3` | `000` in the baseline |
| `[11:7]` | `md_lo` | Low bits of destination tensor register |
| `[6:0]` | `opcode` | `custom-3` = `1111011` |

`TXLU-R` `xlu_op` assignments:

| `xlu_op` | Mnemonic |
|---|---|
| `00000` | `transpose.xlu` |
| `00001` | `reduce.max.xlu` |
| `00010` | `reduce.sum.xlu` |
| `00011` to `11111` | Reserved |

Architectural rules:

- XLU reads tensor operands from `m` registers
- XLU writes tensor results to `m` registers only
- XLU operates on whole tensor registers only
- the initial transpose and reduction view is BF16 over the `64 x 16` interpretation

## 9. Frozen Architectural Constants

The following constants are architecturally frozen in the current baseline.

| Constant | Value |
|---|---:|
| Instruction width | `32` bits |
| Instruction alignment | `4` bytes |
| Scalar register count | `32` |
| Control-flow delay slots | `2` |
| Tensor register count | `64` |
| Tensor-register rows | `64` |
| Tensor-register row bytes | `32` |
| MXU count | `2` |
| MXU weight slots per MXU | `2` |
| MXU weight tile shape | `32 x 16 FP8` |
| DMA channel count | `8` |
| DMA alignment | `32` bytes |
| VPU initial view | `64 x 16 BF16` |
| XLU initial view | `64 x 16 BF16` |

Timing-class constants used by the baseline performance model are defined in the
microarchitecture specification.

## 10. Error Conditions

The following conditions shall terminate execution with architectural error status:

- illegal or unsupported instruction
- misaligned branch target
- misaligned jump target
- misaligned `sld` or `sst`
- DMA issue to a busy channel
- DMA issue with misaligned address or size
- any other explicitly defined fatal architectural contract violation

The baseline architecture shall not emulate these cases silently.

## 11. Conformance Surfaces

This architecture specification is the required contract for:

- the `penguin-model` functional and performance model
- the scalar encoder / assembler
- compiler-emitted assembly and executable bundles
- RTL blocks implementing the current baseline machine
- FPGA and system software that launch or observe Penguin execution
