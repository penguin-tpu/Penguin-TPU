# Scalar Core Encoding And RTL Bring-Up Plan

Status: Proposed implementation plan for the first scalar RTL slice

## 1. Purpose

This document defines the proposed binary encoding baseline for the Penguin
scalar core and a step-by-step RTL implementation plan for bringing that core
up in hardware.

The immediate goal is not to design the entire accelerator ISA at once. The
goal is to make the scalar core concrete enough that RTL work can start without
creating a second unofficial ISA beside the current assembly-level spec.

## 2. Scope

This plan covers:

- the first binary encoding contract for the scalar integer subset
- the relationship between Penguin scalar mnemonics and RV32I field layouts
- the decode outputs the scalar RTL must produce
- a module-by-module RTL implementation plan
- a verification plan that ties RTL back to the existing Python model and test
  vectors

This plan does not yet freeze:

- the binary encoding of DMA, VPU, MXU, XLU, `vload`, `vstore`, or
  `mxu.push.*`
- a full CSR instruction-access model
- caches, interrupts, privilege levels, or a general exception architecture

## 3. Guiding Decisions

### 3.1 Keep the first scalar binary compatible with RV32I field layouts

Decision:

- Penguin scalar instructions should use standard RV32I 32-bit instruction
  formats and bit placements for the first scalar bring-up

Why:

- the existing functional model already groups scalar instructions into
  `RType`, `IType`, `SType`, `BType`, `UType`, and `JType`
- decode logic is simpler and well understood
- assembler, test vectors, and future disassembly tools can reuse mature
  patterns
- semantic deltas such as VMEM-only loads/stores and 2 delay slots do not
  require a new binary encoding

### 3.2 Keep the Penguin mnemonic layer separate from the binary layer

Decision:

- the assembly mnemonics remain Penguin-specific (`saddi`, `sld`, `sst`, and so
  on)
- the binary layer reuses the underlying RV32I opcode and field assignment

Why:

- software stays aligned with the Penguin naming contract already frozen in the
  scalar subset spec
- RTL still benefits from a conservative, familiar encoding

### 3.3 Reserve explicit opcode space for future accelerator instructions

Decision:

- the scalar core should initially implement only the standard scalar opcodes it
  needs
- the overall encoding plan should reserve the RISC-V custom major opcodes for
  Penguin accelerator instructions later

Why:

- this avoids forcing a second encoding migration once tensor RTL starts
- the scalar decoder can explicitly classify unknown custom opcodes as
  “reserved for future Penguin extensions” rather than accidentally treating
  them as illegal forever

## 4. Proposed Encoding Baseline

### 4.1 Fixed-width instruction rule

- every instruction is 32 bits
- every instruction is 4-byte aligned
- IMEM fetch width remains one instruction per access in the first scalar core

### 4.2 Scalar instruction formats

The scalar core should implement the standard six RV32I layouts:

### R-type

Used for:

- `sadd`
- `ssub`
- `ssll`
- `sslt`
- `ssltu`
- `sxor`
- `ssrl`
- `ssra`
- `sor`
- `sand`

Bit layout:

```text
31:25 funct7 | 24:20 rs2 | 19:15 rs1 | 14:12 funct3 | 11:7 rd | 6:0 opcode
```

### I-type

Used for:

- `saddi`
- `sslti`
- `ssltiu`
- `sxori`
- `sori`
- `sandi`
- `sslli`
- `ssrli`
- `ssrai`
- `sld`
- `sjalr`
- `secall`
- `sebreak`

Bit layout:

```text
31:20 imm[11:0] | 19:15 rs1 | 14:12 funct3 | 11:7 rd | 6:0 opcode
```

### S-type

Used for:

- `sst`

Bit layout:

```text
31:25 imm[11:5] | 24:20 rs2 | 19:15 rs1 | 14:12 funct3 | 11:7 imm[4:0] | 6:0 opcode
```

### B-type

Used for:

- `sbeq`
- `sbne`
- `sblt`
- `sbge`
- `sbltu`
- `sbgeu`

Bit layout:

```text
31 imm[12] | 30:25 imm[10:5] | 24:20 rs2 | 19:15 rs1 | 14:12 funct3 |
11:8 imm[4:1] | 7 imm[11] | 6:0 opcode
```

### U-type

Used for:

- `slui`
- `sauipc`

Bit layout:

```text
31:12 imm[31:12] | 11:7 rd | 6:0 opcode
```

### J-type

Used for:

- `sjal`

Bit layout:

```text
31 imm[20] | 30:21 imm[10:1] | 20 imm[11] | 19:12 imm[19:12] | 11:7 rd | 6:0 opcode
```

### 4.3 Scalar mnemonic to binary mapping

The first scalar RTL should use the following opcode map.

| Penguin mnemonic | Underlying form | opcode[6:0] | funct3 | funct7 / imm[11:0] rule |
|---|---|---|---|---|
| `slui` | U | `0110111` | - | standard `lui` encoding |
| `sauipc` | U | `0010111` | - | standard `auipc` encoding |
| `sjal` | J | `1101111` | - | standard `jal` encoding |
| `sjalr` | I | `1100111` | `000` | standard `jalr` encoding |
| `sbeq` | B | `1100011` | `000` | standard `beq` encoding |
| `sbne` | B | `1100011` | `001` | standard `bne` encoding |
| `sblt` | B | `1100011` | `100` | standard `blt` encoding |
| `sbge` | B | `1100011` | `101` | standard `bge` encoding |
| `sbltu` | B | `1100011` | `110` | standard `bltu` encoding |
| `sbgeu` | B | `1100011` | `111` | standard `bgeu` encoding |
| `sld` | I | `0000011` | `010` | binary-compatible with `lw`; Penguin semantics bind it to VMEM |
| `sst` | S | `0100011` | `010` | binary-compatible with `sw`; Penguin semantics bind it to VMEM |
| `saddi` | I | `0010011` | `000` | standard `addi` encoding |
| `sslti` | I | `0010011` | `010` | standard `slti` encoding |
| `ssltiu` | I | `0010011` | `011` | standard `sltiu` encoding |
| `sxori` | I | `0010011` | `100` | standard `xori` encoding |
| `sori` | I | `0010011` | `110` | standard `ori` encoding |
| `sandi` | I | `0010011` | `111` | standard `andi` encoding |
| `sslli` | I | `0010011` | `001` | `funct7=0000000` in imm[11:5] |
| `ssrli` | I | `0010011` | `101` | `funct7=0000000` in imm[11:5] |
| `ssrai` | I | `0010011` | `101` | `funct7=0100000` in imm[11:5] |
| `sadd` | R | `0110011` | `000` | `funct7=0000000` |
| `ssub` | R | `0110011` | `000` | `funct7=0100000` |
| `ssll` | R | `0110011` | `001` | `funct7=0000000` |
| `sslt` | R | `0110011` | `010` | `funct7=0000000` |
| `ssltu` | R | `0110011` | `011` | `funct7=0000000` |
| `sxor` | R | `0110011` | `100` | `funct7=0000000` |
| `ssrl` | R | `0110011` | `101` | `funct7=0000000` |
| `ssra` | R | `0110011` | `101` | `funct7=0100000` |
| `sor` | R | `0110011` | `110` | `funct7=0000000` |
| `sand` | R | `0110011` | `111` | `funct7=0000000` |
| `sfence` | I | `0001111` | `000` | encode as `fence`; no-op in first Penguin scalar core |
| `secall` | I | `1110011` | `000` | `imm[11:0]=000000000000` |
| `sebreak` | I | `1110011` | `000` | `imm[11:0]=000000000001` |

### 4.4 Semantic differences from base RV32I

These are architecture semantics, not encoding changes:

- `sld` / `sst` access VMEM instead of a flat byte-addressed memory map
- branches and jumps have 2 architecturally visible delay slots
- `sfence` is architecturally legal but functionally a no-op in the current
  single-context machine
- misaligned access and misaligned control-transfer targets halt execution
  instead of trapping into a general exception handler

### 4.5 Reserved major opcodes for future Penguin accelerator instructions

The scalar core should not implement these yet, but the plan should reserve
them now:

| Major opcode | RISC-V name | Proposed Penguin use |
|---|---|---|
| `0001011` | `custom-0` | DMA wait, host/control, and future scalar-visible accelerator control |
| `0101011` | `custom-1` | VMEM/tensor movement such as `vload`, `vstore`, `mxu.push.*` |
| `1011011` | `custom-2` | MXU/VPU/XLU compute issue forms |
| `1111011` | `custom-3` | overflow space for future accelerator encodings |

Planning rule:

- the first scalar decoder should classify these opcodes separately from “fully
  illegal” encodings
- the scalar-only top level may still report them as unsupported for now
- later accelerator decode should plug into those reserved paths instead of
  rewriting scalar opcode assignments

## 5. Required RTL Decode Outputs

The first scalar decoder should convert one 32-bit instruction into a compact
internal control record with at least the following fields:

- `valid`
- `illegal`
- `format_class`: `R`, `I`, `S`, `B`, `U`, `J`, `SYSTEM`, `MISC_MEM`, `RESERVED_CUSTOM`
- `scalar_op_class`: `ALU_REG`, `ALU_IMM`, `LOAD`, `STORE`, `BRANCH`, `JUMP`,
  `UPPER_IMM`, `SYSTEM`
- `alu_fn`: `ADD`, `SUB`, `SLT`, `SLTU`, `XOR`, `OR`, `AND`, `SLL`, `SRL`, `SRA`,
  `COPY_B`, `COMPARE_EQ`, `COMPARE_NE`, `COMPARE_LT`, `COMPARE_GE`,
  `COMPARE_LTU`, `COMPARE_GEU`
- `rd`, `rs1`, `rs2`
- `imm32`
- `writes_rd`
- `reads_rs1`
- `reads_rs2`
- `is_branch`
- `is_jump`
- `is_load`
- `is_store`
- `is_fence`
- `is_ecall`
- `is_ebreak`

Immediate generation rules should follow standard RV32I sign-extension and
bit-reassembly behavior.

## 6. Recommended RTL Partitioning

Suggested first scalar RTL subtree:

```text
rtl/penguin_tpu/scalar/
  penguin_scalar_defs.vh
  penguin_scalar_decoder.v
  penguin_scalar_regfile.v
  penguin_scalar_alu.v
  penguin_scalar_branch_unit.v
  penguin_scalar_lsu.v
  penguin_scalar_controller.v
  penguin_scalar_core.v
```

Recommended responsibilities:

- `penguin_scalar_defs.vh`
  - opcode, funct3, funct7, halt-reason, and internal ALU-function constants
- `penguin_scalar_decoder.v`
  - binary field extraction
  - immediate generation
  - illegal-instruction detection
  - control-record generation
- `penguin_scalar_regfile.v`
  - 32 x 32-bit scalar register file
  - hardwired `x0`
- `penguin_scalar_alu.v`
  - integer ALU and compare datapath
- `penguin_scalar_branch_unit.v`
  - branch condition evaluation
  - branch/jump target generation
  - alignment checks
- `penguin_scalar_lsu.v`
  - VMEM word load/store interface
  - 4-byte alignment checks
- `penguin_scalar_controller.v`
  - `pc` sequencing
  - issue/retire sequencing
  - 2-delay-slot redirect bookkeeping
  - halt-status generation
- `penguin_scalar_core.v`
  - top-level integration of fetch, decode, execute, and memory interfaces

## 7. Step-By-Step Implementation Plan

### Step 0: freeze the scalar encoding contract

Actions:

- review this plan against the scalar subset spec
- confirm that the project accepts RV32I-compatible scalar binary encodings
- confirm that future accelerator instructions will use reserved custom opcodes

Deliverables:

- this document accepted as the working plan
- one follow-up action to convert the scalar encoding section into a formal spec

Exit criteria:

- there is one agreed scalar binary story for software, model, and RTL

### Step 1: create shared scalar decode constants

Actions:

- add `penguin_scalar_defs.vh`
- define localparams for all scalar major opcodes, funct3 values, and
  distinguishing funct7 values
- define halt-reason codes for:
  - illegal instruction
  - instruction-address misaligned
  - load misaligned
  - store misaligned
  - `secall`
  - `sebreak`

Deliverables:

- one header file used by all scalar RTL modules

Exit criteria:

- no scalar RTL module hardcodes raw opcode bit patterns inline

### Step 2: implement a standalone scalar decoder

Actions:

- implement `penguin_scalar_decoder.v`
- extract `opcode`, `rd`, `rs1`, `rs2`, `funct3`, `funct7`
- build sign-extended immediates for I/S/B/U/J formats
- generate one decoded control record
- explicitly mark reserved custom opcodes as reserved, not generic garbage

Deliverables:

- synthesizable combinational decoder
- decoder truth-table comments or a small markdown appendix if needed

Exit criteria:

- every legal scalar instruction in Section 4.3 decodes to a unique control
  record
- illegal or unsupported encodings produce `illegal=1`

### Step 3: implement the scalar register file

Actions:

- implement `penguin_scalar_regfile.v`
- provide two read ports and one write port
- hardwire `x0` to zero
- ignore writes to `x0`

Deliverables:

- simple synchronous-write register file module

Exit criteria:

- directed tests prove `x0` always reads as zero

### Step 4: implement the ALU and compare block

Actions:

- implement `penguin_scalar_alu.v`
- support add/sub, logical ops, shifts, signed and unsigned comparisons
- support branch compare outputs in a reusable way

Deliverables:

- standalone ALU module

Exit criteria:

- ALU outputs match the Python model for all scalar arithmetic operations

### Step 5: implement branch and jump target logic

Actions:

- implement `penguin_scalar_branch_unit.v`
- compute:
  - branch-taken result
  - `pc + imm` target
  - `(rs1 + imm) & ~1` `sjalr` target
- check 4-byte alignment after architectural rules are applied

Deliverables:

- standalone branch/jump helper module

Exit criteria:

- target generation matches the functional model
- misaligned targets are detected before commit

### Step 6: implement the control path for 2 delay slots

Actions:

- implement redirect-pending state in `penguin_scalar_controller.v`
- track:
  - pending redirect valid
  - pending redirect target
  - pending redirect countdown
- define precise update priority for younger control transfers in delay slots

Deliverables:

- one control FSM or equivalent sequential control block

Exit criteria:

- directed tests cover:
  - untaken branch
  - taken branch
  - `sjal`
  - `sjalr`
  - younger branch replacing older pending redirect

### Step 7: implement the scalar LSU for VMEM word accesses

Actions:

- implement `penguin_scalar_lsu.v`
- support aligned 32-bit word load/store only
- raise halt status on misaligned load/store
- initially use a simple blocking memory interface

Deliverables:

- load/store unit with VMEM-side request and response signals

Exit criteria:

- `sld` / `sst` pass against directed VMEM tests
- misaligned accesses halt correctly

### Step 8: implement system and halt behavior

Actions:

- wire `sfence`, `secall`, and `sebreak`
- `sfence` becomes a legal no-op
- `secall` and `sebreak` raise explicit terminal status

Deliverables:

- complete scalar halt-status path

Exit criteria:

- software-visible halt reasons match the functional model

### Step 9: integrate a minimal scalar core top level

Actions:

- implement `penguin_scalar_core.v`
- integrate:
  - IMEM fetch
  - decoder
  - register file
  - ALU
  - branch unit
  - LSU
  - controller
- keep the first core single-issue and in-order

Deliverables:

- first executable scalar core top level

Exit criteria:

- one simple hand-written scalar program executes end to end in RTL simulation

### Step 10: add binary encode/decode support on the software side

Actions:

- add binary encoder helpers to the Python assembler
- add a decoder or disassembler helper for testbench visibility
- keep assembly as the human-facing source of truth, but enable binary IMEM
  images for RTL tests

Deliverables:

- one software path that can emit 32-bit encoded scalar instruction words

Exit criteria:

- the same assembly test vectors can drive both the Python model and RTL IMEM
  initialization

### Step 11: connect RTL verification to existing scalar vectors

Actions:

- reuse existing `tests/vectors/programs/scalar/` programs wherever possible
- add cocotb tests for:
  - decoder spot checks
  - ALU/register-file behavior
  - branch/jump delay-slot behavior
  - VMEM load/store behavior
  - end-to-end scalar program execution

Deliverables:

- a reusable RTL regression path for scalar bring-up

Exit criteria:

- RTL scalar tests compare against the Python model, not just ad hoc expected
  values

### Step 12: define the stop point for the first scalar RTL milestone

The first scalar RTL milestone should be considered complete only when all of
the following are true:

- every instruction in the current scalar subset spec has a frozen binary
  encoding
- the scalar core fetches and executes those instructions from IMEM
- `x0` behavior is correct
- 2-delay-slot control-flow behavior is correct
- misaligned control-flow and misaligned `sld` / `sst` halt correctly
- `secall` and `sebreak` report distinct terminal outcomes
- cocotb regression exists and runs in CI

## 8. Verification Plan

### 8.1 Decoder verification

- generate directed binary words for each legal scalar instruction family
- check decoded fields against expected `rd`, `rs1`, `rs2`, immediate, and op
  class
- include illegal encodings:
  - unsupported `funct3`
  - unsupported `funct7`
  - reserved custom opcodes

### 8.2 Datapath verification

- ALU unit tests for all operations
- register-file tests for x0 hardwiring and normal writeback
- LSU tests for aligned and misaligned VMEM accesses

### 8.3 Control-path verification

- branch taken and not taken
- `sjal` link-register correctness
- `sjalr` target masking and alignment behavior
- nested control transfer inside delay slots

### 8.4 End-to-end verification

- run the same scalar assembly programs in:
  - the Python model
  - the RTL scalar core
- compare:
  - final scalar register file
  - final `pc`
  - final VMEM contents touched by the program
  - halt reason

## 9. Open Questions To Resolve Before RTL Coding Starts

- whether scalar RTL should power up non-`x0` registers as zero for bring-up, or
  whether testbenches should explicitly initialize them to match the
  architecturally undefined spec state
- whether the first scalar core exposes a simple single-cycle IMEM/VMEM
  interface, or a request/response handshake from day one
- whether binary assembly artifacts should be checked in beside the existing
  text assembly programs, or generated on demand during tests
- whether `sfence` should remain a distinct decoded op in the controller, or be
  normalized to a generic no-op after decode

## 10. Immediate Next Action After This Plan

The next concrete follow-up should be:

1. convert Section 4 into a formal scalar binary-encoding spec under
   `docs/specs/`
2. implement the standalone scalar decoder in RTL
3. add decoder-focused cocotb tests before integrating the full scalar core
