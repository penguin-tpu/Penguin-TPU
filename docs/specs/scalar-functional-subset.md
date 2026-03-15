# Penguin Scalar Integer ISA Subset

## Scope

This document defines the first scalar architecture-visible subset for Penguin.
The scalar path is now integer-only and is derived from the RISC-V RV32I base
integer instruction set.

Penguin does not expose the RV32I mnemonics directly. Every scalar mnemonic is
the corresponding RV32I mnemonic with a leading `s`, except for memory access:

- scalar load is `sld`
- scalar store is `sst`

This subset intentionally removes:

- floating-point architectural state
- floating-point instructions
- byte and halfword loads
- byte and halfword stores
- support for unaligned load/store

The goal of this subset is to freeze the scalar integer contract before the
functional model is expanded.

## Architectural State

### Integer register file

- 32 general-purpose registers
- each register holds one 32-bit value
- `x0` is hardwired to zero

### Execution-control and memory-base state

The scalar subset relies on a small amount of additional architecture-visible state that
is not itself fully defined by RV32I-style integer instructions:

- one shared `mem_base` CSR used by memory-like tensor and DMA instructions to extend
  addressing beyond the raw 32-bit scalar-register range
- execution-control state that enables or halts accelerator fetch
- execution-status state that reports halt, done, or error outcome
- DMA busy state for the 8 architected DMA channels

The current scalar instruction subset does not yet define a full CSR-manipulation ISA for
this state. In the baseline launch model, host-side software initializes or observes it.
Later revisions may allow Penguin scalar code to access the same CSR region through
MMIO-style load/store sequences or direct hardware connections.

### Program counter

- `pc` is a 32-bit byte address
- `pc` must remain 4-byte aligned
- instructions conceptually retire in program order
- unless an instruction overrides control flow, the next `pc` is `pc + 4`

### Control-flow delay slots

- branches have exactly 2 architecturally required delay slots
- jumps have exactly 2 architecturally required delay slots
- the same 2-delay-slot rule applies to `sjal` and `sjalr`
- the 2 sequential instructions immediately following a branch or jump always
  retire before control transfers to the resolved target

For a control-transfer instruction at address `pc`, the target redirection takes
effect only after the instructions at `pc + 4` and `pc + 8` retire.

If a branch or jump produces a non-aligned target, execution must stop with an
instruction-address-misaligned error.

If another branch or jump executes inside a delay slot, that younger
control-transfer instruction replaces any older unresolved redirection. The
younger instruction then contributes its own 2 delay slots before control
redirects to its resolved target.

### Memory regions

- DRAM is byte-addressed backing storage
- IMEM is byte-addressed instruction memory
- VMEM is byte-addressed on-chip data memory
- IMEM base is `0x0010_0000` and IMEM size is `32 KiB`
- VMEM base is `0x0800_0000` and VMEM size is `1 MiB`
- DRAM base is `0x8000_0000` and DRAM size is `16 GiB`
- instruction fetch conceptually reads IMEM
- scalar `sld` and `sst` access VMEM only
- DMA is the architected path between DRAM and VMEM

Scalar loads and stores always transfer exactly one 32-bit word and require
`address % 4 == 0`.

If a scalar load or store uses an unaligned address, execution must stop with a
misaligned-access error. Penguin does not emulate unaligned accesses in hardware
or in the functional model.

## Mnemonic Naming Convention

### Directly prefixed RV32I mnemonics

The following mnemonics are formed by prepending `s` to the RV32I mnemonic:

- `slui`
- `sauipc`
- `sjal`
- `sjalr`
- `sbeq`
- `sbne`
- `sblt`
- `sbge`
- `sbltu`
- `sbgeu`
- `saddi`
- `sslti`
- `ssltiu`
- `sxori`
- `sori`
- `sandi`
- `sslli`
- `ssrli`
- `ssrai`
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
- `sfence`
- `secall`
- `sebreak`

### Memory mnemonics

The RV32I load/store family is collapsed into two Penguin mnemonics:

- `sld rd, imm(rs1)` for aligned 32-bit VMEM loads
- `sst rs2, imm(rs1)` for aligned 32-bit VMEM stores

There are no scalar mnemonics corresponding to `lb`, `lbu`, `lh`, `lhu`, `sb`,
`sh`, or `sw`. The only architecturally visible scalar memory element is one
32-bit word.

DMA uses channelized mnemonics:

- `dma.load.chN rs_dram, rs_vmem, rs_size`
- `dma.store.chN rs_dram, rs_vmem, rs_size`
- `dma.wait.chN`

The first baseline exposes 8 symmetric DMA channels, `ch0` through `ch7`.

## Instruction Semantics

All arithmetic is modulo 2^32 unless signed comparison semantics are stated
explicitly.

### U-type instructions

#### `slui rd, imm20`

Semantics:

`x[rd] <- imm20 << 12`

#### `sauipc rd, imm20`

Semantics:

`x[rd] <- pc + (imm20 << 12)`

### Control-transfer instructions

#### `sjal rd, imm`

Semantics:

- `x[rd] <- pc + 4`
- execute the 2 delay-slot instructions at `pc + 4` and `pc + 8`
- after those 2 delay-slot instructions retire, `pc <- pc + imm`
- if a younger control-transfer instruction executes in one of those delay
  slots, the younger control transfer replaces this pending redirection

#### `sjalr rd, rs1, imm`

Semantics:

- `target <- (x[rs1] + imm) & ~1`
- `x[rd] <- pc + 4`
- execute the 2 delay-slot instructions at `pc + 4` and `pc + 8`
- after those 2 delay-slot instructions retire, `pc <- target`
- if a younger control-transfer instruction executes in one of those delay
  slots, the younger control transfer replaces this pending redirection

### Conditional branches

#### `sbeq rs1, rs2, imm`

If `x[rs1] == x[rs2]`, then branch to `pc + imm` after 2 delay slots, else
continue sequentially after those same 2 delay slots.

If a younger control-transfer instruction executes in one of the delay slots, it
replaces this pending redirection.

#### `sbne rs1, rs2, imm`

If `x[rs1] != x[rs2]`, then branch to `pc + imm` after 2 delay slots, else
continue sequentially after those same 2 delay slots.

If a younger control-transfer instruction executes in one of the delay slots, it
replaces this pending redirection.

#### `sblt rs1, rs2, imm`

If `signed(x[rs1]) < signed(x[rs2])`, then branch to `pc + imm` after 2 delay
slots, else continue sequentially after those same 2 delay slots.

If a younger control-transfer instruction executes in one of the delay slots, it
replaces this pending redirection.

#### `sbge rs1, rs2, imm`

If `signed(x[rs1]) >= signed(x[rs2])`, then branch to `pc + imm` after 2 delay
slots, else continue sequentially after those same 2 delay slots.

If a younger control-transfer instruction executes in one of the delay slots, it
replaces this pending redirection.

#### `sbltu rs1, rs2, imm`

If `unsigned(x[rs1]) < unsigned(x[rs2])`, then branch to `pc + imm` after 2
delay slots, else continue sequentially after those same 2 delay slots.

If a younger control-transfer instruction executes in one of the delay slots, it
replaces this pending redirection.

#### `sbgeu rs1, rs2, imm`

If `unsigned(x[rs1]) >= unsigned(x[rs2])`, then branch to `pc + imm` after 2
delay slots, else continue sequentially after those same 2 delay slots.

If a younger control-transfer instruction executes in one of the delay slots, it
replaces this pending redirection.

### Scalar memory access

#### `sld rd, imm(rs1)`

Semantics:

`x[rd] <- VMEM32[x[rs1] + imm]`

Requirements:

- the effective address must be 4-byte aligned
- exactly 4 bytes are read
- the loaded value is written to `x[rd]`

#### `sst rs2, imm(rs1)`

Semantics:

`VMEM32[x[rs1] + imm] <- x[rs2]`

Requirements:

- the effective address must be 4-byte aligned
- exactly 4 bytes are written
- no partial-byte or partial-halfword stores exist

### DMA transfers

#### `dma.load.chN rs_dram, rs_vmem, rs_size`

Semantics:

- schedule a pending transfer of `x[rs_size]` raw bytes from DRAM at `x[rs_dram]`
  to VMEM at `x[rs_vmem]` on channel `N`
- `x[rs_dram]` and `x[rs_vmem]` must both be 32-byte aligned
- `x[rs_size]` must be a multiple of 32 bytes
- the operation is only legal if channel `N` is currently idle
- completion is not architecturally visible until `dma.wait.chN`

#### `dma.store.chN rs_dram, rs_vmem, rs_size`

Semantics:

- schedule a pending transfer of `x[rs_size]` raw bytes from VMEM at `x[rs_vmem]`
  to DRAM at `x[rs_dram]` on channel `N`
- `x[rs_dram]` and `x[rs_vmem]` must both be 32-byte aligned
- `x[rs_size]` must be a multiple of 32 bytes
- the operation is only legal if channel `N` is currently idle
- completion is not architecturally visible until `dma.wait.chN`

#### `dma.wait.chN`

Semantics:

- block until channel `N` has no pending DMA transfer
- once complete, the destination memory region reflects the transferred bytes
- if channel `N` is already idle at issue time, the instruction completes immediately

### I-type integer compute instructions

#### `saddi rd, rs1, imm`

`x[rd] <- x[rs1] + imm`

#### `sslti rd, rs1, imm`

`x[rd] <- 1 if signed(x[rs1]) < signed(imm) else 0`

#### `ssltiu rd, rs1, imm`

`x[rd] <- 1 if unsigned(x[rs1]) < unsigned(sign_extend(imm)) else 0`

#### `sxori rd, rs1, imm`

`x[rd] <- x[rs1] xor sign_extend(imm)`

#### `sori rd, rs1, imm`

`x[rd] <- x[rs1] or sign_extend(imm)`

#### `sandi rd, rs1, imm`

`x[rd] <- x[rs1] and sign_extend(imm)`

#### `sslli rd, rs1, shamt`

`x[rd] <- x[rs1] << (shamt & 0x1f)`

#### `ssrli rd, rs1, shamt`

`x[rd] <- unsigned(x[rs1]) >> (shamt & 0x1f)`

#### `ssrai rd, rs1, shamt`

`x[rd] <- signed(x[rs1]) >>> (shamt & 0x1f)`

### R-type integer compute instructions

#### `sadd rd, rs1, rs2`

`x[rd] <- x[rs1] + x[rs2]`

#### `ssub rd, rs1, rs2`

`x[rd] <- x[rs1] - x[rs2]`

#### `ssll rd, rs1, rs2`

`x[rd] <- x[rs1] << (x[rs2] & 0x1f)`

#### `sslt rd, rs1, rs2`

`x[rd] <- 1 if signed(x[rs1]) < signed(x[rs2]) else 0`

#### `ssltu rd, rs1, rs2`

`x[rd] <- 1 if unsigned(x[rs1]) < unsigned(x[rs2]) else 0`

#### `sxor rd, rs1, rs2`

`x[rd] <- x[rs1] xor x[rs2]`

#### `ssrl rd, rs1, rs2`

`x[rd] <- unsigned(x[rs1]) >> (x[rs2] & 0x1f)`

#### `ssra rd, rs1, rs2`

`x[rd] <- signed(x[rs1]) >>> (x[rs2] & 0x1f)`

#### `sor rd, rs1, rs2`

`x[rd] <- x[rs1] or x[rs2]`

#### `sand rd, rs1, rs2`

`x[rd] <- x[rs1] and x[rs2]`

### Ordering and environment instructions

#### `sfence`

Penguin currently models a single scalar execution context with no relaxed memory
system. `sfence` is therefore architecturally legal and functionally a no-op,
but it remains part of the ISA so the scalar subset stays structurally aligned
with RV32I.

#### `secall`

`secall` terminates execution with an environment-call halt status.

#### `sebreak`

`sebreak` terminates execution with a breakpoint halt status.

## Functional Model Contract

The scalar functional model must expose the following architecture-visible
behavior:

- a 32-entry integer register file with `x0` hardwired to zero
- a 32-bit program counter
- instruction retirement in program order
- 2 architecturally required delay slots for branches and jumps
- mnemonic dispatch using the scalar `s*` names defined in this document
- distinct DRAM, IMEM, and VMEM regions in the architectural state
- scalar data loads and stores bound to VMEM rather than DRAM
- DMA issue and wait behavior for DRAM <-> VMEM transfers
- explicit DMA failure on channel reuse while a channel is still busy
- explicit DMA failure on misaligned DMA address or size
- explicit instruction-address-misaligned failures for invalid branch and jump
  targets
- explicit misaligned-load and misaligned-store failures for `sld` and `sst`
- explicit termination status for `secall` and `sebreak`

This document does not define a binary encoding. The current contract is the
assembly-level mnemonic and operand behavior.

## Non-Normative Functional Model Implementation Plan

### Phase 1: architectural-state refactor

- add `pc` to the architectural state
- add explicit execution-stop status for normal halt, breakpoint, environment
  call, misaligned load/store, and instruction-address misalignment
- treat the integer register file as the scalar architectural register file
- add pending-target and delay-slot countdown state for control-transfer instructions
- stop extending the scalar path through floating-point state

### Phase 2: decoded instruction representation

- extend the instruction parameter types to cover the RV32I-derived scalar forms
  needed for U-type, B-type, and J-type instructions
- register all scalar mnemonics using the `s*` naming convention from this spec
- map `sld` and `sst` directly rather than carrying legacy `lw` or `sw`

### Phase 3: semantic implementation order

- implement `slui`, `sauipc`, `saddi`, logic-immediate, and shift-immediate ops
  first because they do not require control-flow changes
- implement R-type integer ALU ops next
- implement `sld` and `sst` with strict 4-byte alignment checks
- implement branches, `sjal`, and `sjalr` once `pc` sequencing plus 2-delay-slot
  handling are in place
- implement `sfence`, `secall`, and `sebreak` last

### Phase 4: verification

- replace the old float-oriented smoke tests with integer-only scalar tests
- add one test per instruction family plus dedicated misaligned-access tests
- add short branch and jump programs that verify `pc` updates and link-register
  writes
- add one end-to-end hand-written scalar program that exercises arithmetic,
  control flow, and aligned memory traffic together

### Phase 5: cleanup

- remove or quarantine the legacy float-only demo path once the scalar integer
  path is stable
- update higher-level examples to use scalar integer assembly names from this
  spec
