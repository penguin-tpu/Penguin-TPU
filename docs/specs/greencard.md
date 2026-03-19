# Penguin-TPU Green Card

Conventions used in the opcode tables:

- Bit positions use the RISC-V convention, with bit `31` the MSB and bit `0`
  the LSB.
- `Hex value` is written as `opcode_hex/funct3_hex/funct7_or_imm_hex` when that
  decomposition exists.
- `*` in the `Hex value` column means the field is operand-dependent rather than
  part of the opcode match.
- `pending` means the instruction family is architecturally allocated and its
  assembly-visible behavior is frozen, but the exact bit packing is not yet
  frozen in the encoding supplement.
- Verilog-style descriptions use architectural state names such as `x[...]`,
  `m[...]`, `e[...]`, `pc`, `VMEM[...]`, and `mxuN.acc`.

## 1. Register Set

The baseline scalar procedure-call convention follows standard `RV32I` ABI
practice for the `x` register file. No procedure-call ABI has been frozen yet
for `m`, `e`, or MXU-local state; this card uses the conservative convention
`Caller` for those architected state elements.

| Register | ABI Name | Description | Saver |
|---|---|---|---|
| `pc` | `pc` | Architectural program counter, stored as an instruction-word index | N/A |
| `x0` | `zero` | Constant zero | Fixed |
| `x1` | `ra` | Return address | Caller |
| `x2` | `sp` | Stack pointer | Callee |
| `x3` | `gp` | Global pointer | Fixed |
| `x4` | `tp` | Thread pointer | Fixed |
| `x5` | `t0` | Temporary register 0 | Caller |
| `x6` | `t1` | Temporary register 1 | Caller |
| `x7` | `t2` | Temporary register 2 | Caller |
| `x8` | `s0/fp` | Saved register 0 / frame pointer | Callee |
| `x9` | `s1` | Saved register 1 | Callee |
| `x10` | `a0` | Argument / return value 0 | Caller |
| `x11` | `a1` | Argument / return value 1 | Caller |
| `x12` | `a2` | Argument 2 | Caller |
| `x13` | `a3` | Argument 3 | Caller |
| `x14` | `a4` | Argument 4 | Caller |
| `x15` | `a5` | Argument 5 | Caller |
| `x16` | `a6` | Argument 6 | Caller |
| `x17` | `a7` | Argument 7 | Caller |
| `x18` | `s2` | Saved register 2 | Callee |
| `x19` | `s3` | Saved register 3 | Callee |
| `x20` | `s4` | Saved register 4 | Callee |
| `x21` | `s5` | Saved register 5 | Callee |
| `x22` | `s6` | Saved register 6 | Callee |
| `x23` | `s7` | Saved register 7 | Callee |
| `x24` | `s8` | Saved register 8 | Callee |
| `x25` | `s9` | Saved register 9 | Callee |
| `x26` | `s10` | Saved register 10 | Callee |
| `x27` | `s11` | Saved register 11 | Callee |
| `x28` | `t3` | Temporary register 3 | Caller |
| `x29` | `t4` | Temporary register 4 | Caller |
| `x30` | `t5` | Temporary register 5 | Caller |
| `x31` | `t6` | Temporary register 6 | Caller |
| `m0-m63` | `-` | Flat tensor register file, `4096` bytes per register | Caller |
| `e0-e31` | `-` | Whole-tensor scale register file, one `FP8_E8M0` exponent per register | Caller |
| `mxu0.w0` | `-` | MXU0 lodal FP8 weight slot 0 | Caller |
| `mxu0.w1` | `-` | MXU0 lodal FP8 weight slot 1 | Caller |
| `mxu1.w0` | `-` | MXU1 lodal FP8 weight slot 0 | Caller |
| `mxu1.w1` | `-` | MXU1 lodal FP8 weight slot 1 | Caller |
| `mxu0.acc` | `-` | MXU0 local `64 x 64 BF16` accumulation buffer | Caller |
| `mxu1.acc` | `-` | MXU1 local `64 x 64 BF16` accumulation buffer | Caller |

## 2. Core Instruction Formats

### 2.1 Scalar RV32I-Compatible Formats

| Format | Bits | Width | Field | Notes |
|---|---|---:|---|---|
| `R` | `[31:25]` | 7 | `funct7` | ALU sub-op selector |
| `R` | `[24:20]` | 5 | `rs2` | Source register 2 |
| `R` | `[19:15]` | 5 | `rs1` | Source register 1 |
| `R` | `[14:12]` | 3 | `funct3` | ALU sub-op selector |
| `R` | `[11:7]` | 5 | `rd` | Destination register |
| `R` | `[6:0]` | 7 | `opcode` | Major opcode |
| `I` | `[31:20]` | 12 | `imm[11:0]` | Sign-extended immediate |
| `I` | `[19:15]` | 5 | `rs1` | Source register 1 |
| `I` | `[14:12]` | 3 | `funct3` | Sub-op selector |
| `I` | `[11:7]` | 5 | `rd` | Destination register |
| `I` | `[6:0]` | 7 | `opcode` | Major opcode |
| `S` | `[31:25]` | 7 | `imm[11:5]` | Store immediate upper bits |
| `S` | `[24:20]` | 5 | `rs2` | Store data register |
| `S` | `[19:15]` | 5 | `rs1` | Base register |
| `S` | `[14:12]` | 3 | `funct3` | Width selector |
| `S` | `[11:7]` | 5 | `imm[4:0]` | Store immediate lower bits |
| `S` | `[6:0]` | 7 | `opcode` | Major opcode |
| `B` | `[31]` | 1 | `imm[12]` | Branch offset sign bit |
| `B` | `[30:25]` | 6 | `imm[10:5]` | Branch offset bits |
| `B` | `[24:20]` | 5 | `rs2` | Branch compare source 2 |
| `B` | `[19:15]` | 5 | `rs1` | Branch compare source 1 |
| `B` | `[14:12]` | 3 | `funct3` | Branch condition selector |
| `B` | `[11:8]` | 4 | `imm[4:1]` | Branch offset bits |
| `B` | `[7]` | 1 | `imm[11]` | Branch offset bit |
| `B` | `[6:0]` | 7 | `opcode` | Major opcode |
| `U` | `[31:12]` | 20 | `imm[31:12]` | Upper immediate |
| `U` | `[11:7]` | 5 | `rd` | Destination register |
| `U` | `[6:0]` | 7 | `opcode` | Major opcode |
| `J` | `[31]` | 1 | `imm[20]` | Jump offset sign bit |
| `J` | `[30:21]` | 10 | `imm[10:1]` | Jump offset bits |
| `J` | `[20]` | 1 | `imm[11]` | Jump offset bit |
| `J` | `[19:12]` | 8 | `imm[19:12]` | Jump offset bits |
| `J` | `[11:7]` | 5 | `rd` | Destination register |
| `J` | `[6:0]` | 7 | `opcode` | Major opcode |

### 2.2 System and Delay Format

`fence`, `ecall`, and `ebreak` use standard `RV32I` encodings. `delay` reuses
the `SYSTEM` major opcode.

| Format | Bits | Width | Field | Notes |
|---|---|---:|---|---|
| `SYSTEM-I` | `[31:20]` | 12 | `imm12` | `delay` cycle count or system immediate |
| `SYSTEM-I` | `[19:15]` | 5 | `rs1` | `x0` for `delay`, `ecall`, `ebreak` |
| `SYSTEM-I` | `[14:12]` | 3 | `funct3` | `000` for `delay`, `ecall`, `ebreak` |
| `SYSTEM-I` | `[11:7]` | 5 | `rd` | `x0` for `delay`, `ecall`, `ebreak` |
| `SYSTEM-I` | `[6:0]` | 7 | `opcode` | `1110011` |

### 2.3 Accelerator Register-Field Conventions

These conventions apply whenever a custom format names tensor or scale
registers in RISC-V-like field positions.

| Bits | Width | Field | Meaning |
|---|---:|---|---|
| `[11:7]` | 5 | `rd` | Low bits of `vd` or standard `x/e` destination |
| `[19:15]` | 5 | `rs1` | Low bits of `vs1` or standard `x/e` source |
| `[24:20]` | 5 | `rs2` | Low bits of `vs2` or standard `x` source |
| `[14]` | 1 | `vs2_hi` | High bit of `m` register source 2 when used |
| `[13]` | 1 | `vs1_hi` | High bit of `m` register source 1 when used |
| `[12]` | 1 | `vd_hi` | High bit of `m` register destination when used |

Tensor register reconstruction:

- `vd = {bit[12], bit[11:7]}`
- `vs1 = {bit[13], bit[19:15]}`
- `vs2 = {bit[14], bit[24:20]}`

Scaled VMEM tensor-transfer immediate convention:

- `imm12_32` is a signed `12`-bit immediate in units of `32` bytes
- effective byte offset = `sign_extend(imm12_32) << 5`

### 2.4 Frozen Custom Whole-Register Formats

#### `TVEC-R` (`custom-2`, VPU)

| Bits | Width | Field | Notes |
|---|---|---:|---|---|
| `[31:27]` | 5 | `vec_fn` | VPU operation selector |
| `[26:25]` | 2 | `0` | Reserved, shall be zero |
| `[24:20]` | 5 | `vs2_lo` | `m` source 2 low bits; zero for unary ops |
| `[19:15]` | 5 | `vs1_lo` | `m` source 1 low bits |
| `[14]` | 1 | `vs2_hi` | `m` source 2 high bit; zero for unary ops |
| `[13]` | 1 | `vs1_hi` | `m` source 1 high bit |
| `[12]` | 1 | `vd_hi` | `m` destination high bit |
| `[11:7]` | 5 | `vd_lo` | `m` destination low bits |
| `[6:0]` | 7 | `opcode` | `1011011` (`custom-2`) |

#### `TXLU-R` (`custom-3`, XLU)

| Bits | Width | Field | Notes |
|---|---|---:|---|---|
| `[31:27]` | 5 | `xlu_op` | XLU operation selector |
| `[26:25]` | 2 | `0` | Reserved, shall be zero |
| `[24:20]` | 5 | `0` | Reserved, shall be zero |
| `[19:15]` | 5 | `vs1_lo` | `m` source low bits |
| `[14]` | 1 | `0` | Reserved, shall be zero |
| `[13]` | 1 | `vs1_hi` | `m` source high bit |
| `[12]` | 1 | `vd_hi` | `m` destination high bit |
| `[11:7]` | 5 | `vd_lo` | `m` destination low bits |
| `[6:0]` | 7 | `opcode` | `1111011` (`custom-3`) |

### 2.5 Custom Families With Encoding Supplement Pending

The following families are architecturally allocated, but the exact per-bit
field packing is not yet frozen in the architecture specification:

| Family | Major Opcode | Frozen Assembly Contract |
|---|---|---|
| `custom-0` | `0001011` | `seli`, `seld`, `dma.*`, `vload`, `vstore`, `vmatpush.*`, `vload.weight.*`, `vmatpop.*` |
| `custom-1` | `0101011` | `vmatmul.*`, `vmatpush.bf16.acc.*` |

Software may rely on the mnemonic repertoire and architectural semantics of
those instruction families, but not yet on a frozen bit-exact subformat.

## 3. Instructions

### 3.1 Scalar Core Instruction Set

| Mnemonic | Fmt | Opcode Binary | `funct3` Binary | `funct7` or `imm` Binary | Hex Value | Name | Description (in Verilog) |
|---|---|---|---|---|---|---|---|
| `lui` | `U` | `0110111` | `---` | `imm[31:12]` | `37` | Load Upper Immediate | `x[rd] = {imm[31:12], 12'b0}` |
| `auipc` | `U` | `0010111` | `---` | `imm[31:12]` | `17` | Add Upper Immediate to PC | `x[rd] = pc + {imm[31:12], 12'b0}` |
| `jal` | `J` | `1101111` | `---` | `imm[20|10:1|11|19:12]` | `6F` | Jump And Link | `x[rd] = pc + 1; pc = pc + imm after 2 delay slots` |
| `jalr` | `I` | `1100111` | `000` | `imm[11:0]` | `67/0/*` | Jump And Link Register | `tmp = x[rs1] + imm; x[rd] = pc + 1; pc = tmp after 2 delay slots` |
| `beq` | `B` | `1100011` | `000` | `imm[12|10:5|4:1|11]` | `63/0/*` | Branch Equal | `if (x[rs1] == x[rs2]) pc = pc + imm after 2 delay slots` |
| `bne` | `B` | `1100011` | `001` | `imm[12|10:5|4:1|11]` | `63/1/*` | Branch Not Equal | `if (x[rs1] != x[rs2]) pc = pc + imm after 2 delay slots` |
| `blt` | `B` | `1100011` | `100` | `imm[12|10:5|4:1|11]` | `63/4/*` | Branch Less Than | `if ($signed(x[rs1]) < $signed(x[rs2])) pc = pc + imm after 2 delay slots` |
| `bge` | `B` | `1100011` | `101` | `imm[12|10:5|4:1|11]` | `63/5/*` | Branch Greater Or Equal | `if ($signed(x[rs1]) >= $signed(x[rs2])) pc = pc + imm after 2 delay slots` |
| `bltu` | `B` | `1100011` | `110` | `imm[12|10:5|4:1|11]` | `63/6/*` | Branch Less Than Unsigned | `if (x[rs1] < x[rs2]) pc = pc + imm after 2 delay slots` |
| `bgeu` | `B` | `1100011` | `111` | `imm[12|10:5|4:1|11]` | `63/7/*` | Branch Greater Or Equal Unsigned | `if (x[rs1] >= x[rs2]) pc = pc + imm after 2 delay slots` |
| `lb` | `I` | `0000011` | `000` | `imm[11:0]` | `03/0/*` | Load Byte | `x[rd] = {{24{VMEM8[x[rs1] + imm][7]}}, VMEM8[x[rs1] + imm]}` |
| `lh` | `I` | `0000011` | `001` | `imm[11:0]` | `03/1/*` | Load Halfword | `x[rd] = {{16{VMEM16[x[rs1] + imm][15]}}, VMEM16[x[rs1] + imm]}` |
| `lw` | `I` | `0000011` | `010` | `imm[11:0]` | `03/2/*` | Load Word | `x[rd] = VMEM32[x[rs1] + imm]` |
| `lbu` | `I` | `0000011` | `100` | `imm[11:0]` | `03/4/*` | Load Byte Unsigned | `x[rd] = {24'b0, VMEM8[x[rs1] + imm]}` |
| `lhu` | `I` | `0000011` | `101` | `imm[11:0]` | `03/5/*` | Load Halfword Unsigned | `x[rd] = {16'b0, VMEM16[x[rs1] + imm]}` |
| `sb` | `S` | `0100011` | `000` | `imm[11:5|4:0]` | `23/0/*` | Store Byte | `VMEM8[x[rs1] + imm] = x[rs2][7:0]` |
| `sh` | `S` | `0100011` | `001` | `imm[11:5|4:0]` | `23/1/*` | Store Halfword | `VMEM16[x[rs1] + imm] = x[rs2][15:0]` |
| `sw` | `S` | `0100011` | `010` | `imm[11:5|4:0]` | `23/2/*` | Store Word | `VMEM32[x[rs1] + imm] = x[rs2]` |
| `addi` | `I` | `0010011` | `000` | `imm[11:0]` | `13/0/*` | Add Immediate | `x[rd] = x[rs1] + imm` |
| `slti` | `I` | `0010011` | `010` | `imm[11:0]` | `13/2/*` | Set Less Than Immediate | `x[rd] = ($signed(x[rs1]) < $signed(imm))` |
| `sltiu` | `I` | `0010011` | `011` | `imm[11:0]` | `13/3/*` | Set Less Than Immediate Unsigned | `x[rd] = (x[rs1] < $unsigned($signed(imm)))` |
| `xori` | `I` | `0010011` | `100` | `imm[11:0]` | `13/4/*` | XOR Immediate | `x[rd] = x[rs1] ^ imm` |
| `ori` | `I` | `0010011` | `110` | `imm[11:0]` | `13/6/*` | OR Immediate | `x[rd] = x[rs1] | imm` |
| `andi` | `I` | `0010011` | `111` | `imm[11:0]` | `13/7/*` | AND Immediate | `x[rd] = x[rs1] & imm` |
| `slli` | `I` | `0010011` | `001` | `0000000_shamt` | `13/1/00` | Shift Left Logical Immediate | `x[rd] = x[rs1] << shamt` |
| `srli` | `I` | `0010011` | `101` | `0000000_shamt` | `13/5/00` | Shift Right Logical Immediate | `x[rd] = x[rs1] >> shamt` |
| `srai` | `I` | `0010011` | `101` | `0100000_shamt` | `13/5/20` | Shift Right Arithmetic Immediate | `x[rd] = $signed(x[rs1]) >>> shamt` |
| `add` | `R` | `0110011` | `000` | `0000000` | `33/0/00` | Add | `x[rd] = x[rs1] + x[rs2]` |
| `sub` | `R` | `0110011` | `000` | `0100000` | `33/0/20` | Subtract | `x[rd] = x[rs1] - x[rs2]` |
| `sll` | `R` | `0110011` | `001` | `0000000` | `33/1/00` | Shift Left Logical | `x[rd] = x[rs1] << x[rs2][4:0]` |
| `slt` | `R` | `0110011` | `010` | `0000000` | `33/2/00` | Set Less Than | `x[rd] = ($signed(x[rs1]) < $signed(x[rs2]))` |
| `sltu` | `R` | `0110011` | `011` | `0000000` | `33/3/00` | Set Less Than Unsigned | `x[rd] = (x[rs1] < x[rs2])` |
| `xor` | `R` | `0110011` | `100` | `0000000` | `33/4/00` | XOR | `x[rd] = x[rs1] ^ x[rs2]` |
| `srl` | `R` | `0110011` | `101` | `0000000` | `33/5/00` | Shift Right Logical | `x[rd] = x[rs1] >> x[rs2][4:0]` |
| `sra` | `R` | `0110011` | `101` | `0100000` | `33/5/20` | Shift Right Arithmetic | `x[rd] = $signed(x[rs1]) >>> x[rs2][4:0]` |
| `or` | `R` | `0110011` | `110` | `0000000` | `33/6/00` | OR | `x[rd] = x[rs1] | x[rs2]` |
| `and` | `R` | `0110011` | `111` | `0000000` | `33/7/00` | AND | `x[rd] = x[rs1] & x[rs2]` |
| `fence` | `I` | `0001111` | `000` | `000000000000` | `0F/0/000` | Fence | `/* architecturally legal baseline no-op */` |
| `delay N` | `SYSTEM-I` | `1110011` | `000` | `N[11:0]` | `73/0/*` | Frontend Delay | `hold_decode_issue_for_N_cycles();` |
| `ecall` | `SYSTEM-I` | `1110011` | `000` | `000000000000` | `73/0/000` | Environment Call | `halt_reason = ECALL; halt = 1'b1;` |
| `ebreak` | `SYSTEM-I` | `1110011` | `000` | `000000000001` | `73/0/001` | Breakpoint | `halt_reason = EBREAK; halt = 1'b1;` |

### 3.2 Scale and Tensor-Movement Families (`custom-0`)

The exact `custom-0` bit packing is still pending. The rows below document the
frozen architectural forms and major-opcode allocation.

| Mnemonic | Fmt | Opcode Binary | `funct3` Binary | `funct7` or `imm` Binary | Hex Value | Name | Description (in Verilog) |
|---|---|---|---|---|---|---|---|
| `seli eD, imm8` | pending | `0001011` | pending | pending | `0B/*/*` | Scale Immediate | `e[D] = imm8;` |
| `seld eD, imm(xR)` | pending | `0001011` | pending | pending | `0B/*/*` | Scale Load | `e[D] = VMEM8[x[R] + imm];` |
| `dma.load.chN xD, xV, xS` | pending | `0001011` | pending | pending | `0B/*/*` | DMA Load | `issue_dma_load(channel=N, dram_addr=x[D], vmem_addr=x[V], size=x[S]);` |
| `dma.store.chN xD, xV, xS` | pending | `0001011` | pending | pending | `0B/*/*` | DMA Store | `issue_dma_store(channel=N, dram_addr=x[D], vmem_addr=x[V], size=x[S]);` |
| `dma.wait.chN` | pending | `0001011` | pending | pending | `0B/*/*` | DMA Wait | `wait_until_dma_channel_idle(channel=N);` |
| `vload mD, imm(xR)` | pending | `0001011` | pending | pending | `0B/*/*` | Tensor Load | `m[D] = VMEM_tensor[x[R] + (imm12_32 << 5)];` |
| `vstore mS, imm(xR)` | pending | `0001011` | pending | pending | `0B/*/*` | Tensor Store | `VMEM_tensor[x[R] + (imm12_32 << 5)] = m[S];` |
| `vmatpush.mxu0 wD, mS` | pending | `0001011` | pending | pending | `0B/*/*` | Push Tensor To MXU0 Weight Slot | `mxu0.w[D] = m[S];` |
| `vmatpush.mxu1 wD, mS` | pending | `0001011` | pending | pending | `0B/*/*` | Push Tensor To MXU1 Weight Slot | `mxu1.w[D] = m[S];` |
| `vload.weight.mxu0 wD, xR` | pending | `0001011` | pending | pending | `0B/*/*` | Load MXU0 Weight From VMEM | `mxu0.w[D] = VMEM_tensor[x[R]];` |
| `vload.weight.mxu1 wD, xR` | pending | `0001011` | pending | pending | `0B/*/*` | Load MXU1 Weight From VMEM | `mxu1.w[D] = VMEM_tensor[x[R]];` |
| `vmatpop.bf16.acc.mxu0 mD` | pending | `0001011` | pending | pending | `0B/*/*` | Pop MXU0 BF16 Accumulator | `{m[D], m[D + 1]} = mxu0.acc;` |
| `vmatpop.bf16.acc.mxu1 mD` | pending | `0001011` | pending | pending | `0B/*/*` | Pop MXU1 BF16 Accumulator | `{m[D], m[D + 1]} = mxu1.acc;` |
| `vmatpop.fp8.acc.mxu0 mD` | pending | `0001011` | pending | pending | `0B/*/*` | Pop MXU0 FP8 Accumulator View | `m[D] = quantize_fp8(mxu0.acc);` |
| `vmatpop.fp8.acc.mxu1 mD` | pending | `0001011` | pending | pending | `0B/*/*` | Pop MXU1 FP8 Accumulator View | `m[D] = quantize_fp8(mxu1.acc);` |

### 3.3 MXU Launch Families (`custom-1`)

The exact `custom-1` bit packing is still pending. The rows below document the
frozen architectural forms and major-opcode allocation.

| Mnemonic | Fmt | Opcode Binary | `funct3` Binary | `funct7` or `imm` Binary | Hex Value | Name | Description (in Verilog) |
|---|---|---|---|---|---|---|---|
| `vmatpush.bf16.acc.mxu0 mS` | pending | `0101011` | pending | pending | `2B/*/*` | Push BF16 Tile To MXU0 Accumulator | `mxu0.acc = {m[S], m[S + 1]};` |
| `vmatpush.bf16.acc.mxu1 mS` | pending | `0101011` | pending | pending | `2B/*/*` | Push BF16 Tile To MXU1 Accumulator | `mxu1.acc = {m[S], m[S + 1]};` |
| `vmatmul.mxu0 mS, wW` | pending | `0101011` | pending | pending | `2B/*/*` | MXU0 Matmul | `mxu0.acc = matmul_fp8_to_bf16(m[S], mxu0.w[W]);` |
| `vmatmul.acc.mxu0 mS, wW` | pending | `0101011` | pending | pending | `2B/*/*` | MXU0 Matmul Accumulate | `mxu0.acc = mxu0.acc + matmul_fp8_to_bf16(m[S], mxu0.w[W]);` |
| `vmatmul.mxu1 mS, wW` | pending | `0101011` | pending | pending | `2B/*/*` | MXU1 Matmul | `mxu1.acc = matmul_fp8_to_bf16(m[S], mxu1.w[W]);` |
| `vmatmul.acc.mxu1 mS, wW` | pending | `0101011` | pending | pending | `2B/*/*` | MXU1 Matmul Accumulate | `mxu1.acc = mxu1.acc + matmul_fp8_to_bf16(m[S], mxu1.w[W]);` |

### 3.4 VPU Whole-Register Instructions (`custom-2`, `TVEC-R`)

In `TVEC-R`, bits `[14:12]` carry `{vs2_hi, vs1_hi, vd_hi}` rather than a
fixed `funct3` sub-op selector. The VPU operation selector lives in
`vec_fn = bits[31:27]`.

| Mnemonic | Fmt | Opcode Binary | `funct3` Binary | `funct7` or `imm` Binary | Hex Value | Name | Description (in Verilog) |
|---|---|---|---|---|---|---|---|
| `vadd mD, mS1, mS2` | `TVEC-R` | `1011011` | `{vs2_hi,vs1_hi,vd_hi}` | `00000_00` | `5B/*/00` | Vector Add | `m[D] = bf16_add(m[S1], m[S2]);` |
| `vsub mD, mS1, mS2` | `TVEC-R` | `1011011` | `{vs2_hi,vs1_hi,vd_hi}` | `00001_00` | `5B/*/04` | Vector Subtract | `m[D] = bf16_sub(m[S1], m[S2]);` |
| `vmul mD, mS1, mS2` | `TVEC-R` | `1011011` | `{vs2_hi,vs1_hi,vd_hi}` | `00010_00` | `5B/*/08` | Vector Multiply | `m[D] = bf16_mul(m[S1], m[S2]);` |
| `vmax mD, mS1, mS2` | `TVEC-R` | `1011011` | `{vs2_hi,vs1_hi,vd_hi}` | `00011_00` | `5B/*/0C` | Vector Maximum | `m[D] = bf16_max(m[S1], m[S2]);` |
| `vmin mD, mS1, mS2` | `TVEC-R` | `1011011` | `{vs2_hi,vs1_hi,vd_hi}` | `00100_00` | `5B/*/10` | Vector Minimum | `m[D] = bf16_min(m[S1], m[S2]);` |
| `vmov mD, mS1` | `TVEC-R` | `1011011` | `{1'b0,vs1_hi,vd_hi}` | `10000_00` | `5B/*/40` | Vector Move | `m[D] = m[S1];` |
| `vrelu mD, mS1` | `TVEC-R` | `1011011` | `{1'b0,vs1_hi,vd_hi}` | `10001_00` | `5B/*/44` | Vector ReLU | `m[D] = bf16_relu(m[S1]);` |
| `vexp mD, mS1` | `TVEC-R` | `1011011` | `{1'b0,vs1_hi,vd_hi}` | `10010_00` | `5B/*/48` | Vector Exponential | `m[D] = bf16_exp(m[S1]);` |
| `vrecip mD, mS1` | `TVEC-R` | `1011011` | `{1'b0,vs1_hi,vd_hi}` | `10011_00` | `5B/*/4C` | Vector Reciprocal | `m[D] = bf16_recip(m[S1]);` |

### 3.5 XLU Whole-Register Instructions (`custom-3`, `TXLU-R`)

In `TXLU-R`, bits `[14:12]` carry `{1'b0, vs1_hi, vd_hi}`. The XLU operation
selector lives in `xlu_op = bits[31:27]`.

| Mnemonic | Fmt | Opcode Binary | `funct3` Binary | `funct7` or `imm` Binary | Hex Value | Name | Description (in Verilog) |
|---|---|---|---|---|---|---|---|
| `transpose.xlu mD, mS1` | `TXLU-R` | `1111011` | `{1'b0,vs1_hi,vd_hi}` | `00000_00` | `7B/*/00` | Matrix Transpose | `m[D] = bf16_transpose(m[S1]);` |
| `reduce.max.xlu mD, mS1` | `TXLU-R` | `1111011` | `{1'b0,vs1_hi,vd_hi}` | `00001_00` | `7B/*/04` | Row Reduce Maximum | `m[D] = bf16_row_reduce_max(m[S1]);` |
| `reduce.sum.xlu mD, mS1` | `TXLU-R` | `1111011` | `{1'b0,vs1_hi,vd_hi}` | `00010_00` | `7B/*/08` | Row Reduce Sum | `m[D] = bf16_row_reduce_sum(m[S1]);` |

## 4. Architectural Design Parameters

| Item | Value |
|---|---:|
| Instruction width | `32` bits |
| Instruction alignment | `4` bytes |
| Control-flow delay slots | `2` |
| Scalar registers | `32` |
| Tensor registers | `64` |
| Scale registers | `32` |
| Tensor register storage | `64 rows x 64 bytes = 4096 bytes` |
| MXU count | `2` |
| MXU weight slots per MXU | `2` |
| MXU accumulator storage | `64 x 64 BF16` |
| DMA channels | `8` |
| VPU major opcode | `custom-2 = 1011011` |
| XLU major opcode | `custom-3 = 1111011` |
| `IMEM` base | `0x0002_0000` |
| `VMEM` base | `0x2000_0000` |
| `DRAM` base | `0x8000_0000` |
