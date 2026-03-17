`ifndef PENGUIN_SCALAR_DEFS_VH
`define PENGUIN_SCALAR_DEFS_VH

// Scalar major opcodes
`define PENGUIN_OPCODE_LOAD        7'b0000011
`define PENGUIN_OPCODE_MISC_MEM    7'b0001111
`define PENGUIN_OPCODE_OP_IMM      7'b0010011
`define PENGUIN_OPCODE_AUIPC       7'b0010111
`define PENGUIN_OPCODE_STORE       7'b0100011
`define PENGUIN_OPCODE_OP          7'b0110011
`define PENGUIN_OPCODE_LUI         7'b0110111
`define PENGUIN_OPCODE_BRANCH      7'b1100011
`define PENGUIN_OPCODE_JALR        7'b1100111
`define PENGUIN_OPCODE_JAL         7'b1101111
`define PENGUIN_OPCODE_SYSTEM      7'b1110011
`define PENGUIN_OPCODE_CUSTOM_0    7'b0001011
`define PENGUIN_OPCODE_CUSTOM_1    7'b0101011
`define PENGUIN_OPCODE_CUSTOM_2    7'b1011011
`define PENGUIN_OPCODE_CUSTOM_3    7'b1111011

// Common funct3 values
`define PENGUIN_FUNCT3_000         3'b000
`define PENGUIN_FUNCT3_001         3'b001
`define PENGUIN_FUNCT3_010         3'b010
`define PENGUIN_FUNCT3_011         3'b011
`define PENGUIN_FUNCT3_100         3'b100
`define PENGUIN_FUNCT3_101         3'b101
`define PENGUIN_FUNCT3_110         3'b110
`define PENGUIN_FUNCT3_111         3'b111

// Common funct7 values
`define PENGUIN_FUNCT7_0000000     7'b0000000
`define PENGUIN_FUNCT7_0100000     7'b0100000

// System immediates
`define PENGUIN_SYSTEM_ECALL       12'h000
`define PENGUIN_SYSTEM_EBREAK      12'h001

// Format classes
`define PENGUIN_FMT_R              4'd0
`define PENGUIN_FMT_I              4'd1
`define PENGUIN_FMT_S              4'd2
`define PENGUIN_FMT_B              4'd3
`define PENGUIN_FMT_U              4'd4
`define PENGUIN_FMT_J              4'd5
`define PENGUIN_FMT_SYSTEM         4'd6
`define PENGUIN_FMT_MISC_MEM       4'd7
`define PENGUIN_FMT_RESERVED       4'd8

// Scalar op classes
`define PENGUIN_SCALAR_OP_ALU_REG   4'd0
`define PENGUIN_SCALAR_OP_ALU_IMM   4'd1
`define PENGUIN_SCALAR_OP_LOAD      4'd2
`define PENGUIN_SCALAR_OP_STORE     4'd3
`define PENGUIN_SCALAR_OP_BRANCH    4'd4
`define PENGUIN_SCALAR_OP_JUMP      4'd5
`define PENGUIN_SCALAR_OP_UPPER_IMM 4'd6
`define PENGUIN_SCALAR_OP_SYSTEM    4'd7

// ALU / compare functions
`define PENGUIN_ALU_ADD            5'd0
`define PENGUIN_ALU_SUB            5'd1
`define PENGUIN_ALU_SLT            5'd2
`define PENGUIN_ALU_SLTU           5'd3
`define PENGUIN_ALU_XOR            5'd4
`define PENGUIN_ALU_OR             5'd5
`define PENGUIN_ALU_AND            5'd6
`define PENGUIN_ALU_SLL            5'd7
`define PENGUIN_ALU_SRL            5'd8
`define PENGUIN_ALU_SRA            5'd9
`define PENGUIN_ALU_COPY_B         5'd10
`define PENGUIN_ALU_COMPARE_EQ     5'd11
`define PENGUIN_ALU_COMPARE_NE     5'd12
`define PENGUIN_ALU_COMPARE_LT     5'd13
`define PENGUIN_ALU_COMPARE_GE     5'd14
`define PENGUIN_ALU_COMPARE_LTU    5'd15
`define PENGUIN_ALU_COMPARE_GEU    5'd16

// Halt reasons
`define PENGUIN_HALT_NONE               4'd0
`define PENGUIN_HALT_ILLEGAL_INSN       4'd1
`define PENGUIN_HALT_INSN_MISALIGNED    4'd2
`define PENGUIN_HALT_LOAD_MISALIGNED    4'd3
`define PENGUIN_HALT_STORE_MISALIGNED   4'd4
`define PENGUIN_HALT_ECALL              4'd5
`define PENGUIN_HALT_EBREAK             4'd6

// Preliminary VPU custom encodings
`define PENGUIN_VPU_FUNCT7_BINARY       7'b0000000
`define PENGUIN_VPU_FUNCT3_VADD         3'b000

`endif
