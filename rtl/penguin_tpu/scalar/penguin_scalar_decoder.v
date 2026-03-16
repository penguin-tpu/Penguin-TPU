`timescale 1ns / 1ps

`include "penguin_scalar_defs.vh"

module penguin_scalar_decoder (
    input  wire [31:0] instruction_word,
    output reg         valid,
    output reg         illegal,
    output reg  [3:0]  format_class,
    output reg  [3:0]  scalar_op_class,
    output reg  [4:0]  alu_fn,
    output reg  [4:0]  rd,
    output reg  [4:0]  rs1,
    output reg  [4:0]  rs2,
    output reg  [31:0] imm32,
    output reg         writes_rd,
    output reg         reads_rs1,
    output reg         reads_rs2,
    output reg         is_branch,
    output reg         is_jump,
    output reg         is_load,
    output reg         is_store,
    output reg         is_fence,
    output reg         is_ecall,
    output reg         is_ebreak,
    output reg         is_reserved_custom
);

    wire [6:0] opcode = instruction_word[6:0];
    wire [4:0] field_rd = instruction_word[11:7];
    wire [2:0] funct3 = instruction_word[14:12];
    wire [4:0] field_rs1 = instruction_word[19:15];
    wire [4:0] field_rs2 = instruction_word[24:20];
    wire [6:0] funct7 = instruction_word[31:25];
    wire [11:0] system_imm = instruction_word[31:20];

    wire [31:0] i_imm = {{20{instruction_word[31]}}, instruction_word[31:20]};
    wire [31:0] s_imm = {{20{instruction_word[31]}}, instruction_word[31:25], instruction_word[11:7]};
    wire [31:0] b_imm = {
        {19{instruction_word[31]}},
        instruction_word[31],
        instruction_word[7],
        instruction_word[30:25],
        instruction_word[11:8],
        1'b0
    };
    wire [31:0] u_imm = {instruction_word[31:12], 12'b0};
    wire [31:0] j_imm = {
        {11{instruction_word[31]}},
        instruction_word[31],
        instruction_word[19:12],
        instruction_word[20],
        instruction_word[30:21],
        1'b0
    };

    always @* begin
        valid = 1'b1;
        illegal = 1'b0;
        format_class = `PENGUIN_FMT_I;
        scalar_op_class = `PENGUIN_SCALAR_OP_ALU_IMM;
        alu_fn = `PENGUIN_ALU_ADD;
        rd = field_rd;
        rs1 = field_rs1;
        rs2 = field_rs2;
        imm32 = 32'd0;
        writes_rd = 1'b0;
        reads_rs1 = 1'b0;
        reads_rs2 = 1'b0;
        is_branch = 1'b0;
        is_jump = 1'b0;
        is_load = 1'b0;
        is_store = 1'b0;
        is_fence = 1'b0;
        is_ecall = 1'b0;
        is_ebreak = 1'b0;
        is_reserved_custom = 1'b0;

        case (opcode)
            `PENGUIN_OPCODE_LUI: begin
                format_class = `PENGUIN_FMT_U;
                scalar_op_class = `PENGUIN_SCALAR_OP_UPPER_IMM;
                alu_fn = `PENGUIN_ALU_COPY_B;
                imm32 = u_imm;
                writes_rd = 1'b1;
            end
            `PENGUIN_OPCODE_AUIPC: begin
                format_class = `PENGUIN_FMT_U;
                scalar_op_class = `PENGUIN_SCALAR_OP_UPPER_IMM;
                alu_fn = `PENGUIN_ALU_ADD;
                imm32 = u_imm;
                writes_rd = 1'b1;
            end
            `PENGUIN_OPCODE_JAL: begin
                format_class = `PENGUIN_FMT_J;
                scalar_op_class = `PENGUIN_SCALAR_OP_JUMP;
                imm32 = j_imm;
                writes_rd = 1'b1;
                is_jump = 1'b1;
            end
            `PENGUIN_OPCODE_JALR: begin
                format_class = `PENGUIN_FMT_I;
                scalar_op_class = `PENGUIN_SCALAR_OP_JUMP;
                imm32 = i_imm;
                writes_rd = 1'b1;
                reads_rs1 = 1'b1;
                is_jump = 1'b1;
                if (funct3 != `PENGUIN_FUNCT3_000) begin
                    illegal = 1'b1;
                end
            end
            `PENGUIN_OPCODE_BRANCH: begin
                format_class = `PENGUIN_FMT_B;
                scalar_op_class = `PENGUIN_SCALAR_OP_BRANCH;
                imm32 = b_imm;
                reads_rs1 = 1'b1;
                reads_rs2 = 1'b1;
                is_branch = 1'b1;
                case (funct3)
                    `PENGUIN_FUNCT3_000: alu_fn = `PENGUIN_ALU_COMPARE_EQ;
                    `PENGUIN_FUNCT3_001: alu_fn = `PENGUIN_ALU_COMPARE_NE;
                    `PENGUIN_FUNCT3_100: alu_fn = `PENGUIN_ALU_COMPARE_LT;
                    `PENGUIN_FUNCT3_101: alu_fn = `PENGUIN_ALU_COMPARE_GE;
                    `PENGUIN_FUNCT3_110: alu_fn = `PENGUIN_ALU_COMPARE_LTU;
                    `PENGUIN_FUNCT3_111: alu_fn = `PENGUIN_ALU_COMPARE_GEU;
                    default: illegal = 1'b1;
                endcase
            end
            `PENGUIN_OPCODE_LOAD: begin
                format_class = `PENGUIN_FMT_I;
                scalar_op_class = `PENGUIN_SCALAR_OP_LOAD;
                imm32 = i_imm;
                writes_rd = 1'b1;
                reads_rs1 = 1'b1;
                is_load = 1'b1;
                if (funct3 != `PENGUIN_FUNCT3_010) begin
                    illegal = 1'b1;
                end
            end
            `PENGUIN_OPCODE_STORE: begin
                format_class = `PENGUIN_FMT_S;
                scalar_op_class = `PENGUIN_SCALAR_OP_STORE;
                imm32 = s_imm;
                reads_rs1 = 1'b1;
                reads_rs2 = 1'b1;
                is_store = 1'b1;
                if (funct3 != `PENGUIN_FUNCT3_010) begin
                    illegal = 1'b1;
                end
            end
            `PENGUIN_OPCODE_OP_IMM: begin
                format_class = `PENGUIN_FMT_I;
                scalar_op_class = `PENGUIN_SCALAR_OP_ALU_IMM;
                writes_rd = 1'b1;
                reads_rs1 = 1'b1;
                imm32 = i_imm;
                case (funct3)
                    `PENGUIN_FUNCT3_000: alu_fn = `PENGUIN_ALU_ADD;
                    `PENGUIN_FUNCT3_010: alu_fn = `PENGUIN_ALU_SLT;
                    `PENGUIN_FUNCT3_011: alu_fn = `PENGUIN_ALU_SLTU;
                    `PENGUIN_FUNCT3_100: alu_fn = `PENGUIN_ALU_XOR;
                    `PENGUIN_FUNCT3_110: alu_fn = `PENGUIN_ALU_OR;
                    `PENGUIN_FUNCT3_111: alu_fn = `PENGUIN_ALU_AND;
                    `PENGUIN_FUNCT3_001: begin
                        alu_fn = `PENGUIN_ALU_SLL;
                        imm32 = {27'd0, instruction_word[24:20]};
                        if (funct7 != `PENGUIN_FUNCT7_0000000) begin
                            illegal = 1'b1;
                        end
                    end
                    `PENGUIN_FUNCT3_101: begin
                        imm32 = {27'd0, instruction_word[24:20]};
                        if (funct7 == `PENGUIN_FUNCT7_0000000) begin
                            alu_fn = `PENGUIN_ALU_SRL;
                        end else if (funct7 == `PENGUIN_FUNCT7_0100000) begin
                            alu_fn = `PENGUIN_ALU_SRA;
                        end else begin
                            illegal = 1'b1;
                        end
                    end
                    default: illegal = 1'b1;
                endcase
            end
            `PENGUIN_OPCODE_OP: begin
                format_class = `PENGUIN_FMT_R;
                scalar_op_class = `PENGUIN_SCALAR_OP_ALU_REG;
                writes_rd = 1'b1;
                reads_rs1 = 1'b1;
                reads_rs2 = 1'b1;
                case (funct3)
                    `PENGUIN_FUNCT3_000: begin
                        if (funct7 == `PENGUIN_FUNCT7_0000000) begin
                            alu_fn = `PENGUIN_ALU_ADD;
                        end else if (funct7 == `PENGUIN_FUNCT7_0100000) begin
                            alu_fn = `PENGUIN_ALU_SUB;
                        end else begin
                            illegal = 1'b1;
                        end
                    end
                    `PENGUIN_FUNCT3_001: begin
                        alu_fn = `PENGUIN_ALU_SLL;
                        if (funct7 != `PENGUIN_FUNCT7_0000000) begin
                            illegal = 1'b1;
                        end
                    end
                    `PENGUIN_FUNCT3_010: begin
                        alu_fn = `PENGUIN_ALU_SLT;
                        if (funct7 != `PENGUIN_FUNCT7_0000000) begin
                            illegal = 1'b1;
                        end
                    end
                    `PENGUIN_FUNCT3_011: begin
                        alu_fn = `PENGUIN_ALU_SLTU;
                        if (funct7 != `PENGUIN_FUNCT7_0000000) begin
                            illegal = 1'b1;
                        end
                    end
                    `PENGUIN_FUNCT3_100: begin
                        alu_fn = `PENGUIN_ALU_XOR;
                        if (funct7 != `PENGUIN_FUNCT7_0000000) begin
                            illegal = 1'b1;
                        end
                    end
                    `PENGUIN_FUNCT3_101: begin
                        if (funct7 == `PENGUIN_FUNCT7_0000000) begin
                            alu_fn = `PENGUIN_ALU_SRL;
                        end else if (funct7 == `PENGUIN_FUNCT7_0100000) begin
                            alu_fn = `PENGUIN_ALU_SRA;
                        end else begin
                            illegal = 1'b1;
                        end
                    end
                    `PENGUIN_FUNCT3_110: begin
                        alu_fn = `PENGUIN_ALU_OR;
                        if (funct7 != `PENGUIN_FUNCT7_0000000) begin
                            illegal = 1'b1;
                        end
                    end
                    `PENGUIN_FUNCT3_111: begin
                        alu_fn = `PENGUIN_ALU_AND;
                        if (funct7 != `PENGUIN_FUNCT7_0000000) begin
                            illegal = 1'b1;
                        end
                    end
                    default: illegal = 1'b1;
                endcase
            end
            `PENGUIN_OPCODE_MISC_MEM: begin
                format_class = `PENGUIN_FMT_MISC_MEM;
                scalar_op_class = `PENGUIN_SCALAR_OP_SYSTEM;
                is_fence = 1'b1;
                if (funct3 != `PENGUIN_FUNCT3_000) begin
                    illegal = 1'b1;
                end
            end
            `PENGUIN_OPCODE_SYSTEM: begin
                format_class = `PENGUIN_FMT_SYSTEM;
                scalar_op_class = `PENGUIN_SCALAR_OP_SYSTEM;
                if (funct3 != `PENGUIN_FUNCT3_000) begin
                    illegal = 1'b1;
                end else if (system_imm == `PENGUIN_SYSTEM_ECALL) begin
                    is_ecall = 1'b1;
                end else if (system_imm == `PENGUIN_SYSTEM_EBREAK) begin
                    is_ebreak = 1'b1;
                end else begin
                    illegal = 1'b1;
                end
            end
            `PENGUIN_OPCODE_CUSTOM_0,
            `PENGUIN_OPCODE_CUSTOM_1,
            `PENGUIN_OPCODE_CUSTOM_2,
            `PENGUIN_OPCODE_CUSTOM_3: begin
                format_class = `PENGUIN_FMT_RESERVED;
                is_reserved_custom = 1'b1;
                illegal = 1'b1;
            end
            default: begin
                illegal = 1'b1;
            end
        endcase
    end

endmodule
