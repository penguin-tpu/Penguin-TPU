`timescale 1ns / 1ps

`include "penguin_scalar_defs.vh"

module penguin_scalar_alu (
    input  wire [4:0]  alu_fn,
    input  wire [31:0] lhs,
    input  wire [31:0] rhs,
    output reg  [31:0] result,
    output reg         compare_true
);

    wire signed [31:0] lhs_signed = lhs;
    wire signed [31:0] rhs_signed = rhs;
    wire [4:0] shift_amount = rhs[4:0];

    always @* begin
        result = 32'd0;
        compare_true = 1'b0;

        case (alu_fn)
            `PENGUIN_ALU_ADD: result = lhs + rhs;
            `PENGUIN_ALU_SUB: result = lhs - rhs;
            `PENGUIN_ALU_SLT: begin
                compare_true = (lhs_signed < rhs_signed);
                result = compare_true ? 32'd1 : 32'd0;
            end
            `PENGUIN_ALU_SLTU: begin
                compare_true = (lhs < rhs);
                result = compare_true ? 32'd1 : 32'd0;
            end
            `PENGUIN_ALU_XOR: result = lhs ^ rhs;
            `PENGUIN_ALU_OR: result = lhs | rhs;
            `PENGUIN_ALU_AND: result = lhs & rhs;
            `PENGUIN_ALU_SLL: result = lhs << shift_amount;
            `PENGUIN_ALU_SRL: result = lhs >> shift_amount;
            `PENGUIN_ALU_SRA: result = lhs_signed >>> shift_amount;
            `PENGUIN_ALU_COPY_B: result = rhs;
            `PENGUIN_ALU_COMPARE_EQ: compare_true = (lhs == rhs);
            `PENGUIN_ALU_COMPARE_NE: compare_true = (lhs != rhs);
            `PENGUIN_ALU_COMPARE_LT: compare_true = (lhs_signed < rhs_signed);
            `PENGUIN_ALU_COMPARE_GE: compare_true = (lhs_signed >= rhs_signed);
            `PENGUIN_ALU_COMPARE_LTU: compare_true = (lhs < rhs);
            `PENGUIN_ALU_COMPARE_GEU: compare_true = (lhs >= rhs);
            default: result = 32'd0;
        endcase

        if (
            (alu_fn == `PENGUIN_ALU_COMPARE_EQ) ||
            (alu_fn == `PENGUIN_ALU_COMPARE_NE) ||
            (alu_fn == `PENGUIN_ALU_COMPARE_LT) ||
            (alu_fn == `PENGUIN_ALU_COMPARE_GE) ||
            (alu_fn == `PENGUIN_ALU_COMPARE_LTU) ||
            (alu_fn == `PENGUIN_ALU_COMPARE_GEU)
        ) begin
            result = compare_true ? 32'd1 : 32'd0;
        end
    end

endmodule
