`timescale 1ns / 1ps

module penguin_scalar_branch_unit (
    input  wire [31:0] pc,
    input  wire [31:0] rs1_value,
    input  wire [31:0] imm32,
    input  wire        is_branch,
    input  wire        is_jal,
    input  wire        is_jalr,
    input  wire        branch_condition_met,
    output reg         redirect_valid,
    output reg  [31:0] redirect_target,
    output reg         target_misaligned
);

    reg [31:0] branch_target;

    always @* begin
        branch_target = pc + imm32;
        redirect_valid = 1'b0;
        redirect_target = 32'd0;
        target_misaligned = 1'b0;

        if (is_jalr) begin
            redirect_valid = 1'b1;
            redirect_target = (rs1_value + imm32) & 32'hFFFF_FFFE;
            target_misaligned = (redirect_target[1:0] != 2'b00);
        end else if (is_jal) begin
            redirect_valid = 1'b1;
            redirect_target = branch_target;
            target_misaligned = (branch_target[1:0] != 2'b00);
        end else if (is_branch && branch_condition_met) begin
            redirect_valid = 1'b1;
            redirect_target = branch_target;
            target_misaligned = (branch_target[1:0] != 2'b00);
        end
    end

endmodule
