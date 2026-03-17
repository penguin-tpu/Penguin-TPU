`timescale 1ns / 1ps

`include "penguin_scalar_defs.vh"

module PenguinScalarController #
(
    parameter [31:0] RESET_PC = 32'd0
)
(
    input  wire        clock,
    input  wire        reset,
    input  wire        enable,
    input  wire        instruction_illegal,
    input  wire        current_pc_misaligned,
    input  wire        load_misaligned,
    input  wire        store_misaligned,
    input  wire        hold,
    input  wire        redirect_valid,
    input  wire [31:0] redirect_target,
    input  wire        redirect_misaligned,
    input  wire        is_ecall,
    input  wire        is_ebreak,
    output reg  [31:0] pc,
    output reg         halted,
    output reg  [3:0]  halt_reason,
    output reg         pending_redirect_valid,
    output reg  [31:0] pending_redirect_target,
    output reg  [1:0]  pending_redirect_count
);

    reg [31:0] next_pc;
    reg next_pending_valid;
    reg [31:0] next_pending_target;
    reg [1:0] next_pending_count;

    always @* begin
        next_pc = pc + 32'd4;
        next_pending_valid = pending_redirect_valid;
        next_pending_target = pending_redirect_target;
        next_pending_count = pending_redirect_count;

        if (redirect_valid) begin
            next_pending_valid = 1'b1;
            next_pending_target = redirect_target;
            next_pending_count = 2'd2;
        end else if (pending_redirect_valid) begin
            if (pending_redirect_count == 2'd1) begin
                next_pc = pending_redirect_target;
                next_pending_valid = 1'b0;
                next_pending_target = 32'd0;
                next_pending_count = 2'd0;
            end else begin
                next_pending_count = pending_redirect_count - 2'd1;
            end
        end
    end

    always @(posedge clock) begin
        if (reset) begin
            pc <= RESET_PC;
            halted <= 1'b0;
            halt_reason <= `PENGUIN_HALT_NONE;
            pending_redirect_valid <= 1'b0;
            pending_redirect_target <= 32'd0;
            pending_redirect_count <= 2'd0;
        end else if (enable && !halted) begin
            if (current_pc_misaligned) begin
                halted <= 1'b1;
                halt_reason <= `PENGUIN_HALT_INSN_MISALIGNED;
            end else if (instruction_illegal) begin
                halted <= 1'b1;
                halt_reason <= `PENGUIN_HALT_ILLEGAL_INSN;
            end else if (redirect_valid && redirect_misaligned) begin
                halted <= 1'b1;
                halt_reason <= `PENGUIN_HALT_INSN_MISALIGNED;
            end else if (load_misaligned) begin
                halted <= 1'b1;
                halt_reason <= `PENGUIN_HALT_LOAD_MISALIGNED;
            end else if (store_misaligned) begin
                halted <= 1'b1;
                halt_reason <= `PENGUIN_HALT_STORE_MISALIGNED;
            end else if (is_ecall) begin
                halted <= 1'b1;
                halt_reason <= `PENGUIN_HALT_ECALL;
            end else if (is_ebreak) begin
                halted <= 1'b1;
                halt_reason <= `PENGUIN_HALT_EBREAK;
            end else if (hold) begin
                pc <= pc;
                pending_redirect_valid <= pending_redirect_valid;
                pending_redirect_target <= pending_redirect_target;
                pending_redirect_count <= pending_redirect_count;
            end else begin
                pc <= next_pc;
                pending_redirect_valid <= next_pending_valid;
                pending_redirect_target <= next_pending_target;
                pending_redirect_count <= next_pending_count;
            end
        end
    end

endmodule
