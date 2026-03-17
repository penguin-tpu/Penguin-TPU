`timescale 1ns / 1ps

`include "penguin_scalar_defs.vh"

module PenguinScalarCore #
(
    parameter [31:0] RESET_PC = 32'd0
)
(
    input  wire        clock,
    input  wire        reset,
    input  wire        enable,
    output wire [31:0] imem_addr,
    input  wire [31:0] imem_rdata,
    output wire        dmem_valid,
    output wire        dmem_write,
    output wire [31:0] dmem_addr,
    output wire [31:0] dmem_wdata,
    input  wire [31:0] dmem_rdata,
    input  wire [4:0]  debug_reg_addr,
    output wire [31:0] debug_reg_data,
    output wire [31:0] debug_pc,
    output wire        halted,
    output wire [3:0]  halt_reason
);

    wire        decode_valid;
    wire        decode_illegal;
    wire [3:0]  format_class;
    wire [3:0]  scalar_op_class;
    wire [4:0]  decode_alu_fn;
    wire [4:0]  decode_rd;
    wire [4:0]  decode_rs1;
    wire [4:0]  decode_rs2;
    wire [31:0] decode_imm32;
    wire        writes_rd;
    wire        reads_rs1;
    wire        reads_rs2;
    wire        is_branch;
    wire        is_jump;
    wire        is_load;
    wire        is_store;
    wire        is_fence;
    wire        is_ecall;
    wire        is_ebreak;
    wire        is_vadd;
    wire        is_reserved_custom;

    wire [31:0] current_pc;
    wire        pending_redirect_valid_unused;
    wire [31:0] pending_redirect_target_unused;
    wire [1:0]  pending_redirect_count_unused;

    wire [31:0] rs1_data;
    wire [31:0] rs2_data;
    wire [31:0] alu_result;
    wire        alu_compare_true;
    reg  [31:0] alu_lhs;
    reg  [31:0] alu_rhs;
    reg  [4:0]  alu_fn_selected;
    wire        branch_redirect_valid;
    wire [31:0] branch_redirect_target;
    wire        branch_target_misaligned;
    wire        dmem_valid_int;
    wire        dmem_write_int;
    wire [31:0] dmem_addr_int;
    wire [31:0] dmem_wdata_int;
    wire [31:0] lsu_dmem_rdata;
    wire [31:0] lsu_load_data;
    wire        load_misaligned;
    wire        store_misaligned;
    wire        vpu_mmio_selected;
    wire [31:0] vpu_mmio_rdata;
    wire        vpu_execute_stall;
    wire        dmem_targets_vpu_mmio;
    wire        internal_load_selected;
    wire        external_dmem_valid;
    wire        external_dmem_write;

    wire current_pc_misaligned = (current_pc[1:0] != 2'b00);
    wire instruction_illegal = decode_illegal || is_reserved_custom || !decode_valid;
    wire instruction_halts =
        current_pc_misaligned ||
        instruction_illegal ||
        (branch_redirect_valid && branch_target_misaligned) ||
        load_misaligned ||
        store_misaligned ||
        is_ecall ||
        is_ebreak;

    reg [31:0] writeback_data;
    reg writeback_enable;

    PenguinScalarDecoder decoder_inst (
        .instruction_word(imem_rdata),
        .valid(decode_valid),
        .illegal(decode_illegal),
        .format_class(format_class),
        .scalar_op_class(scalar_op_class),
        .alu_fn(decode_alu_fn),
        .rd(decode_rd),
        .rs1(decode_rs1),
        .rs2(decode_rs2),
        .imm32(decode_imm32),
        .writes_rd(writes_rd),
        .reads_rs1(reads_rs1),
        .reads_rs2(reads_rs2),
        .is_branch(is_branch),
        .is_jump(is_jump),
        .is_load(is_load),
        .is_store(is_store),
        .is_fence(is_fence),
        .is_ecall(is_ecall),
        .is_ebreak(is_ebreak),
        .is_vadd(is_vadd),
        .is_reserved_custom(is_reserved_custom)
    );

    PenguinScalarRegfile regfile_inst (
        .clock(clock),
        .reset(reset),
        .rs1_addr(decode_rs1),
        .rs2_addr(decode_rs2),
        .rs1_data(rs1_data),
        .rs2_data(rs2_data),
        .debug_addr(debug_reg_addr),
        .debug_data(debug_reg_data),
        .write_enable(writeback_enable),
        .write_addr(decode_rd),
        .write_data(writeback_data)
    );

    PenguinScalarAlu alu_inst (
        .alu_fn(alu_fn_selected),
        .lhs(alu_lhs),
        .rhs(alu_rhs),
        .result(alu_result),
        .compare_true(alu_compare_true)
    );

    PenguinScalarBranchUnit branch_unit_inst (
        .pc(current_pc),
        .rs1_value(rs1_data),
        .imm32(decode_imm32),
        .is_branch(is_branch),
        .is_jal(is_jump && (format_class == `PENGUIN_FMT_J)),
        .is_jalr(is_jump && (format_class == `PENGUIN_FMT_I)),
        .branch_condition_met(alu_compare_true),
        .redirect_valid(branch_redirect_valid),
        .redirect_target(branch_redirect_target),
        .target_misaligned(branch_target_misaligned)
    );

    PenguinScalarLsu lsu_inst (
        .base_addr(rs1_data),
        .store_data(rs2_data),
        .imm32(decode_imm32),
        .is_load(is_load),
        .is_store(is_store),
        .dmem_rdata(lsu_dmem_rdata),
        .dmem_valid(dmem_valid_int),
        .dmem_write(dmem_write_int),
        .dmem_addr(dmem_addr_int),
        .dmem_wdata(dmem_wdata_int),
        .load_data(lsu_load_data),
        .load_misaligned(load_misaligned),
        .store_misaligned(store_misaligned)
    );

    PenguinScalarController #(
        .RESET_PC(RESET_PC)
    ) controller_inst (
        .clock(clock),
        .reset(reset),
        .enable(enable),
        .instruction_illegal(instruction_illegal),
        .current_pc_misaligned(current_pc_misaligned),
        .load_misaligned(load_misaligned),
        .store_misaligned(store_misaligned),
        .hold(vpu_execute_stall),
        .redirect_valid(branch_redirect_valid),
        .redirect_target(branch_redirect_target),
        .redirect_misaligned(branch_target_misaligned),
        .is_ecall(is_ecall),
        .is_ebreak(is_ebreak),
        .pc(current_pc),
        .halted(halted),
        .halt_reason(halt_reason),
        .pending_redirect_valid(pending_redirect_valid_unused),
        .pending_redirect_target(pending_redirect_target_unused),
        .pending_redirect_count(pending_redirect_count_unused)
    );

    PenguinPreliminaryVpu preliminary_vpu_inst (
        .clock(clock),
        .reset(reset),
        .execute_vadd(is_vadd && !instruction_halts),
        .execute_md(decode_rd),
        .execute_ms1(decode_rs1),
        .execute_ms2(decode_rs2),
        .mmio_valid(dmem_valid_int && !instruction_halts),
        .mmio_write(dmem_write_int),
        .mmio_addr(dmem_addr_int),
        .mmio_wdata(dmem_wdata_int),
        .mmio_rdata(vpu_mmio_rdata),
        .mmio_selected(vpu_mmio_selected),
        .execute_stall(vpu_execute_stall)
    );

    assign imem_addr = current_pc;
    assign debug_pc = current_pc;
    assign dmem_targets_vpu_mmio = dmem_valid_int && vpu_mmio_selected;
    assign internal_load_selected = dmem_targets_vpu_mmio && !dmem_write_int;
    assign lsu_dmem_rdata = internal_load_selected ? vpu_mmio_rdata : dmem_rdata;
    assign external_dmem_valid = dmem_valid_int && !instruction_halts && !dmem_targets_vpu_mmio;
    assign external_dmem_write = dmem_write_int && !instruction_halts && !dmem_targets_vpu_mmio;
    assign dmem_valid = external_dmem_valid;
    assign dmem_write = external_dmem_write;
    assign dmem_addr = dmem_addr_int;
    assign dmem_wdata = dmem_wdata_int;

    always @* begin
        alu_lhs = rs1_data;
        alu_rhs = decode_imm32;
        alu_fn_selected = decode_alu_fn;

        if (scalar_op_class == `PENGUIN_SCALAR_OP_ALU_REG) begin
            alu_rhs = rs2_data;
        end else if (scalar_op_class == `PENGUIN_SCALAR_OP_BRANCH) begin
            alu_rhs = rs2_data;
        end else if (scalar_op_class == `PENGUIN_SCALAR_OP_UPPER_IMM) begin
            if (decode_alu_fn == `PENGUIN_ALU_ADD) begin
                alu_lhs = current_pc;
            end else begin
                alu_lhs = 32'd0;
            end
            alu_rhs = decode_imm32;
        end

        writeback_enable = 1'b0;
        writeback_data = 32'd0;

        if (!instruction_halts) begin
            if (is_jump) begin
                writeback_enable = writes_rd;
                writeback_data = current_pc + 32'd4;
            end else if (is_load) begin
                writeback_enable = writes_rd;
                writeback_data = lsu_load_data;
            end else if (
                (scalar_op_class == `PENGUIN_SCALAR_OP_ALU_REG) ||
                (scalar_op_class == `PENGUIN_SCALAR_OP_ALU_IMM) ||
                (scalar_op_class == `PENGUIN_SCALAR_OP_UPPER_IMM)
            ) begin
                writeback_enable = writes_rd;
                writeback_data = alu_result;
            end
        end
    end

endmodule
