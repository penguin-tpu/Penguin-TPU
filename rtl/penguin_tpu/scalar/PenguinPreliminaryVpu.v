`timescale 1ns / 1ps

module PenguinPreliminaryVpu #(
    //------------------------------------------------------------------------------
    //  Logic declarations
    //------------------------------------------------------------------------------

    parameter [31:0] mmio_base_addr = 32'h0000_0200
) (
    input  wire        clock,
    input  wire        reset,
    input  wire        execute_vadd,
    input  wire [4:0]  execute_md,
    input  wire [4:0]  execute_ms1,
    input  wire [4:0]  execute_ms2,
    input  wire        mmio_valid,
    input  wire        mmio_write,
    input  wire [31:0] mmio_addr,
    input  wire [31:0] mmio_wdata,
    output reg  [31:0] mmio_rdata,
    output wire        mmio_selected,
    output wire        execute_stall
);

    //------------------------------------------------------------------------------
    //  Parameters / localparams
    //------------------------------------------------------------------------------

    localparam integer NUM_MREG = 32;
    localparam [31:0] MREG_MMIO_LIMIT_ADDR = mmio_base_addr + 32'd128;
    localparam [31:0] STATUS_ADDR = mmio_base_addr + 32'd128;

    //------------------------------------------------------------------------------
    //  Logic declarations
    //------------------------------------------------------------------------------

    reg [15:0] mreg_file [0:NUM_MREG - 1];
    reg        vpu_busy;
    reg        execute_done;
    reg [4:0]  pending_md;
    reg        operand_a_valid;
    reg        operand_b_valid;
    reg [15:0] operand_a_data;
    reg [15:0] operand_b_data;
    integer    mreg_index;

    wire       mmio_hits_mreg;
    wire [4:0] mmio_mreg_index;
    wire        result_valid;
    wire [15:0] result_data;

    //------------------------------------------------------------------------------
    //  Core logic: MMIO decode
    //------------------------------------------------------------------------------

    assign mmio_hits_mreg = (mmio_addr >= mmio_base_addr) && (mmio_addr < MREG_MMIO_LIMIT_ADDR);
    assign mmio_mreg_index = mmio_addr[6:2];
    assign mmio_selected = mmio_hits_mreg || (mmio_addr == STATUS_ADDR);
    assign execute_stall = execute_vadd && !execute_done;

    always @* begin
        mmio_rdata = 32'd0;

        if (mmio_hits_mreg) begin
            mmio_rdata = {16'd0, mreg_file[mmio_mreg_index]};
        end
        else if (mmio_addr == STATUS_ADDR) begin
            mmio_rdata = {30'd0, execute_done, vpu_busy};
        end
    end

    Bf16Adder bf16_adder_inst (
        .aclk(clock),
        .s_axis_a_tvalid(operand_a_valid),
        .s_axis_a_tdata(operand_a_data),
        .s_axis_b_tvalid(operand_b_valid),
        .s_axis_b_tdata(operand_b_data),
        .m_axis_result_tvalid(result_valid),
        .m_axis_result_tdata(result_data)
    );

    //------------------------------------------------------------------------------
    //  Core logic: sequential state
    //------------------------------------------------------------------------------

    always @(posedge clock) begin
        if (reset) begin
            vpu_busy <= 1'b0;
            execute_done <= 1'b0;
            pending_md <= 5'd0;
            operand_a_valid <= 1'b0;
            operand_b_valid <= 1'b0;
            operand_a_data <= 16'd0;
            operand_b_data <= 16'd0;

            for (mreg_index = 0; mreg_index < NUM_MREG; mreg_index = mreg_index + 1) begin
                mreg_file[mreg_index] <= 16'd0;
            end
        end
        else begin
            operand_a_valid <= 1'b0;
            operand_b_valid <= 1'b0;

            if (!execute_vadd) begin
                execute_done <= 1'b0;
            end

            if (mmio_valid && mmio_write && mmio_hits_mreg) begin
                mreg_file[mmio_mreg_index] <= mmio_wdata[15:0];
            end

            if (execute_vadd && !vpu_busy && !execute_done) begin
                operand_a_valid <= 1'b1;
                operand_b_valid <= 1'b1;
                operand_a_data <= mreg_file[execute_ms1];
                operand_b_data <= mreg_file[execute_ms2];
                pending_md <= execute_md;
                vpu_busy <= 1'b1;
            end

            if (result_valid) begin
                mreg_file[pending_md] <= result_data;
                vpu_busy <= 1'b0;
                execute_done <= 1'b1;
            end
        end
    end

endmodule
