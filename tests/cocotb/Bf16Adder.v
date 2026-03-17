`timescale 1ns / 1ps

module Bf16Adder (
    input  wire        aclk,
    input  wire        s_axis_a_tvalid,
    input  wire [15:0] s_axis_a_tdata,
    input  wire        s_axis_b_tvalid,
    input  wire [15:0] s_axis_b_tdata,
    output wire        m_axis_result_tvalid,
    output wire [15:0] m_axis_result_tdata
);

    //------------------------------------------------------------------------------
    //  Parameters / localparams
    //------------------------------------------------------------------------------

    localparam integer LATENCY = 8;

    //------------------------------------------------------------------------------
    //  Logic declarations
    //------------------------------------------------------------------------------

    reg [LATENCY - 1:0] valid_pipe;
    reg [15:0] data_pipe [0:LATENCY - 1];
    integer pipe_index;

    //------------------------------------------------------------------------------
    //  Core logic
    //------------------------------------------------------------------------------

    assign m_axis_result_tvalid = valid_pipe[LATENCY - 1];
    assign m_axis_result_tdata = data_pipe[LATENCY - 1];

    initial begin
        valid_pipe = {LATENCY{1'b0}};
        for (pipe_index = 0; pipe_index < LATENCY; pipe_index = pipe_index + 1) begin
            data_pipe[pipe_index] = 16'd0;
        end
    end

    always @(posedge aclk) begin
        valid_pipe[0] <= s_axis_a_tvalid && s_axis_b_tvalid;
        data_pipe[0] <= bf16_add(s_axis_a_tdata, s_axis_b_tdata);

        for (pipe_index = 1; pipe_index < LATENCY; pipe_index = pipe_index + 1) begin
            valid_pipe[pipe_index] <= valid_pipe[pipe_index - 1];
            data_pipe[pipe_index] <= data_pipe[pipe_index - 1];
        end
    end

    function [15:0] bf16_add;
        input [15:0] lhs;
        input [15:0] rhs;

        reg lhs_sign;
        reg rhs_sign;
        reg result_sign;
        reg [7:0] lhs_exp;
        reg [7:0] rhs_exp;
        reg [7:0] result_exp;
        reg [7:0] lhs_mantissa;
        reg [7:0] rhs_mantissa;
        reg [7:0] larger_mantissa;
        reg [7:0] smaller_mantissa;
        reg [7:0] normalized_mantissa;
        reg [8:0] sum_mantissa;
        reg [7:0] exp_delta;
        integer shift_count;
        begin
            lhs_sign = lhs[15];
            rhs_sign = rhs[15];
            lhs_exp = lhs[14:7];
            rhs_exp = rhs[14:7];
            lhs_mantissa = (lhs_exp == 8'd0) ? {1'b0, lhs[6:0]} : {1'b1, lhs[6:0]};
            rhs_mantissa = (rhs_exp == 8'd0) ? {1'b0, rhs[6:0]} : {1'b1, rhs[6:0]};

            if ((lhs_exp == 8'd0) && (lhs[6:0] == 7'd0)) begin
                bf16_add = rhs;
            end
            else if ((rhs_exp == 8'd0) && (rhs[6:0] == 7'd0)) begin
                bf16_add = lhs;
            end
            else begin
                if ((lhs_exp > rhs_exp) || ((lhs_exp == rhs_exp) && (lhs_mantissa >= rhs_mantissa))) begin
                    result_exp = lhs_exp;
                    result_sign = lhs_sign;
                    larger_mantissa = lhs_mantissa;
                    smaller_mantissa = rhs_mantissa;
                    exp_delta = lhs_exp - rhs_exp;
                end
                else begin
                    result_exp = rhs_exp;
                    result_sign = rhs_sign;
                    larger_mantissa = rhs_mantissa;
                    smaller_mantissa = lhs_mantissa;
                    exp_delta = rhs_exp - lhs_exp;
                end

                if (exp_delta > 8'd7) begin
                    smaller_mantissa = 8'd0;
                end
                else begin
                    smaller_mantissa = smaller_mantissa >> exp_delta;
                end

                if (lhs_sign == rhs_sign) begin
                    sum_mantissa = {1'b0, larger_mantissa} + {1'b0, smaller_mantissa};
                    if (sum_mantissa[8]) begin
                        normalized_mantissa = sum_mantissa[8:1];
                        result_exp = result_exp + 8'd1;
                    end
                    else begin
                        normalized_mantissa = sum_mantissa[7:0];
                    end
                end
                else begin
                    sum_mantissa = {1'b0, larger_mantissa} - {1'b0, smaller_mantissa};
                    normalized_mantissa = sum_mantissa[7:0];
                    for (shift_count = 0; shift_count < 7; shift_count = shift_count + 1) begin
                        if ((normalized_mantissa[7] == 1'b0) && (result_exp > 8'd0) && (normalized_mantissa != 8'd0)) begin
                            normalized_mantissa = normalized_mantissa << 1;
                            result_exp = result_exp - 8'd1;
                        end
                    end
                end

                if (normalized_mantissa == 8'd0) begin
                    bf16_add = 16'd0;
                end
                else begin
                    bf16_add = {result_sign, result_exp, normalized_mantissa[6:0]};
                end
            end
        end
    endfunction

endmodule
