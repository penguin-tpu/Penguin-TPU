`timescale 1ns / 1ps

/*
 * Minimal FPGA bring-up top level.
 *
 * Sends "Hello World\r\n" over UART once per second. The UART core itself uses
 * the alexforencich/verilog-uart AXI-stream wrapper vendored into this tree.
 */
module PenguinUartHelloTop #
(
    parameter integer CLK_FREQ_HZ = 100_000_000,
    parameter integer BAUD_RATE = 115200
)
(
    input  wire sys_clk_i,
    input  wire cpu_resetn,
    input  wire uart_tx_in,
    output wire uart_rx_out
);

    localparam integer HELLO_LEN = 13;
    localparam [31:0] HELLO_LAST_INDEX_CALC = HELLO_LEN - 1;
    localparam [3:0] HELLO_LAST_INDEX = HELLO_LAST_INDEX_CALC[3:0];
    localparam [31:0] UART_PRESCALE_CALC = (CLK_FREQ_HZ + (BAUD_RATE * 4)) / (BAUD_RATE * 8);
    localparam [15:0] UART_PRESCALE = UART_PRESCALE_CALC[15:0];

    reg [31:0] second_counter_reg = 0;
    reg        message_active_reg = 0;
    reg [3:0]  message_index_reg = 0;
    reg [7:0]  uart_tx_data_reg = 8'd0;
    reg        uart_tx_valid_reg = 0;

    wire       uart_tx_ready;
    wire       uart_tx_busy;
    wire [7:0] uart_rx_data_unused;
    wire       uart_rx_valid_unused;
    wire       uart_rx_busy_unused;
    wire       uart_rx_overrun_unused;
    wire       uart_rx_frame_unused;
    wire       clock = sys_clk_i;
    wire       reset = !cpu_resetn;

    function [7:0] hello_byte;
        input [3:0] index;
        begin
            case (index)
                4'd0:  hello_byte = "H";
                4'd1:  hello_byte = "e";
                4'd2:  hello_byte = "l";
                4'd3:  hello_byte = "l";
                4'd4:  hello_byte = "o";
                4'd5:  hello_byte = " ";
                4'd6:  hello_byte = "W";
                4'd7:  hello_byte = "o";
                4'd8:  hello_byte = "r";
                4'd9:  hello_byte = "l";
                4'd10: hello_byte = "d";
                4'd11: hello_byte = 8'h0d;
                4'd12: hello_byte = 8'h0a;
                default: hello_byte = 8'h00;
            endcase
        end
    endfunction

    Uart uart_inst (
        .clock(clock),
        .reset(reset),
        .s_axis_tdata(uart_tx_data_reg),
        .s_axis_tvalid(uart_tx_valid_reg),
        .s_axis_tready(uart_tx_ready),
        .m_axis_tdata(uart_rx_data_unused),
        .m_axis_tvalid(uart_rx_valid_unused),
        .m_axis_tready(1'b1),
        .rxd(uart_tx_in),
        .txd(uart_rx_out),
        .tx_busy(uart_tx_busy),
        .rx_busy(uart_rx_busy_unused),
        .rx_overrun_error(uart_rx_overrun_unused),
        .rx_frame_error(uart_rx_frame_unused),
        .prescale(UART_PRESCALE)
    );

    always @(posedge clock) begin
        if (reset) begin
            second_counter_reg <= 0;
            message_active_reg <= 0;
            message_index_reg <= 0;
            uart_tx_data_reg <= 8'd0;
            uart_tx_valid_reg <= 0;
        end else begin
            if (second_counter_reg == CLK_FREQ_HZ-1) begin
                second_counter_reg <= 0;
                if (!message_active_reg && !uart_tx_busy) begin
                    message_active_reg <= 1'b1;
                    message_index_reg <= 0;
                end
            end else begin
                second_counter_reg <= second_counter_reg + 1'b1;
            end

            if (uart_tx_valid_reg && uart_tx_ready) begin
                uart_tx_valid_reg <= 1'b0;

                if (message_index_reg == HELLO_LAST_INDEX) begin
                    message_active_reg <= 1'b0;
                    message_index_reg <= 0;
                end else begin
                    message_index_reg <= message_index_reg + 1'b1;
                end
            end

            if (message_active_reg && !uart_tx_valid_reg && !uart_tx_busy) begin
                uart_tx_data_reg <= hello_byte(message_index_reg);
                uart_tx_valid_reg <= 1'b1;
            end
        end
    end

endmodule
