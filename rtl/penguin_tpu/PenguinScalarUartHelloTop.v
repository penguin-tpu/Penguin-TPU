`timescale 1ns / 1ps

module PenguinScalarUartHelloTop #(
    parameter integer clk_freq_hz = 100_000_000,
    parameter integer baud_rate = 115200,
    // Keep the board image cycle-accurate by default; tests may override this
    // to accelerate the counter-driven 1 Hz loop in simulation.
    parameter [31:0] cycle_counter_increment = 32'd1
) (
    input  wire sys_clk_i,
    input  wire cpu_resetn,
    input  wire uart_tx_in,
    output wire uart_rx_out
);

    //------------------------------------------------------------------------------
    //  Parameters / localparams
    //------------------------------------------------------------------------------

    localparam integer IMEM_WORDS = 256;
    localparam integer CORE_CLK_FREQ_HZ = clk_freq_hz / 2;
    localparam [31:0] UART_STATUS_ADDR = 32'h0000_0100;
    localparam [31:0] UART_TX_ADDR = 32'h0000_0104;
    localparam [31:0] CYCLE_COUNTER_ADDR = 32'h0000_0108;
    localparam [31:0] UART_PRESCALE_CALC = (CORE_CLK_FREQ_HZ + (baud_rate * 4)) / (baud_rate * 8);
    localparam [15:0] UART_PRESCALE = UART_PRESCALE_CALC[15:0];

    //------------------------------------------------------------------------------
    //  Logic declarations
    //------------------------------------------------------------------------------

    reg [31:0] imem [0:IMEM_WORDS - 1];
    reg [31:0] cycle_counter_reg;
    integer imem_index;

    wire        clock;
    wire        clock_wiz_locked;
    wire        clock_100mhz_unused;
    wire        clock_50mhz;
    wire        reset;
    wire [31:0] scalar_imem_addr;
    wire [31:0] scalar_imem_rdata;
    wire        scalar_dmem_valid;
    wire        scalar_dmem_write;
    wire [31:0] scalar_dmem_addr;
    wire [31:0] scalar_dmem_wdata;
    wire [31:0] scalar_dmem_rdata;
    wire [31:0] scalar_debug_reg_data_unused;
    wire [31:0] scalar_debug_pc_unused;
    wire        scalar_halted;
    wire [3:0]  scalar_halt_reason_unused;
    wire [7:0]  uart_tx_data;
    wire        uart_tx_valid;
    wire        uart_tx_ready;
    wire        uart_tx_busy;
    wire [7:0]  uart_rx_data_unused;
    wire        uart_rx_valid_unused;
    wire        uart_rx_busy_unused;
    wire        uart_rx_overrun_unused;
    wire        uart_rx_frame_unused;

    //------------------------------------------------------------------------------
    //  Core logic: clocking and MMIO decode
    //------------------------------------------------------------------------------

    assign clock = clock_50mhz;
    assign reset = !cpu_resetn || !clock_wiz_locked;
    assign uart_tx_data = scalar_dmem_wdata[7:0];
    assign uart_tx_valid = scalar_dmem_valid && scalar_dmem_write && (scalar_dmem_addr == UART_TX_ADDR);
    assign scalar_imem_rdata =
        (scalar_imem_addr[31:10] == 22'd0) ? imem[scalar_imem_addr[9:2]] : 32'h0010_0073;
    assign scalar_dmem_rdata =
        (scalar_dmem_addr == UART_STATUS_ADDR) ? {31'd0, uart_tx_ready} :
        (scalar_dmem_addr == CYCLE_COUNTER_ADDR) ? cycle_counter_reg :
        32'd0;

    ClockingWizard clock_wiz_inst (
        .clk_out1(clock_100mhz_unused),
        .clk_out2(clock_50mhz),
        .resetn(cpu_resetn),
        .locked(clock_wiz_locked),
        .clk_in1(sys_clk_i)
    );
    
    //------------------------------------------------------------------------------
    //  Core logic: program ROM initialization
    //------------------------------------------------------------------------------

    initial begin
        for (imem_index = 0; imem_index < IMEM_WORDS; imem_index = imem_index + 1) begin
            imem[imem_index] = 32'h0010_0073;
        end

        `include "scalar/penguin_scalar_uart_vadd_program_init.vh"
    end

    //------------------------------------------------------------------------------
    //  Core logic: sequential state
    //------------------------------------------------------------------------------

    always @(posedge clock) begin
        if (reset) begin
            cycle_counter_reg <= 32'd0;
        end
        else begin
            cycle_counter_reg <= cycle_counter_reg + cycle_counter_increment;
        end
    end

    //------------------------------------------------------------------------------
    //  Core logic: instantiated blocks
    //------------------------------------------------------------------------------

    PenguinScalarCore scalar_core_inst (
        .clock(clock),
        .reset(reset),
        .enable(!scalar_halted),
        .imem_addr(scalar_imem_addr),
        .imem_rdata(scalar_imem_rdata),
        .dmem_valid(scalar_dmem_valid),
        .dmem_write(scalar_dmem_write),
        .dmem_addr(scalar_dmem_addr),
        .dmem_wdata(scalar_dmem_wdata),
        .dmem_rdata(scalar_dmem_rdata),
        .debug_reg_addr(5'd0),
        .debug_reg_data(scalar_debug_reg_data_unused),
        .debug_pc(scalar_debug_pc_unused),
        .halted(scalar_halted),
        .halt_reason(scalar_halt_reason_unused)
    );

    Uart uart_inst (
        .clock(clock),
        .reset(reset),
        .s_axis_tdata(uart_tx_data),
        .s_axis_tvalid(uart_tx_valid),
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

endmodule
