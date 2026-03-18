`timescale 1ns / 1ps

module PenguinScalarUartHelloTop #(
    parameter integer clk_freq_hz = 50_000_000,
    parameter integer baud_rate = 115200,
    parameter [31:0] cycle_counter_increment = 32'd1
) (
    input  wire        sys_clk_i,
    input  wire        cpu_resetn,
    input  wire        uart_tx_in,
    output wire        uart_rx_out,
    inout  wire [15:0] ddr3_dq,
    inout  wire [1:0]  ddr3_dqs_n,
    inout  wire [1:0]  ddr3_dqs_p,
    output wire [14:0] ddr3_addr,
    output wire [2:0]  ddr3_ba,
    output wire        ddr3_ras_n,
    output wire        ddr3_cas_n,
    output wire        ddr3_we_n,
    output wire        ddr3_reset_n,
    output wire [0:0]  ddr3_ck_p,
    output wire [0:0]  ddr3_ck_n,
    output wire [0:0]  ddr3_cke,
    output wire [1:0]  ddr3_dm,
    output wire [0:0]  ddr3_odt
);

    //------------------------------------------------------------------------------
    //  Parameters / localparams
    //------------------------------------------------------------------------------

    localparam integer IMEM_WORDS = 256;
    localparam [31:0] UART_STATUS_ADDR = 32'h0000_0100;
    localparam [31:0] UART_TX_ADDR = 32'h0000_0104;
    localparam [31:0] CYCLE_COUNTER_ADDR = 32'h0000_0108;
    localparam [31:0] DRAM_STATUS_ADDR = 32'h0000_010C;
    localparam [31:0] DRAM_BASE_ADDR = 32'h8000_0000;
    localparam [31:0] UART_PRESCALE_CALC = (clk_freq_hz + (baud_rate * 4)) / (baud_rate * 8);
    localparam [15:0] UART_PRESCALE = UART_PRESCALE_CALC[15:0];

    //------------------------------------------------------------------------------
    //  Logic declarations
    //------------------------------------------------------------------------------

    reg [31:0] imem [0:IMEM_WORDS - 1];
    reg [31:0] cycle_counter_reg;
    integer imem_index;

    wire        clock;
    wire        reset;
    wire        dram_clock;
    wire        dram_reset;
    wire        sys_clock_bufg;
    wire        clock_wiz_locked;
    wire        core_clock;
    wire        mig_ref_clock;
    wire        mig_ui_clk;
    wire        mig_ui_clk_sync_rst;
    wire        mig_mmcm_locked;
    wire        mig_init_calib_complete;
    wire [11:0] mig_device_temp_unused;
    wire        mig_app_sr_active_unused;
    wire        mig_app_ref_ack_unused;
    wire        mig_app_zq_ack_unused;

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
    wire [7:0]  uart_rx_data_unused;
    wire        uart_rx_valid_unused;
    wire        uart_rx_busy_unused;
    wire        uart_rx_overrun_unused;
    wire        uart_rx_frame_unused;
    wire        uart_tx_busy_unused;

    wire        scalar_dmem_targets_mmio;
    wire        scalar_dmem_targets_dram;
    wire [31:0] scalar_mmio_rdata;
    wire        dram_request_stall;
    wire [31:0] dram_request_rdata;
    reg  [1:0]  core_reset_sync_reg;
    reg  [1:0]  dram_reset_sync_reg;

    wire [0:0]  mig_s_axi_awid;
    wire [28:0] mig_s_axi_awaddr;
    wire [7:0]  mig_s_axi_awlen;
    wire [2:0]  mig_s_axi_awsize;
    wire [1:0]  mig_s_axi_awburst;
    wire [0:0]  mig_s_axi_awlock;
    wire [3:0]  mig_s_axi_awcache;
    wire [2:0]  mig_s_axi_awprot;
    wire [3:0]  mig_s_axi_awqos;
    wire        mig_s_axi_awvalid;
    wire        mig_s_axi_awready;
    wire [127:0] mig_s_axi_wdata;
    wire [15:0] mig_s_axi_wstrb;
    wire        mig_s_axi_wlast;
    wire        mig_s_axi_wvalid;
    wire        mig_s_axi_wready;
    wire        mig_s_axi_bready;
    wire [0:0]  mig_s_axi_bid;
    wire [1:0]  mig_s_axi_bresp;
    wire        mig_s_axi_bvalid;
    wire [0:0]  mig_s_axi_arid;
    wire [28:0] mig_s_axi_araddr;
    wire [7:0]  mig_s_axi_arlen;
    wire [2:0]  mig_s_axi_arsize;
    wire [1:0]  mig_s_axi_arburst;
    wire [0:0]  mig_s_axi_arlock;
    wire [3:0]  mig_s_axi_arcache;
    wire [2:0]  mig_s_axi_arprot;
    wire [3:0]  mig_s_axi_arqos;
    wire        mig_s_axi_arvalid;
    wire        mig_s_axi_arready;
    wire        mig_s_axi_rready;
    wire [0:0]  mig_s_axi_rid;
    wire [127:0] mig_s_axi_rdata;
    wire [1:0]  mig_s_axi_rresp;
    wire        mig_s_axi_rlast;
    wire        mig_s_axi_rvalid;

    //------------------------------------------------------------------------------
    //  Core logic: clocking and decode
    //------------------------------------------------------------------------------

    assign clock = core_clock;
    assign dram_clock = mig_ui_clk;
    assign reset = core_reset_sync_reg[0];
    assign dram_reset = dram_reset_sync_reg[0];

    assign uart_tx_data = scalar_dmem_wdata[7:0];
    assign scalar_imem_rdata =
        (scalar_imem_addr[31:10] == 22'd0) ? imem[scalar_imem_addr[9:2]] : 32'h0010_0073;
    assign scalar_dmem_targets_mmio = scalar_dmem_valid && (scalar_dmem_addr[31:12] == 20'd0);
    assign scalar_dmem_targets_dram = scalar_dmem_valid && (scalar_dmem_addr[31] == 1'b1);
    assign scalar_mmio_rdata =
        (scalar_dmem_addr == UART_STATUS_ADDR) ? {31'd0, uart_tx_ready} :
        (scalar_dmem_addr == CYCLE_COUNTER_ADDR) ? cycle_counter_reg :
        (scalar_dmem_addr == DRAM_STATUS_ADDR) ? {
            28'd0,
            clock_wiz_locked,
            mig_mmcm_locked,
            mig_init_calib_complete,
            !mig_ui_clk_sync_rst
        } :
        32'd0;
    assign scalar_dmem_rdata = scalar_dmem_targets_dram ? dram_request_rdata : scalar_mmio_rdata;
    assign uart_tx_valid =
        scalar_dmem_valid &&
        scalar_dmem_write &&
        scalar_dmem_targets_mmio &&
        (scalar_dmem_addr == UART_TX_ADDR);

    BUFG system_clock_bufg_inst (
        .I(sys_clk_i),
        .O(sys_clock_bufg)
    );

    ClockingWizard clock_wiz_inst (
        .clk_out1(core_clock),
        .clk_out2(mig_ref_clock),
        .resetn(cpu_resetn),
        .locked(clock_wiz_locked),
        .clk_in1(sys_clock_bufg)
    );

    Mig7Series mig_inst (
        .ddr3_dq(ddr3_dq),
        .ddr3_dqs_n(ddr3_dqs_n),
        .ddr3_dqs_p(ddr3_dqs_p),
        .ddr3_addr(ddr3_addr),
        .ddr3_ba(ddr3_ba),
        .ddr3_ras_n(ddr3_ras_n),
        .ddr3_cas_n(ddr3_cas_n),
        .ddr3_we_n(ddr3_we_n),
        .ddr3_reset_n(ddr3_reset_n),
        .ddr3_ck_p(ddr3_ck_p),
        .ddr3_ck_n(ddr3_ck_n),
        .ddr3_cke(ddr3_cke),
        .ddr3_dm(ddr3_dm),
        .ddr3_odt(ddr3_odt),
        .sys_clk_i(sys_clock_bufg),
        .clk_ref_i(mig_ref_clock),
        .ui_clk(mig_ui_clk),
        .ui_clk_sync_rst(mig_ui_clk_sync_rst),
        .mmcm_locked(mig_mmcm_locked),
        .aresetn(~dram_reset),
        .app_sr_req(1'b0),
        .app_ref_req(1'b0),
        .app_zq_req(1'b0),
        .app_sr_active(mig_app_sr_active_unused),
        .app_ref_ack(mig_app_ref_ack_unused),
        .app_zq_ack(mig_app_zq_ack_unused),
        .s_axi_awid(mig_s_axi_awid),
        .s_axi_awaddr(mig_s_axi_awaddr),
        .s_axi_awlen(mig_s_axi_awlen),
        .s_axi_awsize(mig_s_axi_awsize),
        .s_axi_awburst(mig_s_axi_awburst),
        .s_axi_awlock(mig_s_axi_awlock),
        .s_axi_awcache(mig_s_axi_awcache),
        .s_axi_awprot(mig_s_axi_awprot),
        .s_axi_awqos(mig_s_axi_awqos),
        .s_axi_awvalid(mig_s_axi_awvalid),
        .s_axi_awready(mig_s_axi_awready),
        .s_axi_wdata(mig_s_axi_wdata),
        .s_axi_wstrb(mig_s_axi_wstrb),
        .s_axi_wlast(mig_s_axi_wlast),
        .s_axi_wvalid(mig_s_axi_wvalid),
        .s_axi_wready(mig_s_axi_wready),
        .s_axi_bready(mig_s_axi_bready),
        .s_axi_bid(mig_s_axi_bid),
        .s_axi_bresp(mig_s_axi_bresp),
        .s_axi_bvalid(mig_s_axi_bvalid),
        .s_axi_arid(mig_s_axi_arid),
        .s_axi_araddr(mig_s_axi_araddr),
        .s_axi_arlen(mig_s_axi_arlen),
        .s_axi_arsize(mig_s_axi_arsize),
        .s_axi_arburst(mig_s_axi_arburst),
        .s_axi_arlock(mig_s_axi_arlock),
        .s_axi_arcache(mig_s_axi_arcache),
        .s_axi_arprot(mig_s_axi_arprot),
        .s_axi_arqos(mig_s_axi_arqos),
        .s_axi_arvalid(mig_s_axi_arvalid),
        .s_axi_arready(mig_s_axi_arready),
        .s_axi_rready(mig_s_axi_rready),
        .s_axi_rid(mig_s_axi_rid),
        .s_axi_rdata(mig_s_axi_rdata),
        .s_axi_rresp(mig_s_axi_rresp),
        .s_axi_rlast(mig_s_axi_rlast),
        .s_axi_rvalid(mig_s_axi_rvalid),
        .init_calib_complete(mig_init_calib_complete),
        .device_temp(mig_device_temp_unused),
        .sys_rst(cpu_resetn)
    );

    PenguinDramClockCrossing #(
        .dram_base_addr(DRAM_BASE_ADDR)
    ) dram_clock_crossing_inst (
        .core_clock(clock),
        .core_reset(reset),
        .request_valid(scalar_dmem_targets_dram),
        .request_write(scalar_dmem_write),
        .request_addr(scalar_dmem_addr),
        .request_wdata(scalar_dmem_wdata),
        .request_stall(dram_request_stall),
        .request_rdata(dram_request_rdata),
        .axi_clock(dram_clock),
        .axi_reset(dram_reset),
        .s_axi_awid(mig_s_axi_awid),
        .s_axi_awaddr(mig_s_axi_awaddr),
        .s_axi_awlen(mig_s_axi_awlen),
        .s_axi_awsize(mig_s_axi_awsize),
        .s_axi_awburst(mig_s_axi_awburst),
        .s_axi_awlock(mig_s_axi_awlock),
        .s_axi_awcache(mig_s_axi_awcache),
        .s_axi_awprot(mig_s_axi_awprot),
        .s_axi_awqos(mig_s_axi_awqos),
        .s_axi_awvalid(mig_s_axi_awvalid),
        .s_axi_awready(mig_s_axi_awready),
        .s_axi_wdata(mig_s_axi_wdata),
        .s_axi_wstrb(mig_s_axi_wstrb),
        .s_axi_wlast(mig_s_axi_wlast),
        .s_axi_wvalid(mig_s_axi_wvalid),
        .s_axi_wready(mig_s_axi_wready),
        .s_axi_bready(mig_s_axi_bready),
        .s_axi_bid(mig_s_axi_bid),
        .s_axi_bresp(mig_s_axi_bresp),
        .s_axi_bvalid(mig_s_axi_bvalid),
        .s_axi_arid(mig_s_axi_arid),
        .s_axi_araddr(mig_s_axi_araddr),
        .s_axi_arlen(mig_s_axi_arlen),
        .s_axi_arsize(mig_s_axi_arsize),
        .s_axi_arburst(mig_s_axi_arburst),
        .s_axi_arlock(mig_s_axi_arlock),
        .s_axi_arcache(mig_s_axi_arcache),
        .s_axi_arprot(mig_s_axi_arprot),
        .s_axi_arqos(mig_s_axi_arqos),
        .s_axi_arvalid(mig_s_axi_arvalid),
        .s_axi_arready(mig_s_axi_arready),
        .s_axi_rready(mig_s_axi_rready),
        .s_axi_rid(mig_s_axi_rid),
        .s_axi_rdata(mig_s_axi_rdata),
        .s_axi_rresp(mig_s_axi_rresp),
        .s_axi_rlast(mig_s_axi_rlast),
        .s_axi_rvalid(mig_s_axi_rvalid)
    );

    //------------------------------------------------------------------------------
    //  Core logic: program ROM initialization
    //------------------------------------------------------------------------------

    initial begin
        for (imem_index = 0; imem_index < IMEM_WORDS; imem_index = imem_index + 1) begin
            imem[imem_index] = 32'h0010_0073;
        end

        `include "scalar/penguin_scalar_uart_dram_hello_program_init.vh"
    end

    //------------------------------------------------------------------------------
    //  Core logic: sequential state
    //------------------------------------------------------------------------------

    always @(posedge clock) begin
        if (reset) begin
            cycle_counter_reg <= 32'd0;
        end else begin
            cycle_counter_reg <= cycle_counter_reg + cycle_counter_increment;
        end
    end

    always @(posedge clock or negedge cpu_resetn) begin
        if (!cpu_resetn) begin
            core_reset_sync_reg <= 2'b11;
        end else begin
            if (!clock_wiz_locked) begin
                core_reset_sync_reg <= 2'b11;
            end else begin
                core_reset_sync_reg <= {1'b0, core_reset_sync_reg[1]};
            end
        end
    end

    always @(posedge dram_clock or negedge cpu_resetn) begin
        if (!cpu_resetn) begin
            dram_reset_sync_reg <= 2'b11;
        end else begin
            if (!clock_wiz_locked || mig_ui_clk_sync_rst) begin
                dram_reset_sync_reg <= 2'b11;
            end else begin
                dram_reset_sync_reg <= {1'b0, dram_reset_sync_reg[1]};
            end
        end
    end

    //------------------------------------------------------------------------------
    //  Core logic: instantiated blocks
    //------------------------------------------------------------------------------

    PenguinScalarCore scalar_core_inst (
        .clock(clock),
        .reset(reset),
        .enable(!scalar_halted && !dram_request_stall),
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
        .tx_busy(uart_tx_busy_unused),
        .rx_busy(uart_rx_busy_unused),
        .rx_overrun_error(uart_rx_overrun_unused),
        .rx_frame_error(uart_rx_frame_unused),
        .prescale(UART_PRESCALE)
    );

endmodule
