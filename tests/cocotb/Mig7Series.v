`timescale 1ns / 1ps

module Mig7Series (
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
    output wire [0:0]  ddr3_odt,
    input  wire        sys_clk_i,
    input  wire        clk_ref_i,
    output wire        ui_clk,
    output reg         ui_clk_sync_rst,
    output reg         mmcm_locked,
    input  wire        aresetn,
    input  wire        app_sr_req,
    input  wire        app_ref_req,
    input  wire        app_zq_req,
    output wire        app_sr_active,
    output wire        app_ref_ack,
    output wire        app_zq_ack,
    input  wire [0:0]  s_axi_awid,
    input  wire [28:0] s_axi_awaddr,
    input  wire [7:0]  s_axi_awlen,
    input  wire [2:0]  s_axi_awsize,
    input  wire [1:0]  s_axi_awburst,
    input  wire [0:0]  s_axi_awlock,
    input  wire [3:0]  s_axi_awcache,
    input  wire [2:0]  s_axi_awprot,
    input  wire [3:0]  s_axi_awqos,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [127:0] s_axi_wdata,
    input  wire [15:0] s_axi_wstrb,
    input  wire        s_axi_wlast,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    input  wire        s_axi_bready,
    output reg  [0:0]  s_axi_bid,
    output reg  [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire [0:0]  s_axi_arid,
    input  wire [28:0] s_axi_araddr,
    input  wire [7:0]  s_axi_arlen,
    input  wire [2:0]  s_axi_arsize,
    input  wire [1:0]  s_axi_arburst,
    input  wire [0:0]  s_axi_arlock,
    input  wire [3:0]  s_axi_arcache,
    input  wire [2:0]  s_axi_arprot,
    input  wire [3:0]  s_axi_arqos,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    input  wire        s_axi_rready,
    output reg  [0:0]  s_axi_rid,
    output reg  [127:0] s_axi_rdata,
    output reg  [1:0]  s_axi_rresp,
    output reg         s_axi_rlast,
    output reg         s_axi_rvalid,
    output reg         init_calib_complete,
    output wire [11:0] device_temp,
    input  wire        sys_rst
);

    //------------------------------------------------------------------------------
    //  Parameters / localparams
    //------------------------------------------------------------------------------

    localparam integer MEM_WORDS = 1024;

    //------------------------------------------------------------------------------
    //  Logic declarations
    //------------------------------------------------------------------------------

    reg [127:0] memory [0:MEM_WORDS - 1];
    reg [3:0] init_counter;

    reg        write_addr_seen;
    reg        write_data_seen;
    reg [28:0] write_addr_latched;
    reg [127:0] write_data_latched;
    reg [15:0] write_strb_latched;

    reg        read_pending;
    reg [28:0] read_addr_latched;

    integer mem_index;
    integer byte_index;

    wire sys_reset_active;
    wire memory_ready;
    wire [9:0] write_word_index;
    wire [9:0] read_word_index;

    //------------------------------------------------------------------------------
    //  Core logic
    //------------------------------------------------------------------------------

    assign sys_reset_active = !sys_rst;
    assign ui_clk = sys_clk_i;
    assign s_axi_awready = memory_ready;
    assign s_axi_wready = memory_ready;
    assign s_axi_arready = memory_ready;
    assign app_sr_active = 1'b0;
    assign app_ref_ack = 1'b0;
    assign app_zq_ack = 1'b0;
    assign ddr3_addr = 15'd0;
    assign ddr3_ba = 3'd0;
    assign ddr3_ras_n = 1'b1;
    assign ddr3_cas_n = 1'b1;
    assign ddr3_we_n = 1'b1;
    assign ddr3_reset_n = 1'b1;
    assign ddr3_ck_p = 1'b0;
    assign ddr3_ck_n = 1'b0;
    assign ddr3_cke = 1'b0;
    assign ddr3_dm = 2'd0;
    assign ddr3_odt = 1'b0;
    assign ddr3_dq = 16'hzzzz;
    assign ddr3_dqs_n = 2'bzz;
    assign ddr3_dqs_p = 2'bzz;
    assign device_temp = 12'd0;
    assign memory_ready = init_calib_complete && aresetn && !sys_reset_active;
    assign write_word_index = write_addr_latched[13:4];
    assign read_word_index = read_addr_latched[13:4];

    always @(posedge ui_clk) begin
        if (sys_reset_active) begin
            init_counter <= 4'd0;
            ui_clk_sync_rst <= 1'b1;
            mmcm_locked <= 1'b0;
            init_calib_complete <= 1'b0;
            write_addr_seen <= 1'b0;
            write_data_seen <= 1'b0;
            read_pending <= 1'b0;
            s_axi_bvalid <= 1'b0;
            s_axi_rvalid <= 1'b0;
            s_axi_bid <= 1'b0;
            s_axi_bresp <= 2'b00;
            s_axi_rid <= 1'b0;
            s_axi_rresp <= 2'b00;
            s_axi_rlast <= 1'b0;
            s_axi_rdata <= 128'd0;

            for (mem_index = 0; mem_index < MEM_WORDS; mem_index = mem_index + 1) begin
                memory[mem_index] = 128'd0;
            end
        end else begin
            mmcm_locked <= 1'b1;

            if (!init_calib_complete) begin
                init_counter <= init_counter + 4'd1;
                if (init_counter == 4'd3) begin
                    init_calib_complete <= 1'b1;
                    ui_clk_sync_rst <= 1'b0;
                end
            end

            if (memory_ready && s_axi_awvalid) begin
                write_addr_seen <= 1'b1;
                write_addr_latched <= s_axi_awaddr;
                s_axi_bid <= s_axi_awid;
            end

            if (memory_ready && s_axi_wvalid) begin
                write_data_seen <= 1'b1;
                write_data_latched <= s_axi_wdata;
                write_strb_latched <= s_axi_wstrb;
            end

            if (write_addr_seen && write_data_seen && !s_axi_bvalid) begin
                for (byte_index = 0; byte_index < 16; byte_index = byte_index + 1) begin
                    if (write_strb_latched[byte_index]) begin
                        memory[write_word_index][byte_index * 8 +: 8] <=
                            write_data_latched[byte_index * 8 +: 8];
                    end
                end
                s_axi_bvalid <= 1'b1;
                s_axi_bresp <= 2'b00;
                write_addr_seen <= 1'b0;
                write_data_seen <= 1'b0;
            end

            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (memory_ready && s_axi_arvalid && !read_pending && !s_axi_rvalid) begin
                read_pending <= 1'b1;
                read_addr_latched <= s_axi_araddr;
                s_axi_rid <= s_axi_arid;
            end

            if (read_pending) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rdata <= memory[read_word_index];
                s_axi_rresp <= 2'b00;
                s_axi_rlast <= 1'b1;
                read_pending <= 1'b0;
            end

            if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
                s_axi_rlast <= 1'b0;
            end
        end
    end

endmodule
