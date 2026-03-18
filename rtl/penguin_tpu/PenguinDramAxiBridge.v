`timescale 1ns / 1ps

module PenguinDramAxiBridge #(
    //------------------------------------------------------------------------------
    //  Parameters / localparams
    //------------------------------------------------------------------------------

    parameter [31:0] dram_base_addr = 32'h8000_0000
) (
    input  wire        clock,
    input  wire        reset,
    input  wire        request_valid,
    input  wire        request_write,
    input  wire [31:0] request_addr,
    input  wire [31:0] request_wdata,
    output wire        request_stall,
    output reg  [31:0] request_rdata,
    output reg  [0:0]  s_axi_awid,
    output reg  [28:0] s_axi_awaddr,
    output reg  [7:0]  s_axi_awlen,
    output reg  [2:0]  s_axi_awsize,
    output reg  [1:0]  s_axi_awburst,
    output reg  [0:0]  s_axi_awlock,
    output reg  [3:0]  s_axi_awcache,
    output reg  [2:0]  s_axi_awprot,
    output reg  [3:0]  s_axi_awqos,
    output reg         s_axi_awvalid,
    input  wire        s_axi_awready,
    output reg  [127:0] s_axi_wdata,
    output reg  [15:0] s_axi_wstrb,
    output reg         s_axi_wlast,
    output reg         s_axi_wvalid,
    input  wire        s_axi_wready,
    output reg         s_axi_bready,
    input  wire [0:0]  s_axi_bid,
    input  wire [1:0]  s_axi_bresp,
    input  wire        s_axi_bvalid,
    output reg  [0:0]  s_axi_arid,
    output reg  [28:0] s_axi_araddr,
    output reg  [7:0]  s_axi_arlen,
    output reg  [2:0]  s_axi_arsize,
    output reg  [1:0]  s_axi_arburst,
    output reg  [0:0]  s_axi_arlock,
    output reg  [3:0]  s_axi_arcache,
    output reg  [2:0]  s_axi_arprot,
    output reg  [3:0]  s_axi_arqos,
    output reg         s_axi_arvalid,
    input  wire        s_axi_arready,
    output reg         s_axi_rready,
    input  wire [0:0]  s_axi_rid,
    input  wire [127:0] s_axi_rdata,
    input  wire [1:0]  s_axi_rresp,
    input  wire        s_axi_rlast,
    input  wire        s_axi_rvalid
);

    //------------------------------------------------------------------------------
    //  Parameters / localparams
    //------------------------------------------------------------------------------

    localparam [2:0] AXI_SIZE_16_BYTES = 3'b100;
    localparam [1:0] AXI_BURST_INCR = 2'b01;

    //------------------------------------------------------------------------------
    //  Logic declarations
    //------------------------------------------------------------------------------

    reg        transaction_busy;
    reg        request_done;
    reg  [1:0] request_word_index;

    wire [31:0] dram_relative_addr;
    wire [31:0] aligned_relative_addr;
    wire [1:0] word_index;
    wire [15:0] request_wstrb_shifted;
    wire [127:0] request_wdata_shifted;

    //------------------------------------------------------------------------------
    //  Core logic: combinational helpers
    //------------------------------------------------------------------------------

    assign request_stall = request_valid && !request_done;
    assign dram_relative_addr = request_addr - dram_base_addr;
    assign aligned_relative_addr = {dram_relative_addr[31:4], 4'b0000};
    assign word_index = dram_relative_addr[3:2];
    assign request_wstrb_shifted = 16'h000F << (word_index * 4);
    assign request_wdata_shifted = {96'd0, request_wdata} << (word_index * 32);

    //------------------------------------------------------------------------------
    //  Core logic: sequential state
    //------------------------------------------------------------------------------

    always @(posedge clock) begin
        if (reset) begin
            transaction_busy <= 1'b0;
            request_done <= 1'b0;
            request_rdata <= 32'd0;
            request_word_index <= 2'd0;

            s_axi_awid <= 1'b0;
            s_axi_awaddr <= 29'd0;
            s_axi_awlen <= 8'd0;
            s_axi_awsize <= AXI_SIZE_16_BYTES;
            s_axi_awburst <= AXI_BURST_INCR;
            s_axi_awlock <= 1'b0;
            s_axi_awcache <= 4'b0011;
            s_axi_awprot <= 3'b000;
            s_axi_awqos <= 4'd0;
            s_axi_awvalid <= 1'b0;

            s_axi_wdata <= 128'd0;
            s_axi_wstrb <= 16'd0;
            s_axi_wlast <= 1'b1;
            s_axi_wvalid <= 1'b0;
            s_axi_bready <= 1'b0;

            s_axi_arid <= 1'b0;
            s_axi_araddr <= 29'd0;
            s_axi_arlen <= 8'd0;
            s_axi_arsize <= AXI_SIZE_16_BYTES;
            s_axi_arburst <= AXI_BURST_INCR;
            s_axi_arlock <= 1'b0;
            s_axi_arcache <= 4'b0011;
            s_axi_arprot <= 3'b000;
            s_axi_arqos <= 4'd0;
            s_axi_arvalid <= 1'b0;
            s_axi_rready <= 1'b0;
        end
        else begin
            if (!request_valid) begin
                request_done <= 1'b0;
            end

            if (request_valid && !transaction_busy && !request_done) begin
                transaction_busy <= 1'b1;
                request_word_index <= word_index;

                if (request_write) begin
                    s_axi_awaddr <= aligned_relative_addr[28:0];
                    s_axi_awvalid <= 1'b1;
                    s_axi_wdata <= request_wdata_shifted;
                    s_axi_wstrb <= request_wstrb_shifted;
                    s_axi_wvalid <= 1'b1;
                    s_axi_bready <= 1'b1;
                end
                else begin
                    s_axi_araddr <= aligned_relative_addr[28:0];
                    s_axi_arvalid <= 1'b1;
                    s_axi_rready <= 1'b1;
                end
            end

            if (s_axi_awvalid && s_axi_awready) begin
                s_axi_awvalid <= 1'b0;
            end

            if (s_axi_wvalid && s_axi_wready) begin
                s_axi_wvalid <= 1'b0;
            end

            if (s_axi_bready && s_axi_bvalid) begin
                transaction_busy <= 1'b0;
                request_done <= 1'b1;
                s_axi_bready <= 1'b0;
            end

            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_arvalid <= 1'b0;
            end

            if (s_axi_rready && s_axi_rvalid) begin
                request_rdata <= s_axi_rdata[request_word_index * 32 +: 32];
                transaction_busy <= 1'b0;
                request_done <= 1'b1;
                s_axi_rready <= 1'b0;
            end
        end
    end

endmodule
