`timescale 1ns / 1ps

module PenguinDramClockCrossing #(
    parameter [31:0] dram_base_addr = 32'h8000_0000
) (
    input  wire         core_clock,
    input  wire         core_reset,
    input  wire         request_valid,
    input  wire         request_write,
    input  wire [31:0]  request_addr,
    input  wire [31:0]  request_wdata,
    output wire         request_stall,
    output reg  [31:0]  request_rdata,
    input  wire         axi_clock,
    input  wire         axi_reset,
    output wire [0:0]   s_axi_awid,
    output wire [28:0]  s_axi_awaddr,
    output wire [7:0]   s_axi_awlen,
    output wire [2:0]   s_axi_awsize,
    output wire [1:0]   s_axi_awburst,
    output wire [0:0]   s_axi_awlock,
    output wire [3:0]   s_axi_awcache,
    output wire [2:0]   s_axi_awprot,
    output wire [3:0]   s_axi_awqos,
    output wire         s_axi_awvalid,
    input  wire         s_axi_awready,
    output wire [127:0] s_axi_wdata,
    output wire [15:0]  s_axi_wstrb,
    output wire         s_axi_wlast,
    output wire         s_axi_wvalid,
    input  wire         s_axi_wready,
    output wire         s_axi_bready,
    input  wire [0:0]   s_axi_bid,
    input  wire [1:0]   s_axi_bresp,
    input  wire         s_axi_bvalid,
    output wire [0:0]   s_axi_arid,
    output wire [28:0]  s_axi_araddr,
    output wire [7:0]   s_axi_arlen,
    output wire [2:0]   s_axi_arsize,
    output wire [1:0]   s_axi_arburst,
    output wire [0:0]   s_axi_arlock,
    output wire [3:0]   s_axi_arcache,
    output wire [2:0]   s_axi_arprot,
    output wire [3:0]   s_axi_arqos,
    output wire         s_axi_arvalid,
    input  wire         s_axi_arready,
    output wire         s_axi_rready,
    input  wire [0:0]   s_axi_rid,
    input  wire [127:0] s_axi_rdata,
    input  wire [1:0]   s_axi_rresp,
    input  wire         s_axi_rlast,
    input  wire         s_axi_rvalid
);

    reg        core_request_done;
    reg        core_request_inflight;
    reg        core_request_toggle;
    reg        core_request_write_reg;
    reg [31:0] core_request_addr_reg;
    reg [31:0] core_request_wdata_reg;
    reg        core_ack_toggle_seen;

    (* ASYNC_REG = "TRUE" *) reg        core_ack_toggle_sync_1;
    (* ASYNC_REG = "TRUE" *) reg        core_ack_toggle_sync_2;
    (* ASYNC_REG = "TRUE" *) reg [31:0] core_response_rdata_sync_1;
    (* ASYNC_REG = "TRUE" *) reg [31:0] core_response_rdata_sync_2;

    (* ASYNC_REG = "TRUE" *) reg        axi_request_toggle_sync_1;
    (* ASYNC_REG = "TRUE" *) reg        axi_request_toggle_sync_2;
    (* ASYNC_REG = "TRUE" *) reg        axi_request_write_sync_1;
    (* ASYNC_REG = "TRUE" *) reg        axi_request_write_sync_2;
    (* ASYNC_REG = "TRUE" *) reg [31:0] axi_request_addr_sync_1;
    (* ASYNC_REG = "TRUE" *) reg [31:0] axi_request_addr_sync_2;
    (* ASYNC_REG = "TRUE" *) reg [31:0] axi_request_wdata_sync_1;
    (* ASYNC_REG = "TRUE" *) reg [31:0] axi_request_wdata_sync_2;

    reg        axi_request_active;
    reg        axi_request_write_reg;
    reg [31:0] axi_request_addr_reg;
    reg [31:0] axi_request_wdata_reg;
    reg        axi_request_toggle_seen;
    reg        axi_ack_toggle;
    reg [31:0] axi_response_rdata_reg;

    wire        axi_request_stall;
    wire [31:0] axi_request_rdata;

    assign request_stall = request_valid && !core_request_done;

    always @(posedge core_clock) begin
        if (core_reset) begin
            core_request_done <= 1'b0;
            core_request_inflight <= 1'b0;
            core_request_toggle <= 1'b0;
            core_request_write_reg <= 1'b0;
            core_request_addr_reg <= 32'd0;
            core_request_wdata_reg <= 32'd0;
            core_ack_toggle_seen <= 1'b0;
            core_ack_toggle_sync_1 <= 1'b0;
            core_ack_toggle_sync_2 <= 1'b0;
            core_response_rdata_sync_1 <= 32'd0;
            core_response_rdata_sync_2 <= 32'd0;
            request_rdata <= 32'd0;
        end else begin
            core_ack_toggle_sync_1 <= axi_ack_toggle;
            core_ack_toggle_sync_2 <= core_ack_toggle_sync_1;
            core_response_rdata_sync_1 <= axi_response_rdata_reg;
            core_response_rdata_sync_2 <= core_response_rdata_sync_1;

            if (!request_valid) begin
                core_request_done <= 1'b0;
            end

            if (request_valid && !core_request_inflight && !core_request_done) begin
                core_request_inflight <= 1'b1;
                core_request_toggle <= ~core_request_toggle;
                core_request_write_reg <= request_write;
                core_request_addr_reg <= request_addr;
                core_request_wdata_reg <= request_wdata;
            end

            if (core_request_inflight && (core_ack_toggle_sync_2 != core_ack_toggle_seen)) begin
                core_request_inflight <= 1'b0;
                core_request_done <= 1'b1;
                core_ack_toggle_seen <= core_ack_toggle_sync_2;
                request_rdata <= core_response_rdata_sync_2;
            end
        end
    end

    always @(posedge axi_clock) begin
        if (axi_reset) begin
            axi_request_toggle_sync_1 <= 1'b0;
            axi_request_toggle_sync_2 <= 1'b0;
            axi_request_write_sync_1 <= 1'b0;
            axi_request_write_sync_2 <= 1'b0;
            axi_request_addr_sync_1 <= 32'd0;
            axi_request_addr_sync_2 <= 32'd0;
            axi_request_wdata_sync_1 <= 32'd0;
            axi_request_wdata_sync_2 <= 32'd0;
            axi_request_active <= 1'b0;
            axi_request_write_reg <= 1'b0;
            axi_request_addr_reg <= 32'd0;
            axi_request_wdata_reg <= 32'd0;
            axi_request_toggle_seen <= 1'b0;
            axi_ack_toggle <= 1'b0;
            axi_response_rdata_reg <= 32'd0;
        end else begin
            axi_request_toggle_sync_1 <= core_request_toggle;
            axi_request_toggle_sync_2 <= axi_request_toggle_sync_1;
            axi_request_write_sync_1 <= core_request_write_reg;
            axi_request_write_sync_2 <= axi_request_write_sync_1;
            axi_request_addr_sync_1 <= core_request_addr_reg;
            axi_request_addr_sync_2 <= axi_request_addr_sync_1;
            axi_request_wdata_sync_1 <= core_request_wdata_reg;
            axi_request_wdata_sync_2 <= axi_request_wdata_sync_1;

            if (!axi_request_active && (axi_request_toggle_sync_2 != axi_request_toggle_seen)) begin
                axi_request_active <= 1'b1;
                axi_request_toggle_seen <= axi_request_toggle_sync_2;
                axi_request_write_reg <= axi_request_write_sync_2;
                axi_request_addr_reg <= axi_request_addr_sync_2;
                axi_request_wdata_reg <= axi_request_wdata_sync_2;
            end

            if (axi_request_active && !axi_request_stall) begin
                axi_request_active <= 1'b0;
                axi_ack_toggle <= ~axi_ack_toggle;
                axi_response_rdata_reg <= axi_request_rdata;
            end
        end
    end

    PenguinDramAxiBridge #(
        .dram_base_addr(dram_base_addr)
    ) dram_axi_bridge_inst (
        .clock(axi_clock),
        .reset(axi_reset),
        .request_valid(axi_request_active),
        .request_write(axi_request_write_reg),
        .request_addr(axi_request_addr_reg),
        .request_wdata(axi_request_wdata_reg),
        .request_stall(axi_request_stall),
        .request_rdata(axi_request_rdata),
        .s_axi_awid(s_axi_awid),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awlen(s_axi_awlen),
        .s_axi_awsize(s_axi_awsize),
        .s_axi_awburst(s_axi_awburst),
        .s_axi_awlock(s_axi_awlock),
        .s_axi_awcache(s_axi_awcache),
        .s_axi_awprot(s_axi_awprot),
        .s_axi_awqos(s_axi_awqos),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wlast(s_axi_wlast),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bready(s_axi_bready),
        .s_axi_bid(s_axi_bid),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_arid(s_axi_arid),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arlen(s_axi_arlen),
        .s_axi_arsize(s_axi_arsize),
        .s_axi_arburst(s_axi_arburst),
        .s_axi_arlock(s_axi_arlock),
        .s_axi_arcache(s_axi_arcache),
        .s_axi_arprot(s_axi_arprot),
        .s_axi_arqos(s_axi_arqos),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rready(s_axi_rready),
        .s_axi_rid(s_axi_rid),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rlast(s_axi_rlast),
        .s_axi_rvalid(s_axi_rvalid)
    );

endmodule
