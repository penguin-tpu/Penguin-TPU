`timescale 1ns / 1ps

module penguin_scalar_regfile (
    input  wire        clock,
    input  wire        reset,
    input  wire [4:0]  rs1_addr,
    input  wire [4:0]  rs2_addr,
    output wire [31:0] rs1_data,
    output wire [31:0] rs2_data,
    input  wire [4:0]  debug_addr,
    output wire [31:0] debug_data,
    input  wire        write_enable,
    input  wire [4:0]  write_addr,
    input  wire [31:0] write_data
);

    reg [31:0] xreg [0:31];
    integer index;

    assign rs1_data = (rs1_addr == 5'd0) ? 32'd0 : xreg[rs1_addr];
    assign rs2_data = (rs2_addr == 5'd0) ? 32'd0 : xreg[rs2_addr];
    assign debug_data = (debug_addr == 5'd0) ? 32'd0 : xreg[debug_addr];

    always @(posedge clock) begin
        if (reset) begin
            for (index = 0; index < 32; index = index + 1) begin
                xreg[index] <= 32'd0;
            end
        end else if (write_enable && (write_addr != 5'd0)) begin
            xreg[write_addr] <= write_data;
        end
    end

endmodule
