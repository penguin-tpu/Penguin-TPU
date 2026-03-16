`timescale 1ns / 1ps

module penguin_scalar_lsu (
    input  wire [31:0] base_addr,
    input  wire [31:0] store_data,
    input  wire [31:0] imm32,
    input  wire        is_load,
    input  wire        is_store,
    input  wire [31:0] dmem_rdata,
    output reg         dmem_valid,
    output reg         dmem_write,
    output reg  [31:0] dmem_addr,
    output reg  [31:0] dmem_wdata,
    output reg  [31:0] load_data,
    output reg         load_misaligned,
    output reg         store_misaligned
);

    reg [31:0] effective_addr;

    always @* begin
        effective_addr = base_addr + imm32;
        dmem_valid = 1'b0;
        dmem_write = 1'b0;
        dmem_addr = effective_addr;
        dmem_wdata = store_data;
        load_data = dmem_rdata;
        load_misaligned = 1'b0;
        store_misaligned = 1'b0;

        if (is_load) begin
            load_misaligned = (effective_addr[1:0] != 2'b00);
            if (!load_misaligned) begin
                dmem_valid = 1'b1;
            end
        end

        if (is_store) begin
            store_misaligned = (effective_addr[1:0] != 2'b00);
            if (!store_misaligned) begin
                dmem_valid = 1'b1;
                dmem_write = 1'b1;
            end
        end
    end

endmodule
