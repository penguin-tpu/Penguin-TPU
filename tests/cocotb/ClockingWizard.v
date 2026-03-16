`timescale 1ns / 1ps

module ClockingWizard (
    output wire clk_out1,
    output wire clk_out2,
    input  wire resetn,
    output wire locked,
    input  wire clk_in1
);

    //------------------------------------------------------------------------------
    //  Logic declarations
    //------------------------------------------------------------------------------

    reg clk_out2_reg;

    //------------------------------------------------------------------------------
    //  Core logic
    //------------------------------------------------------------------------------

    assign clk_out1 = clk_in1;
    assign clk_out2 = clk_out2_reg;
    assign locked = resetn;

    always @(posedge clk_in1) begin
        if (!resetn) begin
            clk_out2_reg <= 1'b0;
        end
        else begin
            clk_out2_reg <= ~clk_out2_reg;
        end
    end

endmodule
