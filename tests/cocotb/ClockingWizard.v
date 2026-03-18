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

    assign clk_out1 = clk_in1;
    assign clk_out2 = clk_in1;
    assign locked = resetn;

endmodule
