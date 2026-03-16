open_project /home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.xpr

# ==============================================================================
# Clocking Wizard IP
# ==============================================================================

if {[llength [get_ips -quiet ClockingWizard]] == 0} {
    create_ip -name clk_wiz -vendor xilinx.com -library ip -version 6.0 -module_name ClockingWizard
}

set_property -dict [list \
  CONFIG.CLKIN1_JITTER_PS {100.0} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {100.000} \
  CONFIG.CLKOUT2_JITTER {151.636} \
  CONFIG.CLKOUT2_PHASE_ERROR {98.575} \
  CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {50.000} \
  CONFIG.CLKOUT2_USED {true} \
  CONFIG.MMCM_CLKOUT1_DIVIDE {20} \
  CONFIG.NUM_OUT_CLKS {2} \
  CONFIG.PRIM_IN_FREQ {100.000} \
  CONFIG.RESET_PORT {resetn} \
  CONFIG.RESET_TYPE {ACTIVE_LOW} \
] [get_ips ClockingWizard]

set clock_wiz_xci [get_files /home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.srcs/sources_1/ip/ClockingWizard/ClockingWizard.xci]

generate_target {instantiation_template} $clock_wiz_xci
generate_target all $clock_wiz_xci


# ==============================================================================
# Floating Point IP
# ==============================================================================

create_ip -name floating_point -vendor xilinx.com -library ip -version 7.1 -module_name Bf16Adder
set_property -dict [list \
  CONFIG.A_Precision_Type {Custom} \
  CONFIG.Add_Sub_Value {Add} \
  CONFIG.C_A_Exponent_Width {8} \
  CONFIG.C_A_Fraction_Width {8} \
  CONFIG.C_Accum_Input_Msb {7} \
  CONFIG.C_Accum_Lsb {-9} \
  CONFIG.C_Accum_Msb {32} \
  CONFIG.C_Latency {8} \
  CONFIG.C_Mult_Usage {No_Usage} \
  CONFIG.C_Rate {1} \
  CONFIG.C_Result_Exponent_Width {8} \
  CONFIG.C_Result_Fraction_Width {8} \
  CONFIG.Flow_Control {NonBlocking} \
  CONFIG.Has_RESULT_TREADY {false} \
  CONFIG.Maximum_Latency {false} \
  CONFIG.Result_Precision_Type {Custom} \
] [get_ips Bf16Adder]

generate_target {instantiation_template} [get_files /home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.srcs/sources_1/ip/Bf16Adder/Bf16Adder.xci]
generate_target all [get_files  /home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.srcs/sources_1/ip/Bf16Adder/Bf16Adder.xci]


# Update Compile Order

update_compile_order -fileset sources_1

close_project
