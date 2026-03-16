open_project /home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.xpr

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
update_compile_order -fileset sources_1

close_project
