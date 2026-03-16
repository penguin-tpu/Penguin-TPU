if {[info exists ::env(PENGUIN_VIVADO_TARGET)]} {
    set target_name $::env(PENGUIN_VIVADO_TARGET)
} else {
    set target_name "uart_hello"
}

if {$target_name eq "sclar_core"} {
    set target_name "scalar_core"
}

if {$target_name eq "uart_hello"} {
    set top_name "penguin_uart_hello_top"
} elseif {$target_name eq "scalar_core"} {
    set top_name "penguin_scalar_uart_hello_top"
} else {
    error "unsupported PENGUIN_VIVADO_TARGET '${target_name}'"
}

open_project /home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.xpr

# constraint file
add_files -fileset constrs_1 -norecurse /home/tk/Desktop/Penguin-TPU/rtl/constraints/NexysVideo_Master.xdc

# design sources
add_files { \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/uart.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/uart_rx.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/uart_tx.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/penguin_uart_hello_top.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/penguin_scalar_uart_hello_top.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/penguin_scalar_decoder.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/penguin_scalar_regfile.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/penguin_scalar_alu.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/penguin_scalar_branch_unit.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/penguin_scalar_lsu.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/penguin_scalar_controller.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/penguin_scalar_core.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/penguin_scalar_uart_hello_program_init.vh \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/penguin_scalar_defs.vh \
}

# active synthesis top
set_property top $top_name [current_fileset]

# simulation sources

# update compile order
update_compile_order -fileset sources_1

close_project
