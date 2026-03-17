if {[info exists ::env(PENGUIN_VIVADO_TARGET)]} {
    set target_name $::env(PENGUIN_VIVADO_TARGET)
} else {
    set target_name "uart_hello"
}

if {$target_name eq "sclar_core"} {
    set target_name "scalar_core"
}

if {$target_name eq "uart_hello"} {
    set top_name "PenguinUartHelloTop"
} elseif {$target_name eq "scalar_core"} {
    set top_name "PenguinScalarUartHelloTop"
} else {
    error "unsupported PENGUIN_VIVADO_TARGET '${target_name}'"
}

open_project /home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.xpr

set sources_fs [get_filesets sources_1]
set constrs_fs [get_filesets constrs_1]
set sim_fs [get_filesets sim_1]

set existing_source_files [get_files -quiet -of_objects $sources_fs]
if {[llength $existing_source_files] > 0} {
    remove_files -fileset $sources_fs $existing_source_files
}

set existing_constraint_files [get_files -quiet -of_objects $constrs_fs]
if {[llength $existing_constraint_files] > 0} {
    remove_files -fileset $constrs_fs $existing_constraint_files
}

# constraint file
add_files -fileset $constrs_fs -norecurse /home/tk/Desktop/Penguin-TPU/rtl/constraints/NexysVideo_Master.xdc

# design sources
add_files -fileset $sources_fs { \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/Uart.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/UartRx.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/UartTx.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/PenguinUartHelloTop.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/PenguinScalarUartHelloTop.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/PenguinScalarDecoder.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/PenguinScalarRegfile.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/PenguinScalarAlu.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/PenguinScalarBranchUnit.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/PenguinScalarLsu.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/PenguinScalarController.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/PenguinPreliminaryVpu.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/PenguinScalarCore.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/penguin_scalar_uart_vadd_program_init.vh \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/scalar/penguin_scalar_defs.vh \
}

# active synthesis top
set_property top $top_name $sources_fs
set_property top $top_name $sim_fs

# simulation sources

# update compile order
update_compile_order -fileset sources_1

close_project
