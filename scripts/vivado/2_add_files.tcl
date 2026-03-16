open_project /home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.xpr

# constraint file
add_files -fileset constrs_1 -norecurse /home/tk/Desktop/Penguin-TPU/rtl/constraints/NexysVideo_Master.xdc

# design sources
add_files { \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/uart.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/uart_rx.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/uart_tx.v \
  /home/tk/Desktop/Penguin-TPU/rtl/penguin_tpu/penguin_uart_hello_top.v\
}

# simulation sources

# update compile order
update_compile_order -fileset sources_1

close_project
