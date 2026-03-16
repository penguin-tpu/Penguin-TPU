open_hw_manager
open_project /home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.xpr

set top_name [get_property top [current_fileset]]
set bit_path [format "/home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.runs/impl_1/%s.bit" $top_name]

connect_hw_server -allow_non_jtag

open_hw_target

set_property PROGRAM.FILE $bit_path [get_hw_devices xc7a200t_0]
current_hw_device [get_hw_devices xc7a200t_0]
refresh_hw_device -update_hw_probes false [lindex [get_hw_devices xc7a200t_0] 0]

set_property PROBES.FILE {} [get_hw_devices xc7a200t_0]
set_property FULL_PROBES.FILE {} [get_hw_devices xc7a200t_0]
set_property PROGRAM.FILE $bit_path [get_hw_devices xc7a200t_0]
program_hw_devices [get_hw_devices xc7a200t_0]

refresh_hw_device [lindex [get_hw_devices xc7a200t_0] 0]

close_hw_manager
close_project
