create_project VivadoProject /home/tk/Desktop/Penguin-TPU/VivadoProject -part xc7a200tsbg484-1 -force
set_property source_mgmt_mode None [current_project]

# for some reason this generates an error on tcl run.
# skipping this does not seem to affect overall functionality.
# set_property board_part digilentinc.com:nexys_video:part0:1.2 [current_project]
