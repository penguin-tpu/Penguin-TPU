open_project /home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.xpr

# Get the impl_1 run object and its current status
set impl_run [get_runs impl_1]
set impl_status [string tolower [get_property STATUS $impl_run]]

# If the run is currently running, do not try to relaunch it
if { $impl_status eq "running" } {
    puts "impl_1 run is already running; not relaunching."
} else {
    # Reset the run before launching to ensure a clean start
    if { $impl_status ne "notstarted" } {
        reset_run impl_1
    }
    launch_runs impl_1 -to_step write_bitstream -jobs 24
}

wait_on_run impl_1

close_project
