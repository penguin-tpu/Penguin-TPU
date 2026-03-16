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
set_property top $top_name [get_filesets sources_1]

# Reset synthesis and implementation runs so target/top changes are applied
set synth_run [get_runs synth_1]
set synth_status [string tolower [get_property STATUS $synth_run]]
if { $synth_status ne "notstarted" && $synth_status ne "running" } {
    reset_run synth_1
}

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
