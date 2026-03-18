open_hw_manager
open_project /home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.xpr

set top_name [get_property top [current_fileset]]
set bit_path [format "/home/tk/Desktop/Penguin-TPU/VivadoProject/VivadoProject.runs/impl_1/%s.bit" $top_name]

connect_hw_server -allow_non_jtag
refresh_hw_server

set selected_target ""
foreach hw_target [get_hw_targets *] {
    catch {close_hw_target [current_hw_target]}
    current_hw_target $hw_target
    if {[catch {open_hw_target $hw_target}]} {
        catch {close_hw_target $hw_target}
        continue
    }

    refresh_hw_target [current_hw_target]
    if {[llength [get_hw_devices]] > 0} {
        set selected_target $hw_target
        break
    }

    catch {close_hw_target $hw_target}
}

if {$selected_target eq ""} {
    error "no connected hardware target with a programmable device was found"
}

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
