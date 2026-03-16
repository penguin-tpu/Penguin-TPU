# Nexys Video FPGA Bring-Up

This document records the current FPGA bring-up flow for the checked-in Nexys
Video targets on the Digilent Nexys Video board.

## Status

As of March 16, 2026, the flow supports multiple bitstream targets and has been
validated on the connected board for:

- `uart_hello`
  - top: `penguin_uart_hello_top`
  - UART output: `Hello World`
- `scalar_core`
  - top: `penguin_scalar_uart_hello_top`
  - UART output: `hello, this is penguin`

Observed environment details during the successful run:

- Vivado version: `2024.2`
- target part: `xc7a200tsbg484-1`
- programmed device: `xc7a200t_0`
- enumerated USB UART device: `/dev/ttyUSB0`

Observed scalar-core implementation caveat:

- `penguin_scalar_uart_hello_top` currently routes and programs successfully,
  but does not meet timing on the Nexys Video target in Vivado 2024.2
  (`WNS=-2.787 ns`, `TNS=-2266.791 ns` in the routed timing summary)
- despite the timing violation, the programmed board still emitted the expected
  repeating UART string during the March 16, 2026 validation run

Note: the requested serial device `/dev/ttyUSB2` was not present during this
run. Only `/dev/ttyUSB0` existed, and that port produced the expected UART
output.

## Files

- Vivado TCL flow:
  - `scripts/vivado/1_create_project.tcl`
  - `scripts/vivado/2_add_files.tcl`
  - `scripts/vivado/3_generate_bitstream.tcl`
  - `scripts/vivado/4_program_device.tcl`
- wrapper script:
  - `scripts/vivado/run_fpga_bringup.sh`
  - `scripts/vivado/run_hello_world_bringup.sh` (compatibility wrapper for `uart_hello`)
- target names:
  - `uart_hello`
  - `scalar_core`
  - `sclar_core` (accepted as an alias for `scalar_core`)
- UART validation helper:
  - `scripts/vivado/run_fpga_bringup.sh`
  - `scripts/vivado/read_uart_hello.py`

## Run Procedure

Run all commands from the repository root.

Preferred path:

```bash
bash scripts/vivado/run_fpga_bringup.sh --target uart_hello --port /dev/ttyUSB0
```

For the scalar-core target:

```bash
bash scripts/vivado/run_fpga_bringup.sh --target scalar_core --port /dev/ttyUSB0
```

The wrapper script:

- removes the previous `VivadoProject/` by default
- runs the checked-in Vivado TCL flow in order
- selects the active top-level from `--target`
- retries device programming if `4_program_device.tcl` fails intermittently
- validates the UART output with `uv run python`

The retry path is not theoretical. During the March 16, 2026 `scalar_core`
board run, the first `open_hw_target` attempt failed with "No devices
detected", and the second programming attempt succeeded without changing the
bitstream or the cable setup.

Optional flags:

- `--target uart_hello`
- `--target scalar_core`
- `--skip-clean`
- `--port /dev/ttyUSBX`
- `--baud 115200`
- `--timeout 6`
- `--program-retries 3`

Manual path:

1. Create the Vivado project:

```bash
vivado -mode batch -source scripts/vivado/1_create_project.tcl
```

2. Add RTL and constraint files:

```bash
vivado -mode batch -source scripts/vivado/2_add_files.tcl
```

3. Generate the bitstream:

```bash
vivado -mode batch -source scripts/vivado/3_generate_bitstream.tcl
```

4. Program the board:

```bash
vivado -mode batch -source scripts/vivado/4_program_device.tcl
```

If step 4 fails intermittently, rerun the same command. The current hardware
programming path can be flaky, but a retry is often sufficient.

5. Check the UART output:

For `uart_hello`:

```bash
uv run python scripts/vivado/read_uart_hello.py \
  --port /dev/ttyUSB0 \
  --baud 115200 \
  --timeout 6 \
  --expect "Hello World"
```

For `scalar_core`:

```bash
uv run python scripts/vivado/read_uart_hello.py \
  --port /dev/ttyUSB0 \
  --baud 115200 \
  --timeout 6 \
  --expect "hello, this is penguin"
```

If the board enumerates on a different device node, replace `/dev/ttyUSB0`
with the correct path.

## Expected Result

The serial terminal should show target-dependent output:

```text
Hello World
hello, this is penguin
```

- `uart_hello` emits `Hello World\r\n` once per second.
- `scalar_core` continuously emits `hello, this is penguin` with no separator.

## Generated Artifacts

The bitstream is produced at one of:

- `VivadoProject/VivadoProject.runs/impl_1/penguin_uart_hello_top.bit`
- `VivadoProject/VivadoProject.runs/impl_1/penguin_scalar_uart_hello_top.bit`

Useful implementation reports are also written under:

`VivadoProject/VivadoProject.runs/impl_1/`

Key reports include:

- `penguin_uart_hello_top_timing_summary_routed.rpt`
- `penguin_uart_hello_top_utilization_placed.rpt`
- `penguin_uart_hello_top_drc_routed.rpt`
- `penguin_scalar_uart_hello_top_timing_summary_routed.rpt`
- `penguin_scalar_uart_hello_top_utilization_placed.rpt`
- `penguin_scalar_uart_hello_top_drc_routed.rpt`

## Notes

- The FPGA top-level IO names are matched to the Nexys Video constraint file:
  `sys_clk_i`, `cpu_resetn`, `uart_tx_in`, and `uart_rx_out`.
- Internal RTL module clock/reset naming follows the repo convention:
  `clock` and `reset`.
- Python-side UART validation in this repo should be run through `uv`, not bare
  `python`.
- The scalar-core target is functionally working on the board, but timing
  closure remains open and should be treated as the next hardware-quality task.
