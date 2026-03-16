# Nexys Video Hello-World Bring-Up

This document records the current FPGA bring-up flow for the minimal UART
hello-world design on the Digilent Nexys Video board.

## Status

As of March 15, 2026, the flow completed successfully on the connected board,
including a rerun through the wrapper script:

- Vivado project creation succeeded
- synthesis, implementation, and bitstream generation succeeded
- FPGA programming succeeded
- the board emitted `Hello World` over the USB UART connection at 115200 baud
- the wrapper script `scripts/vivado/run_hello_world_bringup.sh` also passed

Observed environment details during the successful run:

- Vivado version: `2024.2`
- target part: `xc7a200tsbg484-1`
- programmed device: `xc7a200t_0`
- enumerated USB UART device: `/dev/ttyUSB0`

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
  - `scripts/vivado/run_hello_world_bringup.sh`
- UART validation helper:
  - `scripts/vivado/read_uart_hello.py`

## Run Procedure

Run all commands from the repository root.

Preferred path:

```bash
bash scripts/vivado/run_hello_world_bringup.sh --port /dev/ttyUSB0
```

The wrapper script:

- removes the previous `VivadoProject/` by default
- runs the checked-in Vivado TCL flow in order
- retries device programming if `4_program_device.tcl` fails intermittently
- validates the UART output with `uv run python`

Optional flags:

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

```bash
uv run python scripts/vivado/read_uart_hello.py \
  --port /dev/ttyUSB0 \
  --baud 115200 \
  --timeout 6 \
  --expect "Hello World"
```

If the board enumerates on a different device node, replace `/dev/ttyUSB0`
with the correct path.

## Expected Result

The serial terminal should show repeating output similar to:

```text
Hello World
Hello World
```

The RTL currently emits `Hello World\r\n` once per second.

## Generated Artifacts

The bitstream is produced at:

`VivadoProject/VivadoProject.runs/impl_1/penguin_uart_hello_top.bit`

Useful implementation reports are also written under:

`VivadoProject/VivadoProject.runs/impl_1/`

Key reports include:

- `penguin_uart_hello_top_timing_summary_routed.rpt`
- `penguin_uart_hello_top_utilization_placed.rpt`
- `penguin_uart_hello_top_drc_routed.rpt`

## Notes

- The FPGA top-level IO names are matched to the Nexys Video constraint file:
  `sys_clk_i`, `cpu_resetn`, `uart_tx_in`, and `uart_rx_out`.
- Internal RTL module clock/reset naming follows the repo convention:
  `clock` and `reset`.
- Python-side UART validation in this repo should be run through `uv`, not bare
  `python`.
