from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RTL_DIR = ROOT / "rtl" / "penguin_tpu"
SCALAR_DIR = RTL_DIR / "scalar"


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_scalar_mmio_cycle_counter_cocotb() -> None:
    verilog_sources = [
        SCALAR_DIR / "penguin_scalar_decoder.v",
        SCALAR_DIR / "penguin_scalar_regfile.v",
        SCALAR_DIR / "penguin_scalar_alu.v",
        SCALAR_DIR / "penguin_scalar_branch_unit.v",
        SCALAR_DIR / "penguin_scalar_lsu.v",
        SCALAR_DIR / "penguin_scalar_controller.v",
        SCALAR_DIR / "penguin_scalar_core.v",
        RTL_DIR / "uart_tx.v",
        RTL_DIR / "uart_rx.v",
        RTL_DIR / "uart.v",
        RTL_DIR / "penguin_scalar_uart_hello_top.v",
    ]

    parameters = {
        "CLK_FREQ_HZ": 2_000,
        "BAUD_RATE": 200,
        "CYCLE_COUNTER_INCREMENT": 7,
    }

    sim_build = ROOT / ".pytest_sim_build" / "scalar_mmio_counter_verilator"
    shutil.rmtree(sim_build, ignore_errors=True)
    sim_build.mkdir(parents=True, exist_ok=True)

    makefiles_dir = subprocess.check_output(
        ["cocotb-config", "--makefiles"],
        cwd=ROOT,
        text=True,
    ).strip()

    pythonpath = str(Path(__file__).resolve().parent)
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath = pythonpath + os.pathsep + existing_pythonpath

    env = os.environ.copy()
    env.update(
        {
            "SIM": "verilator",
            "TOPLEVEL_LANG": "verilog",
            "TOPLEVEL": "penguin_scalar_uart_hello_top",
            "MODULE": "tb_scalar_mmio_counter",
            "VERILOG_SOURCES": " ".join(str(path) for path in verilog_sources),
            "SIM_BUILD": str(sim_build),
            "COCOTB_RESULTS_FILE": str(sim_build / "results.xml"),
            "COCOTB_HDL_TIMEUNIT": "1ns",
            "COCOTB_HDL_TIMEPRECISION": "1ps",
            "COMPILE_ARGS": " ".join(
                [f"-I{RTL_DIR}", f"-I{SCALAR_DIR}"] + [f"-G{name}={value}" for name, value in parameters.items()]
            ),
            "PYTHONPATH": pythonpath,
        }
    )

    subprocess.run(
        ["make", "-f", str(Path(makefiles_dir) / "Makefile.sim")],
        cwd=ROOT,
        env=env,
        check=True,
    )
