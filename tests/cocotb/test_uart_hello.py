from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RTL_DIR = ROOT / "rtl" / "penguin_tpu"


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_uart_hello_world_cocotb() -> None:
    verilog_sources = [
        RTL_DIR / "uart_tx.v",
        RTL_DIR / "uart_rx.v",
        RTL_DIR / "uart.v",
        RTL_DIR / "penguin_uart_hello_top.v",
    ]

    parameters = {
        "CLK_FREQ_HZ": 1600,
        "BAUD_RATE": 200,
    }

    sim_build = ROOT / ".pytest_sim_build" / "uart_hello_verilator"
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
            "TOPLEVEL": "penguin_uart_hello_top",
            "MODULE": "tb_uart_hello",
            "VERILOG_SOURCES": " ".join(str(path) for path in verilog_sources),
            "SIM_BUILD": str(sim_build),
            "COCOTB_RESULTS_FILE": str(sim_build / "results.xml"),
            "COCOTB_HDL_TIMEUNIT": "1ns",
            "COCOTB_HDL_TIMEPRECISION": "1ps",
            "COMPILE_ARGS": " ".join(
                f"-G{name}={value}" for name, value in parameters.items()
            ),
            "PYTHONPATH": pythonpath,
        }
    )
    env.update({f"PARAM_{name}": str(value) for name, value in parameters.items()})

    subprocess.run(
        ["make", "-f", str(Path(makefiles_dir) / "Makefile.sim")],
        cwd=ROOT,
        env=env,
        check=True,
    )
