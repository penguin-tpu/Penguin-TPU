from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RTL_DIR = ROOT / "rtl" / "penguin_tpu"
SCALAR_DIR = RTL_DIR / "scalar"
COCOTB_DIR = ROOT / "tests" / "cocotb"


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_scalar_uart_hello_cocotb() -> None:
    verilog_sources = [
        COCOTB_DIR / "ClockingWizard.v",
        SCALAR_DIR / "PenguinScalarDecoder.v",
        SCALAR_DIR / "PenguinScalarRegfile.v",
        SCALAR_DIR / "PenguinScalarAlu.v",
        SCALAR_DIR / "PenguinScalarBranchUnit.v",
        SCALAR_DIR / "PenguinScalarLsu.v",
        SCALAR_DIR / "PenguinScalarController.v",
        SCALAR_DIR / "PenguinScalarCore.v",
        RTL_DIR / "UartTx.v",
        RTL_DIR / "UartRx.v",
        RTL_DIR / "Uart.v",
        RTL_DIR / "PenguinScalarUartHelloTop.v",
    ]

    parameters = {
        "clk_freq_hz": 4_000,
        "baud_rate": 200,
        "cycle_counter_increment": 25_000,
    }

    sim_build = ROOT / ".pytest_sim_build" / "scalar_uart_hello_verilator"
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
            "TOPLEVEL": "PenguinScalarUartHelloTop",
            "MODULE": "tb_scalar_uart_hello",
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
    env.update({f"PARAM_{name}": str(value) for name, value in parameters.items()})

    subprocess.run(
        ["make", "-f", str(Path(makefiles_dir) / "Makefile.sim")],
        cwd=ROOT,
        env=env,
        check=True,
    )
