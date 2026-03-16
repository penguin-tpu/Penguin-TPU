from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RTL_DIR = ROOT / "rtl" / "penguin_tpu" / "scalar"


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_scalar_decoder_cocotb() -> None:
    sim_build = ROOT / ".pytest_sim_build" / "scalar_decoder_verilator"
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
            "TOPLEVEL": "PenguinScalarDecoder",
            "MODULE": "tb_scalar_decoder",
            "VERILOG_SOURCES": str(RTL_DIR / "PenguinScalarDecoder.v"),
            "COMPILE_ARGS": f"-I{RTL_DIR}",
            "SIM_BUILD": str(sim_build),
            "COCOTB_RESULTS_FILE": str(sim_build / "results.xml"),
            "COCOTB_HDL_TIMEUNIT": "1ns",
            "COCOTB_HDL_TIMEPRECISION": "1ps",
            "PYTHONPATH": pythonpath,
        }
    )

    subprocess.run(
        ["make", "-f", str(Path(makefiles_dir) / "Makefile.sim")],
        cwd=ROOT,
        check=True,
        env=env,
    )
