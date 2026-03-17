from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from .verilog_sources import SCALAR_DIR, scalar_core_verilog_sources

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_scalar_core_cocotb() -> None:
    sim_build = ROOT / ".pytest_sim_build" / "scalar_core_verilator"
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
            "TOPLEVEL": "PenguinScalarCore",
            "MODULE": "tb_scalar_core",
            "VERILOG_SOURCES": " ".join(str(path) for path in scalar_core_verilog_sources()),
            "COMPILE_ARGS": f"-I{SCALAR_DIR}",
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
        env=env,
        check=True,
    )
