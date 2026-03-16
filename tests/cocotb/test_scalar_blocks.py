from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RTL_DIR = ROOT / "rtl" / "penguin_tpu" / "scalar"


def _run_cocotb(top_level: str, module: str, verilog_sources: list[Path], *, compile_args: str = "") -> None:
    sim_build = ROOT / ".pytest_sim_build" / f"{top_level}_verilator"
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
            "TOPLEVEL": top_level,
            "MODULE": module,
            "VERILOG_SOURCES": " ".join(str(path) for path in verilog_sources),
            "COMPILE_ARGS": compile_args,
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


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_scalar_regfile_cocotb() -> None:
    _run_cocotb(
        "penguin_scalar_regfile",
        "tb_scalar_regfile",
        [RTL_DIR / "penguin_scalar_regfile.v"],
    )


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_scalar_alu_cocotb() -> None:
    _run_cocotb(
        "penguin_scalar_alu",
        "tb_scalar_alu",
        [RTL_DIR / "penguin_scalar_alu.v"],
        compile_args=f"-I{RTL_DIR}",
    )


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_scalar_branch_unit_cocotb() -> None:
    _run_cocotb(
        "penguin_scalar_branch_unit",
        "tb_scalar_branch_unit",
        [RTL_DIR / "penguin_scalar_branch_unit.v"],
    )


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_scalar_lsu_cocotb() -> None:
    _run_cocotb(
        "penguin_scalar_lsu",
        "tb_scalar_lsu",
        [RTL_DIR / "penguin_scalar_lsu.v"],
    )
