from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COCOTB_DIR = ROOT / "tests" / "cocotb"
RTL_DIR = ROOT / "rtl" / "penguin_tpu"
SCALAR_DIR = RTL_DIR / "scalar"


def scalar_core_verilog_sources() -> list[Path]:
    return [
        COCOTB_DIR / "Bf16Adder.v",
        SCALAR_DIR / "PenguinScalarDecoder.v",
        SCALAR_DIR / "PenguinScalarRegfile.v",
        SCALAR_DIR / "PenguinScalarAlu.v",
        SCALAR_DIR / "PenguinScalarBranchUnit.v",
        SCALAR_DIR / "PenguinScalarLsu.v",
        SCALAR_DIR / "PenguinScalarController.v",
        SCALAR_DIR / "PenguinPreliminaryVpu.v",
        SCALAR_DIR / "PenguinScalarCore.v",
    ]


def scalar_uart_top_verilog_sources() -> list[Path]:
    return [
        COCOTB_DIR / "ClockingWizard.v",
        *scalar_core_verilog_sources(),
        RTL_DIR / "UartTx.v",
        RTL_DIR / "UartRx.v",
        RTL_DIR / "Uart.v",
        RTL_DIR / "PenguinScalarUartHelloTop.v",
    ]
