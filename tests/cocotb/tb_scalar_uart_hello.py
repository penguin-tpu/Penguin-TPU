from __future__ import annotations

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge
from cocotb.utils import get_sim_time


CLK_FREQ_HZ = int(os.environ["PARAM_CLK_FREQ_HZ"])
BAUD_RATE = int(os.environ["PARAM_BAUD_RATE"])
CLOCK_PERIOD_NS = 10
UART_PRESCALE = (CLK_FREQ_HZ + (BAUD_RATE * 4)) // (BAUD_RATE * 8)
UART_BIT_CYCLES = UART_PRESCALE * 8
MESSAGE = b"hello, this is penguin"


async def read_uart_byte(dut) -> int:
    await FallingEdge(dut.uart_rx_out)
    await ClockCycles(dut.sys_clk_i, UART_BIT_CYCLES + (UART_BIT_CYCLES // 2))

    value = 0
    for bit_index in range(8):
        value |= int(dut.uart_rx_out.value) << bit_index
        await ClockCycles(dut.sys_clk_i, UART_BIT_CYCLES)

    assert int(dut.uart_rx_out.value) == 1
    return value


@cocotb.test()
async def scalar_core_prints_uart_mmio_message(dut) -> None:
    cocotb.start_soon(Clock(dut.sys_clk_i, CLOCK_PERIOD_NS, units="ns").start())

    dut.cpu_resetn.value = 0
    dut.uart_tx_in.value = 1
    await ClockCycles(dut.sys_clk_i, 5)
    dut.cpu_resetn.value = 1

    decoded = bytearray()
    for _ in range(len(MESSAGE)):
        decoded.append(await read_uart_byte(dut))

    assert bytes(decoded) == MESSAGE
    assert int(dut.scalar_halted.value) == 0
