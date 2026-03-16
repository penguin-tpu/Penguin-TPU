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
MESSAGE = b"Hello World\r\n"


async def read_uart_byte(dut) -> tuple[int, int]:
    await FallingEdge(dut.uart_txd)
    start_time_ns = int(get_sim_time("ns"))

    await ClockCycles(dut.clk, UART_BIT_CYCLES + (UART_BIT_CYCLES // 2))

    value = 0
    for bit_index in range(8):
        value |= int(dut.uart_txd.value) << bit_index
        await ClockCycles(dut.clk, UART_BIT_CYCLES)

    stop_bit = int(dut.uart_txd.value)
    assert stop_bit == 1, "UART stop bit must be high"

    return start_time_ns, value


@cocotb.test()
async def hello_world_repeats_at_one_hz(dut) -> None:
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD_NS, units="ns").start())

    dut.rst.value = 1
    dut.uart_rxd.value = 1

    await ClockCycles(dut.clk, 5)
    dut.rst.value = 0

    first_start_time_ns = None
    decoded = bytearray()

    for _ in range(len(MESSAGE)):
        start_time_ns, byte = await read_uart_byte(dut)
        if first_start_time_ns is None:
            first_start_time_ns = start_time_ns
        decoded.append(byte)

    assert bytes(decoded) == MESSAGE

    second_start_time_ns, second_first_byte = await read_uart_byte(dut)
    expected_period_ns = CLK_FREQ_HZ * CLOCK_PERIOD_NS

    assert second_first_byte == MESSAGE[0]
    assert first_start_time_ns is not None
    assert second_start_time_ns - first_start_time_ns == expected_period_ns
