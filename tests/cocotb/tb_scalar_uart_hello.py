from __future__ import annotations

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge
from cocotb.utils import get_sim_time


CLK_FREQ_HZ = int(os.environ["PARAM_clk_freq_hz"])
BAUD_RATE = int(os.environ["PARAM_baud_rate"])
CYCLE_COUNTER_INCREMENT = int(os.environ["PARAM_cycle_counter_increment"])
CLOCK_PERIOD_NS = 10
CORE_CLK_FREQ_HZ = CLK_FREQ_HZ // 2
UART_PRESCALE = (CORE_CLK_FREQ_HZ + (BAUD_RATE * 4)) // (BAUD_RATE * 8)
UART_BIT_CYCLES = UART_PRESCALE * 8
MESSAGE = b"vadd=4040\r\n"
MESSAGE_PERIOD_COUNTER_TICKS = 50_000_000
MESSAGE_PERIOD_CYCLES = MESSAGE_PERIOD_COUNTER_TICKS // CYCLE_COUNTER_INCREMENT
assert MESSAGE_PERIOD_COUNTER_TICKS % CYCLE_COUNTER_INCREMENT == 0


async def read_uart_byte(dut) -> tuple[int, int]:
    await FallingEdge(dut.uart_rx_out)
    start_cycle = int(get_sim_time("ns")) // (CLOCK_PERIOD_NS * 2)
    await ClockCycles(dut.clock, UART_BIT_CYCLES + (UART_BIT_CYCLES // 2))

    value = 0
    for bit_index in range(8):
        value |= int(dut.uart_rx_out.value) << bit_index
        await ClockCycles(dut.clock, UART_BIT_CYCLES)

    assert int(dut.uart_rx_out.value) == 1
    return value, start_cycle


async def read_uart_message(dut) -> tuple[bytes, int]:
    decoded = bytearray()
    first_start_cycle: int | None = None

    for _ in range(len(MESSAGE)):
        value, start_cycle = await read_uart_byte(dut)
        if first_start_cycle is None:
            first_start_cycle = start_cycle
        decoded.append(value)

    assert first_start_cycle is not None
    return bytes(decoded), first_start_cycle


@cocotb.test()
async def scalar_core_prints_uart_mmio_message(dut) -> None:
    cocotb.start_soon(Clock(dut.sys_clk_i, CLOCK_PERIOD_NS, units="ns").start())

    dut.cpu_resetn.value = 0
    dut.uart_tx_in.value = 1
    await ClockCycles(dut.sys_clk_i, 5)
    dut.cpu_resetn.value = 1

    first_message, first_start_cycle = await read_uart_message(dut)
    second_message, second_start_cycle = await read_uart_message(dut)
    third_message, third_start_cycle = await read_uart_message(dut)

    assert first_message == MESSAGE
    assert second_message == MESSAGE
    assert third_message == MESSAGE
    assert (second_start_cycle - first_start_cycle) == MESSAGE_PERIOD_CYCLES
    assert (third_start_cycle - second_start_cycle) == MESSAGE_PERIOD_CYCLES
    assert int(dut.scalar_halted.value) == 0
