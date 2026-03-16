from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


CLOCK_PERIOD_NS = 10


@cocotb.test()
async def mmio_cycle_counter_resets_and_increments(dut) -> None:
    cocotb.start_soon(Clock(dut.sys_clk_i, CLOCK_PERIOD_NS, units="ns").start())

    dut.cpu_resetn.value = 0
    dut.uart_tx_in.value = 1
    await ClockCycles(dut.sys_clk_i, 4)
    assert int(dut.cycle_counter_reg.value) == 0

    dut.cpu_resetn.value = 1
    await ClockCycles(dut.sys_clk_i, 1)
    assert int(dut.cycle_counter_reg.value) == 7
    await ClockCycles(dut.sys_clk_i, 1)
    assert int(dut.cycle_counter_reg.value) == 14
    await ClockCycles(dut.sys_clk_i, 3)
    assert int(dut.cycle_counter_reg.value) == 35

    dut.cpu_resetn.value = 0
    await ClockCycles(dut.sys_clk_i, 1)
    assert int(dut.cycle_counter_reg.value) == 0


@cocotb.test()
async def mmio_cycle_counter_wraps_modulo_32_bits(dut) -> None:
    cocotb.start_soon(Clock(dut.sys_clk_i, CLOCK_PERIOD_NS, units="ns").start())

    dut.cpu_resetn.value = 0
    dut.uart_tx_in.value = 1
    await ClockCycles(dut.sys_clk_i, 2)
    dut.cpu_resetn.value = 1
    await ClockCycles(dut.sys_clk_i, 1)

    dut.cycle_counter_reg.value = 0xFFFF_FFFC
    await ClockCycles(dut.sys_clk_i, 1)
    assert int(dut.cycle_counter_reg.value) == 3
    await ClockCycles(dut.sys_clk_i, 1)
    assert int(dut.cycle_counter_reg.value) == 10
