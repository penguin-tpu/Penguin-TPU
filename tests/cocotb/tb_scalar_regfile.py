from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


@cocotb.test()
async def regfile_preserves_x0_and_updates_written_registers(dut) -> None:
    cocotb.start_soon(Clock(dut.clock, 10, units="ns").start())

    dut.reset.value = 1
    dut.write_enable.value = 0
    dut.rs1_addr.value = 0
    dut.rs2_addr.value = 0
    dut.debug_addr.value = 0
    await ClockCycles(dut.clock, 2)
    dut.reset.value = 0

    dut.write_enable.value = 1
    dut.write_addr.value = 0
    dut.write_data.value = 0xDEADBEEF
    await ClockCycles(dut.clock, 1)

    dut.rs1_addr.value = 0
    await ClockCycles(dut.clock, 1)
    assert int(dut.rs1_data.value) == 0

    dut.write_addr.value = 5
    dut.write_data.value = 0x12345678
    await ClockCycles(dut.clock, 1)

    dut.write_enable.value = 0
    dut.rs1_addr.value = 5
    dut.rs2_addr.value = 0
    dut.debug_addr.value = 5
    await ClockCycles(dut.clock, 1)
    assert int(dut.rs1_data.value) == 0x12345678
    assert int(dut.rs2_data.value) == 0
    assert int(dut.debug_data.value) == 0x12345678
