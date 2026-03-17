from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer


CLOCK_PERIOD_NS = 10
MMIO_BASE_ADDR = 0x0000_0200
STATUS_ADDR = MMIO_BASE_ADDR + 0x80
BF16_ONE = 0x3F80
BF16_TWO = 0x4000
BF16_THREE = 0x4040


async def reset_vpu(dut) -> None:
    dut.reset.value = 1
    dut.execute_vadd.value = 0
    dut.execute_md.value = 0
    dut.execute_ms1.value = 0
    dut.execute_ms2.value = 0
    dut.mmio_valid.value = 0
    dut.mmio_write.value = 0
    dut.mmio_addr.value = 0
    dut.mmio_wdata.value = 0
    await ClockCycles(dut.clock, 2)
    dut.reset.value = 0
    await Timer(1, units="ns")


async def write_mreg(dut, index: int, value: int) -> None:
    dut.mmio_valid.value = 1
    dut.mmio_write.value = 1
    dut.mmio_addr.value = MMIO_BASE_ADDR + (index * 4)
    dut.mmio_wdata.value = value
    await RisingEdge(dut.clock)
    dut.mmio_valid.value = 0
    dut.mmio_write.value = 0
    dut.mmio_addr.value = 0
    dut.mmio_wdata.value = 0
    await Timer(1, units="ns")


async def read_mmio(dut, address: int) -> int:
    dut.mmio_valid.value = 1
    dut.mmio_write.value = 0
    dut.mmio_addr.value = address
    await Timer(1, units="ns")
    value = int(dut.mmio_rdata.value)
    dut.mmio_valid.value = 0
    dut.mmio_addr.value = 0
    return value


@cocotb.test()
async def preliminary_vpu_executes_bf16_vadd_and_exposes_result_over_mmio(dut) -> None:
    cocotb.start_soon(Clock(dut.clock, CLOCK_PERIOD_NS, units="ns").start())

    await reset_vpu(dut)
    await write_mreg(dut, 0, BF16_ONE)
    await write_mreg(dut, 1, BF16_TWO)

    dut.execute_md.value = 2
    dut.execute_ms1.value = 0
    dut.execute_ms2.value = 1
    dut.execute_vadd.value = 1

    await Timer(1, units="ns")
    assert int(dut.execute_stall.value) == 1

    busy_seen = False
    for _ in range(16):
        status_word = await read_mmio(dut, STATUS_ADDR)
        busy_seen |= bool(status_word & 0x1)
        if int(dut.execute_stall.value) == 0:
            break
        await RisingEdge(dut.clock)
    else:
        raise AssertionError("preliminary VPU did not retire vadd within 16 cycles")

    assert busy_seen
    status_word = await read_mmio(dut, STATUS_ADDR)
    assert (status_word & 0x1) == 0
    assert ((status_word >> 1) & 0x1) == 1

    dut.execute_vadd.value = 0
    await RisingEdge(dut.clock)

    result_word = await read_mmio(dut, MMIO_BASE_ADDR + 8)
    assert result_word == BF16_THREE

