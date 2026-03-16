from __future__ import annotations

import cocotb
from cocotb.triggers import Timer


@cocotb.test()
async def lsu_generates_aligned_requests_and_flags_misalignment(dut) -> None:
    dut.base_addr.value = 0x1000
    dut.store_data.value = 0xCAFE_BABE
    dut.imm32.value = 8
    dut.is_load.value = 1
    dut.is_store.value = 0
    dut.dmem_rdata.value = 0x1234_5678
    await Timer(1, units="ns")
    assert int(dut.dmem_valid.value) == 1
    assert int(dut.dmem_write.value) == 0
    assert int(dut.dmem_addr.value) == 0x1008
    assert int(dut.load_data.value) == 0x12345678
    assert int(dut.load_misaligned.value) == 0

    dut.is_load.value = 0
    dut.is_store.value = 1
    dut.imm32.value = 12
    await Timer(1, units="ns")
    assert int(dut.dmem_valid.value) == 1
    assert int(dut.dmem_write.value) == 1
    assert int(dut.dmem_addr.value) == 0x100C
    assert int(dut.dmem_wdata.value) == 0xCAFE_BABE
    assert int(dut.store_misaligned.value) == 0

    dut.is_store.value = 0
    dut.is_load.value = 1
    dut.imm32.value = 2
    await Timer(1, units="ns")
    assert int(dut.load_misaligned.value) == 1
    assert int(dut.dmem_valid.value) == 0

    dut.is_load.value = 0
    dut.is_store.value = 1
    await Timer(1, units="ns")
    assert int(dut.store_misaligned.value) == 1
    assert int(dut.dmem_valid.value) == 0
