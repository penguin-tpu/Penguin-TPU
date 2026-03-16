from __future__ import annotations

import cocotb
from cocotb.triggers import Timer


@cocotb.test()
async def branch_unit_generates_targets_and_detects_misalignment(dut) -> None:
    dut.pc.value = 0x100
    dut.rs1_value.value = 0
    dut.imm32.value = 16
    dut.is_branch.value = 1
    dut.is_jal.value = 0
    dut.is_jalr.value = 0
    dut.branch_condition_met.value = 1
    await Timer(1, units="ns")
    assert int(dut.redirect_valid.value) == 1
    assert int(dut.redirect_target.value) == 0x110
    assert int(dut.target_misaligned.value) == 0

    dut.branch_condition_met.value = 0
    await Timer(1, units="ns")
    assert int(dut.redirect_valid.value) == 0

    dut.is_branch.value = 0
    dut.is_jal.value = 1
    dut.imm32.value = 12
    await Timer(1, units="ns")
    assert int(dut.redirect_valid.value) == 1
    assert int(dut.redirect_target.value) == 0x10C
    assert int(dut.target_misaligned.value) == 0

    dut.is_jal.value = 0
    dut.is_jalr.value = 1
    dut.rs1_value.value = 0x203
    dut.imm32.value = 0
    await Timer(1, units="ns")
    assert int(dut.redirect_valid.value) == 1
    assert int(dut.redirect_target.value) == 0x202
    assert int(dut.target_misaligned.value) == 1
