from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

from penguin_model import EmptyType, Instruction, assemble_text
from penguin_model.scalar_encoding import encode_scalar_instruction


HALT_EBREAK = 6
HALT_LOAD_MISALIGNED = 3

SEBREAK_WORD = encode_scalar_instruction(Instruction("sebreak", EmptyType()))


def encode_program(source: str) -> dict[int, int]:
    program = assemble_text(source, base_address=0)
    return {
        index * 4: encode_scalar_instruction(instruction)
        for index, instruction in enumerate(program.instructions)
    }


async def read_reg(dut, index: int) -> int:
    dut.debug_reg_addr.value = index
    await Timer(1, units="ns")
    return int(dut.debug_reg_data.value)


async def reset_core(dut) -> None:
    dut.enable.value = 0
    dut.reset.value = 1
    dut.imem_rdata.value = 0
    dut.dmem_rdata.value = 0
    dut.debug_reg_addr.value = 0
    await ClockCycles(dut.clock, 2)
    dut.reset.value = 0
    await Timer(1, units="ns")


async def run_program(dut, source: str, *, data_mem: dict[int, int] | None = None, max_cycles: int = 64) -> dict[int, int]:
    imem = encode_program(source)
    memory = {} if data_mem is None else dict(data_mem)

    await reset_core(dut)
    dut.enable.value = 1

    for _ in range(max_cycles):
        await Timer(1, units="ns")
        imem_addr = int(dut.imem_addr.value)
        dmem_addr = int(dut.dmem_addr.value)
        dut.imem_rdata.value = imem.get(imem_addr, SEBREAK_WORD)
        dut.dmem_rdata.value = memory.get(dmem_addr, 0)
        await RisingEdge(dut.clock)
        await Timer(1, units="ns")

        if int(dut.dmem_valid.value) and int(dut.dmem_write.value):
            memory[int(dut.dmem_addr.value)] = int(dut.dmem_wdata.value)

        if int(dut.halted.value):
            return memory

    raise AssertionError("program did not halt within max_cycles")


@cocotb.test()
async def core_executes_arithmetic_load_store_and_delay_slots(dut) -> None:
    cocotb.start_soon(Clock(dut.clock, 10, units="ns").start())

    arithmetic_program = """
        li x1, 7
        saddi x2, x1, 5
        sebreak
    """
    await run_program(dut, arithmetic_program)
    assert await read_reg(dut, 1) == 7
    assert await read_reg(dut, 2) == 12
    assert int(dut.halt_reason.value) == HALT_EBREAK

    load_store_program = """
        li x1, 0x100
        li x2, 0xA5
        sst x2, 0(x1)
        sld x3, 0(x1)
        sebreak
    """
    memory = await run_program(dut, load_store_program)
    assert memory[0x100] == 0xA5
    assert await read_reg(dut, 3) == 0xA5
    assert int(dut.halt_reason.value) == HALT_EBREAK

    delay_slot_program = """
        li x1, 1
        sjal x5, target
        li x2, 2
        li x3, 3
        li x4, 99
    target:
        sebreak
    """
    await run_program(dut, delay_slot_program)
    assert await read_reg(dut, 2) == 2
    assert await read_reg(dut, 3) == 3
    assert await read_reg(dut, 4) == 0
    assert await read_reg(dut, 5) == 8
    assert int(dut.halt_reason.value) == HALT_EBREAK

    younger_redirect_program = """
        sjal x1, older
        sjal x2, younger
        li x3, 33
        li x4, 44
    older:
        li x5, 55
    younger:
        sebreak
    """
    await run_program(dut, younger_redirect_program)
    assert await read_reg(dut, 1) == 4
    assert await read_reg(dut, 2) == 8
    assert await read_reg(dut, 3) == 33
    assert await read_reg(dut, 4) == 44
    assert await read_reg(dut, 5) == 0
    assert int(dut.halt_reason.value) == HALT_EBREAK

    misaligned_load_program = """
        li x1, 0x102
        sld x2, 0(x1)
        sebreak
    """
    await run_program(dut, misaligned_load_program)
    assert await read_reg(dut, 2) == 0
    assert int(dut.halt_reason.value) == HALT_LOAD_MISALIGNED

    vadd_program = """
        li x1, 0x200
        li x2, 0x3F80
        sst x2, 0(x1)
        li x2, 0x4000
        sst x2, 4(x1)
        vadd m2, m0, m1
        sld x3, 8(x1)
        sebreak
    """
    await run_program(dut, vadd_program, max_cycles=96)
    assert await read_reg(dut, 3) == 0x4040
    assert int(dut.halt_reason.value) == HALT_EBREAK
