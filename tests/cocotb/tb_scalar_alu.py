from __future__ import annotations

import cocotb
from cocotb.triggers import Timer


CASES = [
    (0, 5, 7, 12, 0),
    (1, 7, 5, 2, 0),
    (2, 0xFFFF_FFFF, 0, 1, 0),
    (3, 1, 2, 1, 0),
    (4, 0x55AA, 0x0FF0, 0x5A5A, 0),
    (5, 0x5500, 0x00AA, 0x55AA, 0),
    (6, 0x55AA, 0x0FF0, 0x05A0, 0),
    (7, 3, 2, 12, 0),
    (8, 16, 2, 4, 0),
    (9, 0xFFFF_FFF0, 2, 0xFFFF_FFFC, 0),
    (11, 9, 9, 1, 1),
    (12, 9, 7, 1, 1),
    (13, 0xFFFF_FFFF, 1, 1, 1),
    (14, 4, 4, 1, 1),
    (15, 1, 2, 1, 1),
    (16, 2, 2, 1, 1),
]


@cocotb.test()
async def alu_executes_integer_and_compare_ops(dut) -> None:
    for alu_fn, lhs, rhs, expected_result, expected_compare in CASES:
        dut.alu_fn.value = alu_fn
        dut.lhs.value = lhs
        dut.rhs.value = rhs
        await Timer(1, units="ns")
        assert int(dut.result.value) == expected_result
        assert int(dut.compare_true.value) == expected_compare
