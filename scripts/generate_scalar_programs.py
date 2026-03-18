"""Generate checked-in scalar test programs as assembly source."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from penguin_model.memory import DRAM_BASE, VMEM_BASE

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOT = REPO_ROOT / "tests" / "vectors" / "programs" / "scalar"

FAIL_REG = 31
SCRATCH_REG = 30


@dataclass
class AssemblyBuilder:
    lines: list[str] = field(default_factory=list)

    def comment(self, text: str) -> None:
        self.lines.append(f"# {text}")

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")

    def empty(self, mnemonic: str) -> None:
        self.lines.append(f"    {mnemonic}")

    def nop(self) -> None:
        self.lines.append("    nop")

    def delay_slots(self) -> None:
        self.nop()
        self.nop()

    def li(self, rd: int, value: int | str) -> None:
        self.lines.append(f"    li x{rd}, {_format_expr(value)}")

    def r(self, mnemonic: str, *, rd: int, rs1: int, rs2: int) -> None:
        self.lines.append(f"    {mnemonic} x{rd}, x{rs1}, x{rs2}")

    def i(self, mnemonic: str, *, rd: int, rs1: int, imm: int | str) -> None:
        if mnemonic in {"slb", "slh", "slw", "slbu", "slhu", "seld"}:
            self.lines.append(f"    {mnemonic} x{rd}, {_format_expr(imm)}(x{rs1})")
            return
        self.lines.append(f"    {mnemonic} x{rd}, x{rs1}, {_format_expr(imm)}")

    def s(self, mnemonic: str, *, rs1: int, rs2: int, imm: int | str) -> None:
        self.lines.append(f"    {mnemonic} x{rs2}, {_format_expr(imm)}(x{rs1})")

    def u(self, mnemonic: str, *, rd: int, imm: int | str) -> None:
        self.lines.append(f"    {mnemonic} x{rd}, {_format_expr(imm)}")

    def branch(self, mnemonic: str, *, rs1: int, rs2: int, target: str) -> None:
        self.lines.append(f"    {mnemonic} x{rs1}, x{rs2}, {target}")

    def jal(self, *, rd: int, target: str) -> None:
        self.lines.append(f"    sjal x{rd}, {target}")

    def dma(self, mnemonic: str, *, dram_rs: int, vmem_rs: int, size_rs: int) -> None:
        self.lines.append(f"    {mnemonic} x{dram_rs}, x{vmem_rs}, x{size_rs}")

    def emit(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def _format_expr(value: int | str) -> str:
    if isinstance(value, str):
        return value
    if value < 0:
        return str(value)
    return f"0x{value:08x}"


def _expect_reg_eq(builder: AssemblyBuilder, *, reg: int, expected: int, fail_code: int = 1) -> None:
    builder.li(SCRATCH_REG, expected)
    builder.branch("sbne", rs1=reg, rs2=SCRATCH_REG, target=f"fail_{fail_code}")


def _expect_reg_eq_label(
    builder: AssemblyBuilder, *, reg: int, label: str, fail_code: int = 1, offset: int = 0
) -> None:
    offset_text = f" + {offset}" if offset >= 0 else f" - {abs(offset)}"
    builder.li(SCRATCH_REG, f"{label}{offset_text}" if offset else label)
    builder.branch("sbne", rs1=reg, rs2=SCRATCH_REG, target=f"fail_{fail_code}")


def _add_fail_handlers(builder: AssemblyBuilder, fail_codes: range) -> None:
    for fail_code in fail_codes:
        builder.label(f"fail_{fail_code}")
        builder.li(FAIL_REG, fail_code)
        builder.jal(rd=0, target="done")
        builder.delay_slots()


def _finish_program(builder: AssemblyBuilder) -> str:
    builder.jal(rd=0, target="done")
    builder.delay_slots()
    _add_fail_handlers(builder, range(1, 2))
    builder.label("done")
    return builder.emit()


def _directed_u_case(mnemonic: str, imm: int, expected: int) -> str:
    builder = AssemblyBuilder()
    builder.u(mnemonic, rd=3, imm=imm)
    _expect_reg_eq(builder, reg=3, expected=expected)
    return _finish_program(builder)


def _directed_r_case(mnemonic: str, lhs: int, rhs: int, expected: int) -> str:
    builder = AssemblyBuilder()
    builder.li(1, lhs)
    builder.li(2, rhs)
    builder.r(mnemonic, rd=3, rs1=1, rs2=2)
    _expect_reg_eq(builder, reg=3, expected=expected)
    return _finish_program(builder)


def _directed_i_case(mnemonic: str, lhs: int, imm: int, expected: int) -> str:
    builder = AssemblyBuilder()
    builder.li(1, lhs)
    builder.i(mnemonic, rd=3, rs1=1, imm=imm)
    _expect_reg_eq(builder, reg=3, expected=expected)
    return _finish_program(builder)


def _directed_branch_case(mnemonic: str, lhs: int, rhs: int, taken: bool) -> str:
    builder = AssemblyBuilder()
    builder.li(1, lhs)
    builder.li(2, rhs)
    if taken:
        builder.branch(mnemonic, rs1=1, rs2=2, target="pass_target")
        builder.delay_slots()
        builder.jal(rd=0, target="fail_1")
        builder.delay_slots()
        builder.label("pass_target")
    else:
        builder.branch(mnemonic, rs1=1, rs2=2, target="fail_1")
        builder.delay_slots()
    return _finish_program(builder)


def _directed_jal_case() -> str:
    builder = AssemblyBuilder()
    builder.jal(rd=10, target="target")
    builder.delay_slots()
    builder.label("link")
    builder.jal(rd=0, target="fail_1")
    builder.delay_slots()
    builder.label("target")
    _expect_reg_eq_label(builder, reg=10, label="link")
    return _finish_program(builder)


def _directed_jalr_case() -> str:
    builder = AssemblyBuilder()
    builder.li(12, "target + 1")
    builder.i("sjalr", rd=11, rs1=12, imm=0)
    builder.delay_slots()
    builder.label("link")
    builder.jal(rd=0, target="fail_1")
    builder.delay_slots()
    builder.label("target")
    _expect_reg_eq_label(builder, reg=11, label="link")
    return _finish_program(builder)


def _directed_load_case() -> str:
    builder = AssemblyBuilder()
    builder.li(1, VMEM_BASE + 0x40)
    builder.i("slw", rd=3, rs1=1, imm=0)
    _expect_reg_eq(builder, reg=3, expected=3)
    return _finish_program(builder)


def _directed_store_case() -> str:
    builder = AssemblyBuilder()
    builder.li(1, VMEM_BASE + 0x60)
    builder.li(2, 8)
    builder.s("ssw", rs1=1, rs2=2, imm=0)
    builder.i("slw", rd=3, rs1=1, imm=0)
    _expect_reg_eq(builder, reg=3, expected=8)
    return _finish_program(builder)


def _directed_x0_load_case() -> str:
    builder = AssemblyBuilder()
    builder.li(1, VMEM_BASE + 0x48)
    builder.i("slw", rd=0, rs1=1, imm=0)
    _expect_reg_eq(builder, reg=0, expected=0)
    return _finish_program(builder)


def _directed_sfence_case() -> str:
    builder = AssemblyBuilder()
    builder.empty("sfence")
    return _finish_program(builder)


def _reduction_program(*, src_addr: int, out_addr: int, count: int) -> str:
    builder = AssemblyBuilder()
    builder.li(1, src_addr)
    builder.li(2, count)
    builder.li(3, 0)
    builder.li(5, out_addr)
    builder.label("loop")
    builder.i("slw", rd=4, rs1=1, imm=0)
    builder.r("sadd", rd=3, rs1=3, rs2=4)
    builder.i("saddi", rd=1, rs1=1, imm=4)
    builder.i("saddi", rd=2, rs1=2, imm=-1)
    builder.branch("sbne", rs1=2, rs2=0, target="loop")
    builder.delay_slots()
    builder.s("ssw", rs1=5, rs2=3, imm=0)
    return builder.emit()


def _address_generation_program(
    *, base_addr: int, out_addr: int, rows: int, cols: int, row_stride: int
) -> str:
    builder = AssemblyBuilder()
    builder.li(1, out_addr)
    builder.li(2, base_addr)
    builder.li(3, rows)
    builder.li(7, row_stride)
    builder.label("row_loop")
    builder.li(4, cols)
    builder.i("saddi", rd=5, rs1=2, imm=0)
    builder.label("col_loop")
    builder.s("ssw", rs1=1, rs2=5, imm=0)
    builder.i("saddi", rd=1, rs1=1, imm=4)
    builder.i("saddi", rd=5, rs1=5, imm=4)
    builder.i("saddi", rd=4, rs1=4, imm=-1)
    builder.branch("sbne", rs1=4, rs2=0, target="col_loop")
    builder.delay_slots()
    builder.r("sadd", rd=2, rs1=2, rs2=7)
    builder.i("saddi", rd=3, rs1=3, imm=-1)
    builder.branch("sbne", rs1=3, rs2=0, target="row_loop")
    builder.delay_slots()
    return builder.emit()


def _copy_and_checksum_program(
    *, src_addr: int, dst_addr: int, checksum_addr: int, count: int
) -> str:
    builder = AssemblyBuilder()
    builder.li(1, src_addr)
    builder.li(2, dst_addr)
    builder.li(3, count)
    builder.li(4, 0)
    builder.li(5, checksum_addr)
    builder.label("loop")
    builder.i("slw", rd=6, rs1=1, imm=0)
    builder.s("ssw", rs1=2, rs2=6, imm=0)
    builder.r("sadd", rd=4, rs1=4, rs2=6)
    builder.i("saddi", rd=1, rs1=1, imm=4)
    builder.i("saddi", rd=2, rs1=2, imm=4)
    builder.i("saddi", rd=3, rs1=3, imm=-1)
    builder.branch("sbne", rs1=3, rs2=0, target="loop")
    builder.delay_slots()
    builder.s("ssw", rs1=5, rs2=4, imm=0)
    return builder.emit()


def _dma_stage_and_reduce_program(*, src_addr: int, staged_addr: int, out_addr: int, count: int) -> str:
    builder = AssemblyBuilder()
    builder.li(10, src_addr)
    builder.li(11, staged_addr)
    builder.li(12, count * 4)
    builder.dma("dma.load.ch0", dram_rs=10, vmem_rs=11, size_rs=12)
    builder.empty("dma.wait.ch0")
    builder.lines.extend(_reduction_program(src_addr=staged_addr, out_addr=out_addr, count=count).splitlines())
    return builder.emit()


def _dma_overlap_program() -> str:
    builder = AssemblyBuilder()
    builder.li(1, DRAM_BASE + 0x100)
    builder.li(2, VMEM_BASE + 0x100)
    builder.li(3, 32)
    builder.li(20, 0)
    builder.dma("dma.load.ch0", dram_rs=1, vmem_rs=2, size_rs=3)
    for _ in range(9):
        builder.i("saddi", rd=20, rs1=20, imm=1)
    builder.empty("dma.wait.ch0")
    return builder.emit()


def _model_core_sjal_delay_slots() -> str:
    builder = AssemblyBuilder()
    builder.jal(rd=4, target="target")
    builder.li(1, 11)
    builder.li(2, 22)
    builder.li(3, 99)
    builder.label("target")
    builder.li(5, 55)
    return builder.emit()


def _model_core_sjalr_delay_slots() -> str:
    builder = AssemblyBuilder()
    builder.li(1, "target")
    builder.i("sjalr", rd=5, rs1=1, imm=1)
    builder.li(2, 2)
    builder.li(3, 3)
    builder.li(4, 99)
    builder.li(6, 99)
    builder.li(7, 99)
    builder.label("target")
    builder.li(8, 8)
    return builder.emit()


def _model_younger_control_transfer() -> str:
    builder = AssemblyBuilder()
    builder.jal(rd=1, target="older_target")
    builder.jal(rd=2, target="younger_target")
    builder.li(3, 3)
    builder.li(4, 4)
    builder.li(5, 5)
    builder.li(6, 6)
    builder.label("older_target")
    builder.li(7, 7)
    builder.label("younger_target")
    builder.li(8, 8)
    return builder.emit()


def _model_vmem_sum_loop() -> str:
    builder = AssemblyBuilder()
    builder.li(1, VMEM_BASE + 0x40)
    builder.li(2, 4)
    builder.li(3, 0)
    builder.label("loop")
    builder.i("slw", rd=4, rs1=1, imm=0)
    builder.r("sadd", rd=3, rs1=3, rs2=4)
    builder.i("saddi", rd=1, rs1=1, imm=4)
    builder.i("saddi", rd=2, rs1=2, imm=-1)
    builder.branch("sbne", rs1=2, rs2=0, target="loop")
    builder.nop()
    builder.nop()
    builder.s("ssw", rs1=0, rs2=3, imm=VMEM_BASE + 0x80)
    return builder.emit()


def _model_dma_load_wait() -> str:
    builder = AssemblyBuilder()
    builder.li(1, DRAM_BASE + 0x100)
    builder.li(2, VMEM_BASE + 0x80)
    builder.li(3, 32)
    builder.dma("dma.load.ch0", dram_rs=1, vmem_rs=2, size_rs=3)
    builder.empty("dma.wait.ch0")
    return builder.emit()


def _model_dma_store_wait() -> str:
    builder = AssemblyBuilder()
    builder.li(1, DRAM_BASE + 0x180)
    builder.li(2, VMEM_BASE + 0x40)
    builder.li(3, 32)
    builder.dma("dma.store.ch2", dram_rs=1, vmem_rs=2, size_rs=3)
    builder.empty("dma.wait.ch2")
    return builder.emit()


def _model_dma_requires_wait() -> str:
    builder = AssemblyBuilder()
    builder.li(1, DRAM_BASE + 0x100)
    builder.li(2, VMEM_BASE + 0x20)
    builder.li(3, 32)
    builder.dma("dma.load.ch0", dram_rs=1, vmem_rs=2, size_rs=3)
    builder.i("slw", rd=4, rs1=2, imm=0)
    builder.empty("dma.wait.ch0")
    builder.i("slw", rd=5, rs1=2, imm=0)
    return builder.emit()


def _model_salu_progress_while_dma() -> str:
    builder = AssemblyBuilder()
    builder.li(1, DRAM_BASE + 0x100)
    builder.li(2, VMEM_BASE + 0x20)
    builder.li(3, 32)
    builder.dma("dma.load.ch0", dram_rs=1, vmem_rs=2, size_rs=3)
    builder.li(6, 1)
    for _ in range(9):
        builder.i("saddi", rd=6, rs1=6, imm=1)
    builder.empty("dma.wait.ch0")
    builder.i("slw", rd=7, rs1=2, imm=0)
    return builder.emit()


def _model_dma_channel_busy() -> str:
    builder = AssemblyBuilder()
    builder.li(1, DRAM_BASE + 0x100)
    builder.li(2, VMEM_BASE + 0x20)
    builder.li(3, 32)
    builder.dma("dma.load.ch1", dram_rs=1, vmem_rs=2, size_rs=3)
    builder.dma("dma.load.ch1", dram_rs=1, vmem_rs=2, size_rs=3)
    return builder.emit()


def _model_dma_wait_noop() -> str:
    builder = AssemblyBuilder()
    builder.empty("dma.wait.ch3")
    return builder.emit()


def _model_dma_channels_independent() -> str:
    builder = AssemblyBuilder()
    builder.li(1, DRAM_BASE + 0x100)
    builder.li(2, VMEM_BASE + 0x40)
    builder.li(3, 32)
    builder.dma("dma.load.ch0", dram_rs=1, vmem_rs=2, size_rs=3)
    builder.li(4, DRAM_BASE + 0x120)
    builder.li(5, VMEM_BASE + 0x80)
    builder.dma("dma.load.ch1", dram_rs=4, vmem_rs=5, size_rs=3)
    builder.empty("dma.wait.ch1")
    builder.empty("dma.wait.ch0")
    return builder.emit()


def _model_misaligned_load() -> str:
    builder = AssemblyBuilder()
    builder.li(1, 2)
    builder.i("slw", rd=2, rs1=1, imm=0)
    return builder.emit()


def _model_misaligned_store() -> str:
    builder = AssemblyBuilder()
    builder.li(1, 2)
    builder.li(2, 0x1234)
    builder.s("ssw", rs1=1, rs2=2, imm=0)
    return builder.emit()


def _model_misaligned_jump_target() -> str:
    builder = AssemblyBuilder()
    builder.jal(rd=0, target="target + 2")
    builder.label("target")
    return builder.emit()


def _model_misaligned_branch_target() -> str:
    builder = AssemblyBuilder()
    builder.li(1, 1)
    builder.branch("sbeq", rs1=1, rs2=1, target="target + 2")
    builder.li(2, 99)
    builder.li(3, 77)
    builder.label("target")
    return builder.emit()


def _model_reset_dma_inflight() -> str:
    builder = AssemblyBuilder()
    builder.li(1, DRAM_BASE + 0x100)
    builder.li(2, VMEM_BASE + 0x40)
    builder.li(3, 32)
    builder.dma("dma.load.ch0", dram_rs=1, vmem_rs=2, size_rs=3)
    return builder.emit()


def _model_step_limit_loop() -> str:
    builder = AssemblyBuilder()
    builder.label("loop")
    builder.jal(rd=0, target="loop")
    builder.nop()
    builder.nop()
    return builder.emit()


def _model_trace_program() -> str:
    builder = AssemblyBuilder()
    builder.li(1, DRAM_BASE + 0x20)
    builder.li(2, VMEM_BASE + 0x80)
    builder.li(3, 32)
    builder.dma("dma.load.ch0", dram_rs=1, vmem_rs=2, size_rs=3)
    builder.empty("dma.wait.ch0")
    builder.i("slw", rd=4, rs1=2, imm=0)
    builder.s("ssw", rs1=0, rs2=4, imm=VMEM_BASE + 0x90)
    return builder.emit()


def _example_scalar_matmul() -> str:
    builder = AssemblyBuilder()
    builder.li(10, DRAM_BASE + 0x100)
    builder.li(11, VMEM_BASE + 0x40)
    builder.li(12, 32)
    builder.dma("dma.load.ch0", dram_rs=10, vmem_rs=11, size_rs=12)
    builder.li(1, VMEM_BASE + 0x40)
    builder.li(2, 4)
    builder.li(3, 0)
    builder.empty("dma.wait.ch0")
    builder.label("loop")
    builder.i("slw", rd=4, rs1=1, imm=0)
    builder.r("sadd", rd=3, rs1=3, rs2=4)
    builder.i("saddi", rd=1, rs1=1, imm=4)
    builder.i("saddi", rd=2, rs1=2, imm=-1)
    builder.branch("sbne", rs1=2, rs2=0, target="loop")
    builder.s("ssw", rs1=0, rs2=3, imm=VMEM_BASE + 0x80)
    return builder.emit()


def _write_program(relative_path: str, source: str) -> None:
    destination = PROGRAM_ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "# Generated by scripts/generate_scalar_programs.py\n\n" + source,
    )


def main() -> int:
    directed_u_cases = [
        ("directed/u_sauipc.S", _directed_u_case("sauipc", 1, 0x0000_1000)),
        ("directed/u_slui.S", _directed_u_case("slui", 0x12345, 0x1234_5000)),
    ]
    directed_r_cases = [
        ("directed/r_sadd.S", _directed_r_case("sadd", 7, 3, 10)),
        ("directed/r_ssub.S", _directed_r_case("ssub", 7, 3, 4)),
        ("directed/r_sslt.S", _directed_r_case("sslt", -16, 3, 1)),
        ("directed/r_ssltu.S", _directed_r_case("ssltu", 3, -16, 1)),
        ("directed/r_sxor.S", _directed_r_case("sxor", 7, 3, 4)),
        ("directed/r_sor.S", _directed_r_case("sor", 7, 3, 7)),
        ("directed/r_sand.S", _directed_r_case("sand", 7, 3, 3)),
        ("directed/r_ssll.S", _directed_r_case("ssll", 3, 3, 24)),
        ("directed/r_ssrl.S", _directed_r_case("ssrl", -16, 3, 0x1FFF_FFFE)),
        ("directed/r_ssra.S", _directed_r_case("ssra", -16, 3, 0xFFFF_FFFE)),
    ]
    directed_i_cases = [
        ("directed/i_saddi.S", _directed_i_case("saddi", 7, -2, 5)),
        ("directed/i_sslti.S", _directed_i_case("sslti", -16, -8, 1)),
        ("directed/i_ssltiu.S", _directed_i_case("ssltiu", 3, -1, 1)),
        ("directed/i_sxori.S", _directed_i_case("sxori", 7, 0xF, 8)),
        ("directed/i_sori.S", _directed_i_case("sori", 3, 0x8, 11)),
        ("directed/i_sandi.S", _directed_i_case("sandi", 7, 0x6, 6)),
        ("directed/i_sslli.S", _directed_i_case("sslli", 3, 4, 48)),
        ("directed/i_ssrli.S", _directed_i_case("ssrli", -16, 4, 0x0FFF_FFFF)),
        ("directed/i_ssrai.S", _directed_i_case("ssrai", -16, 4, 0xFFFF_FFFF)),
    ]
    directed_branch_cases = [
        ("directed/branch_sbeq_taken.S", _directed_branch_case("sbeq", 5, 5, True)),
        ("directed/branch_sbeq_not_taken.S", _directed_branch_case("sbeq", 5, 6, False)),
        ("directed/branch_sbne_taken.S", _directed_branch_case("sbne", 5, 6, True)),
        ("directed/branch_sbne_not_taken.S", _directed_branch_case("sbne", 5, 5, False)),
        ("directed/branch_sblt_taken.S", _directed_branch_case("sblt", -1, 1, True)),
        ("directed/branch_sblt_not_taken.S", _directed_branch_case("sblt", 4, -1, False)),
        ("directed/branch_sbge_taken.S", _directed_branch_case("sbge", 7, -3, True)),
        ("directed/branch_sbge_not_taken.S", _directed_branch_case("sbge", -4, 2, False)),
        ("directed/branch_sbltu_taken.S", _directed_branch_case("sbltu", 1, 2, True)),
        ("directed/branch_sbltu_not_taken.S", _directed_branch_case("sbltu", -1, 2, False)),
        ("directed/branch_sbgeu_taken.S", _directed_branch_case("sbgeu", -1, 2, True)),
        ("directed/branch_sbgeu_not_taken.S", _directed_branch_case("sbgeu", 1, -1, False)),
    ]
    directed_misc_cases = [
        ("directed/jump_sjal.S", _directed_jal_case()),
        ("directed/jump_sjalr.S", _directed_jalr_case()),
        ("directed/load.S", _directed_load_case()),
        ("directed/store.S", _directed_store_case()),
        ("directed/x0_load.S", _directed_x0_load_case()),
        ("directed/sfence.S", _directed_sfence_case()),
    ]
    workload_cases = [
        (
            "workloads/reduction.S",
            _reduction_program(
                src_addr=VMEM_BASE + 0x200,
                out_addr=VMEM_BASE + 0x280,
                count=8,
            ),
        ),
        (
            "workloads/address_generation.S",
            _address_generation_program(
                base_addr=VMEM_BASE + 0x400,
                out_addr=VMEM_BASE + 0x500,
                rows=2,
                cols=3,
                row_stride=0x20,
            ),
        ),
        (
            "workloads/copy_and_checksum.S",
            _copy_and_checksum_program(
                src_addr=VMEM_BASE + 0x600,
                dst_addr=VMEM_BASE + 0x700,
                checksum_addr=VMEM_BASE + 0x780,
                count=6,
            ),
        ),
        (
            "workloads/dma_stage_and_reduce.S",
            _dma_stage_and_reduce_program(
                src_addr=DRAM_BASE + 0x100,
                staged_addr=VMEM_BASE + 0x200,
                out_addr=VMEM_BASE + 0x280,
                count=8,
            ),
        ),
    ]
    performance_cases = [
        ("performance/address_generation.S", workload_cases[1][1]),
        ("performance/dma_stage_and_reduce.S", workload_cases[3][1]),
        ("performance/dma_overlap.S", _dma_overlap_program()),
    ]
    model_cases = [
        ("model/core_sjal_delay_slots.S", _model_core_sjal_delay_slots()),
        ("model/core_sjalr_delay_slots.S", _model_core_sjalr_delay_slots()),
        ("model/younger_control_transfer.S", _model_younger_control_transfer()),
        ("model/vmem_sum_loop.S", _model_vmem_sum_loop()),
        ("model/dma_load_wait.S", _model_dma_load_wait()),
        ("model/dma_store_wait.S", _model_dma_store_wait()),
        ("model/dma_requires_wait.S", _model_dma_requires_wait()),
        ("model/salu_progress_while_dma.S", _model_salu_progress_while_dma()),
        ("model/dma_channel_busy.S", _model_dma_channel_busy()),
        ("model/dma_wait_noop.S", _model_dma_wait_noop()),
        ("model/dma_channels_independent.S", _model_dma_channels_independent()),
        ("model/misaligned_load.S", _model_misaligned_load()),
        ("model/misaligned_store.S", _model_misaligned_store()),
        ("model/misaligned_jump_target.S", _model_misaligned_jump_target()),
        ("model/misaligned_branch_target.S", _model_misaligned_branch_target()),
        ("model/reset_dma_inflight.S", _model_reset_dma_inflight()),
        ("model/env_secall.S", "secall\n"),
        ("model/env_sebreak.S", "sebreak\n"),
        ("model/step_limit_loop.S", _model_step_limit_loop()),
        ("model/trace_dma_flow.S", _model_trace_program()),
    ]
    example_cases = [
        ("examples/scalar_matmul.S", _example_scalar_matmul()),
    ]

    for relative_path, source in (
        directed_u_cases
        + directed_r_cases
        + directed_i_cases
        + directed_branch_cases
        + directed_misc_cases
        + workload_cases
        + performance_cases
        + model_cases
        + example_cases
    ):
        _write_program(relative_path, source)

    print(f"Generated {len(list(PROGRAM_ROOT.rglob('*.S')))} scalar assembly programs under {PROGRAM_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
