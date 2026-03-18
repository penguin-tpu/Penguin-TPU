"""Import supported upstream rv32ui riscv-tests into flattened Penguin assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path("/tmp/riscv-tests/isa")
OUTPUT_ROOT = REPO_ROOT / "tests" / "vectors" / "programs" / "scalar" / "riscv_isa"

SUPPORTED_STEMS = [
    "add",
    "addi",
    "and",
    "andi",
    "auipc",
    "beq",
    "bge",
    "bgeu",
    "blt",
    "bltu",
    "bne",
    "jal",
    "jalr",
    "lb",
    "lbu",
    "ld_st",
    "lh",
    "lhu",
    "lui",
    "lw",
    "or",
    "ori",
    "sb",
    "sh",
    "simple",
    "sll",
    "slli",
    "slt",
    "slti",
    "sltiu",
    "sltu",
    "sra",
    "srai",
    "srl",
    "srli",
    "st_ld",
    "sub",
    "sw",
    "xor",
    "xori",
]

UPSTREAM_EXCLUDED_STEMS = {
    "fence_i",
    "ma_data",
}

REGISTER_ALIASES = {
    "zero": "x0",
    "ra": "x1",
    "sp": "x2",
    "gp": "x3",
    "tp": "x4",
    "t0": "x5",
    "t1": "x6",
    "t2": "x7",
    "s0": "x8",
    "fp": "x8",
    "s1": "x9",
    "a0": "x10",
    "a1": "x11",
    "a2": "x12",
    "a3": "x13",
    "a4": "x14",
    "a5": "x15",
    "a6": "x16",
    "a7": "x17",
    "s2": "x18",
    "s3": "x19",
    "s4": "x20",
    "s5": "x21",
    "s6": "x22",
    "s7": "x23",
    "s8": "x24",
    "s9": "x25",
    "s10": "x26",
    "s11": "x27",
    "t3": "x28",
    "t4": "x29",
    "t5": "x30",
    "t6": "x31",
    "TESTNUM": "x3",
}

SCALAR_MNEMONICS = {
    "add": "sadd",
    "addi": "saddi",
    "sub": "ssub",
    "and": "sand",
    "andi": "sandi",
    "or": "sor",
    "ori": "sori",
    "xor": "sxor",
    "xori": "sxori",
    "slt": "sslt",
    "slti": "sslti",
    "sltiu": "ssltiu",
    "sltu": "ssltu",
    "sll": "ssll",
    "slli": "sslli",
    "srl": "ssrl",
    "srli": "ssrli",
    "sra": "ssra",
    "srai": "ssrai",
    "beq": "sbeq",
    "bne": "sbne",
    "blt": "sblt",
    "bge": "sbge",
    "bltu": "sbltu",
    "bgeu": "sbgeu",
    "jal": "sjal",
    "jalr": "sjalr",
    "jr": "sjalr",
    "lui": "slui",
    "auipc": "sauipc",
}

RR_STEMS = {"add", "and", "or", "xor", "slt", "sltu", "sll", "srl", "sra", "sub"}
IMM_STEMS = {"addi", "andi", "ori", "xori", "slti", "sltiu", "slli", "srli", "srai"}
BRANCH_STEMS = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}
LOAD_STEMS = {"lb", "lbu", "lh", "lhu", "lw"}
STORE_STEMS = {"sb", "sh", "sw"}

LOCAL_LABEL_PATTERN = re.compile(r"(?P<num>\d+)(?P<dir>[fb])")
MACRO_PATTERN = re.compile(r"^(?P<name>[A-Z0-9_]+)\((?P<args>.*)\);$")
INCLUDE_ALIAS_PATTERN = re.compile(r'#include\s+"../rv64ui/(?P<stem>[A-Za-z0-9_]+)\.S"')


@dataclass(slots=True)
class TestContext:
    stem: str
    branch_suffix: int = 0

    def label(self, base: str) -> str:
        self.branch_suffix += 1
        return f"{self.stem}_{base}_{self.branch_suffix}"

    def local_label(self, testnum: int, num: str) -> str:
        return f"{self.stem}_test_{testnum}_local_{num}"


def _split_args(text: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def _mask_u32(value: int) -> int:
    return value & 0xFFFF_FFFF


def _eval_num(text: str) -> int:
    return int(text, 0)


def _mask_expr(text: str) -> int:
    return _mask_u32(_eval_num(text))


def _sext_imm(text: str) -> int:
    value = _eval_num(text) & 0xFFF
    if value & 0x800:
        value -= 0x1000
    return value


def _register(name: str) -> str:
    name = name.strip()
    return REGISTER_ALIASES.get(name, name)


def _indent(lines: list[str]) -> list[str]:
    return [line if not line or line.endswith(":") else f"    {line}" for line in lines]


def _memory_operand(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"(.+)\(([^)]+)\)", text)
    if match is None:
        return text
    return f"{match.group(1).strip()}({_register(match.group(2).strip())})"


def _emit_load_imm(rd: str, value: int) -> list[str]:
    rd = _register(rd)
    value = _mask_u32(value)
    if value == 0:
        return [f"saddi {rd}, x0, 0"]
    hi = (value + 0x800) >> 12
    lo = value - (hi << 12)
    if lo >= 0x800:
        lo -= 0x1000
    if hi == 0:
        return [f"saddi {rd}, x0, {lo}"]
    lines = [f"slui {rd}, 0x{hi:05x}"]
    if lo != 0:
        lines.append(f"saddi {rd}, {rd}, {lo}")
    return lines


def _branch(mnemonic: str, rs1: str, rs2: str, label: str) -> list[str]:
    return [
        f"{SCALAR_MNEMONICS[mnemonic]} {_register(rs1)}, {_register(rs2)}, {label}",
        "nop",
        "nop",
    ]


def _jump(label: str, rd: str = "x0") -> list[str]:
    return [
        f"sjal {_register(rd)}, {label}",
        "nop",
        "nop",
    ]


def _jump_reg(rs1: str, imm: str, rd: str = "x0") -> list[str]:
    return [
        f"sjalr {_register(rd)}, {_register(rs1)}, {imm}",
        "nop",
        "nop",
    ]


def _test_case_prelude(testnum: int) -> list[str]:
    return [f"test_{testnum}:", *_emit_load_imm("x3", testnum)]


def _test_case_epilogue(testreg: str, correct: int) -> list[str]:
    return [
        *_emit_load_imm("x7", correct),
        * _branch("bne", testreg, "x7", "fail"),
    ]


def _rr_case(testnum: int, mnemonic: str, result: int, val1: int, val2: int) -> list[str]:
    return [
        *_test_case_prelude(testnum),
        *_emit_load_imm("x11", val1),
        *_emit_load_imm("x12", val2),
        f"{SCALAR_MNEMONICS[mnemonic]} x14, x11, x12",
        *_test_case_epilogue("x14", result),
    ]


def _rr_src1_eq_dest_case(testnum: int, mnemonic: str, result: int, val1: int, val2: int) -> list[str]:
    return [
        *_test_case_prelude(testnum),
        *_emit_load_imm("x11", val1),
        *_emit_load_imm("x12", val2),
        f"{SCALAR_MNEMONICS[mnemonic]} x11, x11, x12",
        *_test_case_epilogue("x11", result),
    ]


def _rr_src2_eq_dest_case(testnum: int, mnemonic: str, result: int, val1: int, val2: int) -> list[str]:
    return [
        *_test_case_prelude(testnum),
        *_emit_load_imm("x11", val1),
        *_emit_load_imm("x12", val2),
        f"{SCALAR_MNEMONICS[mnemonic]} x12, x11, x12",
        *_test_case_epilogue("x12", result),
    ]


def _rr_src12_eq_dest_case(testnum: int, mnemonic: str, result: int, val1: int) -> list[str]:
    return [
        *_test_case_prelude(testnum),
        *_emit_load_imm("x11", val1),
        f"{SCALAR_MNEMONICS[mnemonic]} x11, x11, x11",
        *_test_case_epilogue("x11", result),
    ]


def _rr_dest_bypass_case(
    ctx: TestContext, testnum: int, mnemonic: str, result: int, val1: int, val2: int, nops: int
) -> list[str]:
    loop = ctx.label(f"test_{testnum}_loop")
    return [
        *_test_case_prelude(testnum),
        *_emit_load_imm("x4", 0),
        f"{loop}:",
        *_emit_load_imm("x1", val1),
        *_emit_load_imm("x2", val2),
        f"{SCALAR_MNEMONICS[mnemonic]} x14, x1, x2",
        *(["nop"] * nops),
        "saddi x6, x14, 0",
        "saddi x4, x4, 1",
        "saddi x5, x0, 2",
        *_branch("bne", "x4", "x5", loop),
        *_test_case_epilogue("x6", result),
    ]


def _rr_src_bypass_case(
    ctx: TestContext,
    testnum: int,
    mnemonic: str,
    result: int,
    val1: int,
    val2: int,
    src1_nops: int,
    src2_nops: int,
    reverse: bool,
) -> list[str]:
    loop = ctx.label(f"test_{testnum}_loop")
    lines = [
        *_test_case_prelude(testnum),
        *_emit_load_imm("x4", 0),
        f"{loop}:",
    ]
    if reverse:
        lines.extend(_emit_load_imm("x2", val2))
        lines.extend(["nop"] * src1_nops)
        lines.extend(_emit_load_imm("x1", val1))
        lines.extend(["nop"] * src2_nops)
    else:
        lines.extend(_emit_load_imm("x1", val1))
        lines.extend(["nop"] * src1_nops)
        lines.extend(_emit_load_imm("x2", val2))
        lines.extend(["nop"] * src2_nops)
    lines.extend(
        [
            f"{SCALAR_MNEMONICS[mnemonic]} x14, x1, x2",
            "saddi x4, x4, 1",
            "saddi x5, x0, 2",
            *_branch("bne", "x4", "x5", loop),
            *_test_case_epilogue("x14", result),
        ]
    )
    return lines


def _rr_zero_case(testnum: int, mnemonic: str, result: int, val: int | None, which: str) -> list[str]:
    lines = [*_test_case_prelude(testnum)]
    if which == "src1":
        lines.extend(_emit_load_imm("x1", val or 0))
        lines.append(f"{SCALAR_MNEMONICS[mnemonic]} x2, x0, x1")
        lines.extend(_test_case_epilogue("x2", result))
    elif which == "src2":
        lines.extend(_emit_load_imm("x1", val or 0))
        lines.append(f"{SCALAR_MNEMONICS[mnemonic]} x2, x1, x0")
        lines.extend(_test_case_epilogue("x2", result))
    elif which == "src12":
        lines.append(f"{SCALAR_MNEMONICS[mnemonic]} x1, x0, x0")
        lines.extend(_test_case_epilogue("x1", result))
    else:
        lines.extend(_emit_load_imm("x1", val or 0))
        lines.extend(_emit_load_imm("x2", 0))
        lines.append(f"{SCALAR_MNEMONICS[mnemonic]} x0, x1, x2")
        lines.extend(_test_case_epilogue("x0", 0))
    return lines


def _imm_case(testnum: int, mnemonic: str, result: int, val1: int, imm: int) -> list[str]:
    return [
        *_test_case_prelude(testnum),
        *_emit_load_imm("x13", val1),
        f"{SCALAR_MNEMONICS[mnemonic]} x14, x13, {imm}",
        *_test_case_epilogue("x14", result),
    ]


def _imm_src1_eq_dest_case(testnum: int, mnemonic: str, result: int, val1: int, imm: int) -> list[str]:
    return [
        *_test_case_prelude(testnum),
        *_emit_load_imm("x11", val1),
        f"{SCALAR_MNEMONICS[mnemonic]} x11, x11, {imm}",
        *_test_case_epilogue("x11", result),
    ]


def _imm_dest_bypass_case(
    ctx: TestContext, testnum: int, mnemonic: str, result: int, val1: int, imm: int, nops: int
) -> list[str]:
    loop = ctx.label(f"test_{testnum}_loop")
    return [
        *_test_case_prelude(testnum),
        *_emit_load_imm("x4", 0),
        f"{loop}:",
        *_emit_load_imm("x1", val1),
        f"{SCALAR_MNEMONICS[mnemonic]} x14, x1, {imm}",
        *(["nop"] * nops),
        "saddi x6, x14, 0",
        "saddi x4, x4, 1",
        "saddi x5, x0, 2",
        *_branch("bne", "x4", "x5", loop),
        *_test_case_epilogue("x6", result),
    ]


def _imm_src1_bypass_case(
    ctx: TestContext, testnum: int, mnemonic: str, result: int, val1: int, imm: int, nops: int
) -> list[str]:
    loop = ctx.label(f"test_{testnum}_loop")
    return [
        *_test_case_prelude(testnum),
        *_emit_load_imm("x4", 0),
        f"{loop}:",
        *_emit_load_imm("x1", val1),
        *(["nop"] * nops),
        f"{SCALAR_MNEMONICS[mnemonic]} x14, x1, {imm}",
        "saddi x4, x4, 1",
        "saddi x5, x0, 2",
        *_branch("bne", "x4", "x5", loop),
        *_test_case_epilogue("x14", result),
    ]


def _imm_zero_case(testnum: int, mnemonic: str, result: int, val1: int | None, imm: int, which: str) -> list[str]:
    lines = [*_test_case_prelude(testnum)]
    if which == "zerosrc1":
        lines.append(f"{SCALAR_MNEMONICS[mnemonic]} x1, x0, {imm}")
        lines.extend(_test_case_epilogue("x1", result))
    else:
        lines.extend(_emit_load_imm("x1", val1 or 0))
        lines.append(f"{SCALAR_MNEMONICS[mnemonic]} x0, x1, {imm}")
        lines.extend(_test_case_epilogue("x0", 0))
    return lines


def _data_address(stem: str, symbol: str) -> str:
    if stem in {"lb", "lbu"}:
        offsets = {"tdat": 0x100, "tdat1": 0x100, "tdat2": 0x101, "tdat3": 0x102, "tdat4": 0x103}
    elif stem in {"lh", "lhu"}:
        offsets = {"tdat": 0x100, "tdat1": 0x100, "tdat2": 0x102, "tdat3": 0x104, "tdat4": 0x106}
    elif stem == "lw":
        offsets = {"tdat": 0x100, "tdat1": 0x100, "tdat2": 0x104, "tdat3": 0x108, "tdat4": 0x10C}
    elif stem == "sb":
        offsets = {
            "tdat": 0x100,
            "tdat1": 0x100,
            "tdat2": 0x101,
            "tdat3": 0x102,
            "tdat4": 0x103,
            "tdat5": 0x104,
            "tdat6": 0x105,
            "tdat7": 0x106,
            "tdat8": 0x107,
            "tdat9": 0x108,
            "tdat10": 0x109,
        }
    elif stem == "sh":
        offsets = {
            "tdat": 0x100,
            "tdat1": 0x100,
            "tdat2": 0x102,
            "tdat3": 0x104,
            "tdat4": 0x106,
            "tdat5": 0x108,
            "tdat6": 0x10A,
            "tdat7": 0x10C,
            "tdat8": 0x10E,
            "tdat9": 0x110,
            "tdat10": 0x112,
        }
    elif stem == "sw":
        offsets = {
            "tdat": 0x100,
            "tdat1": 0x100,
            "tdat2": 0x104,
            "tdat3": 0x108,
            "tdat4": 0x10C,
            "tdat5": 0x110,
            "tdat6": 0x114,
            "tdat7": 0x118,
            "tdat8": 0x11C,
            "tdat9": 0x120,
            "tdat10": 0x124,
        }
    elif stem in {"ld_st", "st_ld"}:
        offsets = {"tdat": 0x100}
    else:
        raise ValueError(f"Unknown data layout for {stem}")
    return f"VMEM_BASE + 0x{offsets[symbol]:03x}"


def _load_case(stem: str, testnum: int, load_inst: str, result: int, offset: int, base: str) -> list[str]:
    return [
        *_test_case_prelude(testnum),
        *_emit_load_imm("x15", result),
        f"li x2, {_data_address(stem, base)}",
        f"{load_inst} x14, {offset}(x2)",
        *_test_case_epilogue("x14", result),
    ]


def _store_case(
    stem: str,
    testnum: int,
    load_inst: str,
    store_inst: str,
    result: int,
    offset: int,
    base: str,
) -> list[str]:
    return [
        *_test_case_prelude(testnum),
        f"li x2, {_data_address(stem, base)}",
        *_emit_load_imm("x1", result),
        f"{store_inst} x1, {offset}(x2)",
        f"{load_inst} x14, {offset}(x2)",
        *_test_case_epilogue("x14", result),
    ]


def _load_dest_bypass_case(
    ctx: TestContext, testnum: int, load_inst: str, result: int, offset: int, base: str, nops: int
) -> list[str]:
    loop = ctx.label(f"test_{testnum}_loop")
    return [
        f"test_{testnum}:",
        *_emit_load_imm("x3", testnum),
        *_emit_load_imm("x4", 0),
        f"{loop}:",
        f"li x13, {_data_address(ctx.stem, base)}",
        f"{load_inst} x14, {offset}(x13)",
        *(["nop"] * nops),
        "saddi x6, x14, 0",
        *_emit_load_imm("x7", result),
        *_branch("bne", "x6", "x7", "fail"),
        "saddi x4, x4, 1",
        "saddi x5, x0, 2",
        *_branch("bne", "x4", "x5", loop),
    ]


def _load_src1_bypass_case(
    ctx: TestContext, testnum: int, load_inst: str, result: int, offset: int, base: str, nops: int
) -> list[str]:
    loop = ctx.label(f"test_{testnum}_loop")
    return [
        f"test_{testnum}:",
        *_emit_load_imm("x3", testnum),
        *_emit_load_imm("x4", 0),
        f"{loop}:",
        f"li x13, {_data_address(ctx.stem, base)}",
        *(["nop"] * nops),
        f"{load_inst} x14, {offset}(x13)",
        *_emit_load_imm("x7", result),
        *_branch("bne", "x14", "x7", "fail"),
        "saddi x4, x4, 1",
        "saddi x5, x0, 2",
        *_branch("bne", "x4", "x5", loop),
    ]


def _store_bypass_case(
    ctx: TestContext,
    testnum: int,
    load_inst: str,
    store_inst: str,
    result: int,
    offset: int,
    base: str,
    src1_nops: int,
    src2_nops: int,
    reverse: bool,
) -> list[str]:
    loop = ctx.label(f"test_{testnum}_loop")
    lines = [
        f"test_{testnum}:",
        *_emit_load_imm("x3", testnum),
        *_emit_load_imm("x4", 0),
        f"{loop}:",
    ]
    if reverse:
        lines.append(f"li x2, {_data_address(ctx.stem, base)}")
        lines.extend(["nop"] * src1_nops)
        lines.extend(_emit_load_imm("x1", result))
        lines.extend(["nop"] * src2_nops)
        lines.extend(
            [
                f"{store_inst} x1, {offset}(x2)",
                f"{load_inst} x14, {offset}(x2)",
            ]
        )
    else:
        lines.extend(_emit_load_imm("x13", result))
        lines.extend(["nop"] * src1_nops)
        lines.append(f"li x12, {_data_address(ctx.stem, base)}")
        lines.extend(["nop"] * src2_nops)
        lines.extend(
            [
                f"{store_inst} x13, {offset}(x12)",
                f"{load_inst} x14, {offset}(x12)",
            ]
        )
    lines.extend(
        [
            *_emit_load_imm("x7", result),
            *_branch("bne", "x14", "x7", "fail"),
            "saddi x4, x4, 1",
            "saddi x5, x0, 2",
            *_branch("bne", "x4", "x5", loop),
        ]
    )
    return lines


def _ld_st_bypass_case(
    testnum: int, load_inst: str, store_inst: str, result: int, offset: int, base: str
) -> list[str]:
    addr = _data_address("ld_st", base)
    return [
        f"test_{testnum}:",
        *_emit_load_imm("x3", testnum),
        f"li x2, {addr}",
        *_emit_load_imm("x1", result),
        f"{store_inst} x1, {offset}(x2)",
        f"{load_inst} x14, {offset}(x2)",
        f"{store_inst} x14, {offset}(x2)",
        f"{load_inst} x2, {offset}(x2)",
        *_emit_load_imm("x7", result),
        *_branch("bne", "x2", "x7", "fail"),
    ]


def _st_ld_bypass_case(
    testnum: int, load_inst: str, store_inst: str, result: int, offset: int, base: str
) -> list[str]:
    addr = _data_address("st_ld", base)
    return [
        f"test_{testnum}:",
        *_emit_load_imm("x3", testnum),
        f"li x2, {addr}",
        *_emit_load_imm("x1", result),
        f"{store_inst} x1, {offset}(x2)",
        f"{load_inst} x14, {offset}(x2)",
        *_emit_load_imm("x7", result),
        *_branch("bne", "x14", "x7", "fail"),
    ]


def _branch_taken_case(ctx: TestContext, testnum: int, mnemonic: str, val1: int, val2: int) -> list[str]:
    after_forward = ctx.label(f"test_{testnum}_after_forward")
    after_loop = ctx.label(f"test_{testnum}_after_loop")
    back_target = ctx.label(f"test_{testnum}_back_target")
    return [
        f"test_{testnum}:",
        *_emit_load_imm("x3", testnum),
        *_emit_load_imm("x1", val1),
        *_emit_load_imm("x2", val2),
        *(_branch(mnemonic, "x1", "x2", after_forward)),
        *(_branch("bne", "x0", "x3", "fail")),
        f"{back_target}:",
        *(_branch("bne", "x0", "x3", after_loop)),
        f"{after_forward}:",
        *(_branch(mnemonic, "x1", "x2", back_target)),
        *(_branch("bne", "x0", "x3", "fail")),
        f"{after_loop}:",
    ]


def _branch_not_taken_case(ctx: TestContext, testnum: int, mnemonic: str, val1: int, val2: int) -> list[str]:
    label1 = ctx.label(f"test_{testnum}_label1")
    label2 = ctx.label(f"test_{testnum}_label2")
    label3 = ctx.label(f"test_{testnum}_label3")
    return [
        f"test_{testnum}:",
        *_emit_load_imm("x3", testnum),
        *_emit_load_imm("x1", val1),
        *_emit_load_imm("x2", val2),
        *(_branch(mnemonic, "x1", "x2", label1)),
        *(_branch("bne", "x0", "x3", label2)),
        f"{label1}:",
        *(_branch("bne", "x0", "x3", "fail")),
        f"{label2}:",
        *(_branch(mnemonic, "x1", "x2", label1)),
        f"{label3}:",
    ]


def _branch_bypass_case(
    ctx: TestContext, testnum: int, mnemonic: str, val1: int, val2: int, src1_nops: int, src2_nops: int
) -> list[str]:
    loop = ctx.label(f"test_{testnum}_loop")
    return [
        f"test_{testnum}:",
        *_emit_load_imm("x3", testnum),
        *_emit_load_imm("x4", 0),
        f"{loop}:",
        *_emit_load_imm("x1", val1),
        *(["nop"] * src1_nops),
        *_emit_load_imm("x2", val2),
        *(["nop"] * src2_nops),
        *(_branch(mnemonic, "x1", "x2", "fail")),
        "saddi x4, x4, 1",
        "saddi x5, x0, 2",
        *(_branch("bne", "x4", "x5", loop)),
    ]


def _translate_code_block(ctx: TestContext, testnum: int, block: str) -> list[str]:
    block = re.sub(r"/\*.*?\*/", "", block)
    statements = [part.strip() for part in block.split(";") if part.strip()]
    label_map: dict[str, str] = {}
    for stmt in statements:
        if ":" in stmt:
            label = stmt.split(":", 1)[0].strip()
            if label.isdigit():
                label_map[label] = ctx.local_label(testnum, label)
    lines: list[str] = []
    for stmt in statements:
        label = None
        if ":" in stmt:
            raw_label, remainder = stmt.split(":", 1)
            if raw_label.strip().isdigit():
                label = label_map[raw_label.strip()]
            else:
                label = raw_label.strip()
            stmt = remainder.strip()
            lines.append(f"{label}:")
            if not stmt:
                continue
        for old, new in label_map.items():
            stmt = re.sub(rf"\b{old}([fb])\b", lambda m: new, stmt)
        tokens = stmt.split(None, 1)
        mnemonic = tokens[0]
        operand_text = tokens[1] if len(tokens) > 1 else ""
        operands = [part.strip() for part in operand_text.split(",")] if operand_text else []
        if mnemonic in {"li"}:
            lines.extend(_emit_load_imm(_register(operands[0]), _mask_expr(operands[1])))
            continue
        if mnemonic in {"la", "lla"}:
            target = operands[1]
            if target in {"tdat", "tdat1", "tdat2", "tdat3", "tdat4", "tdat5", "tdat6", "tdat7", "tdat8", "tdat9", "tdat10"}:
                lines.append(f"li {_register(operands[0])}, {_data_address(ctx.stem, target)}")
            else:
                lines.append(f"li {_register(operands[0])}, {target}")
            continue
        if mnemonic == "mv":
            lines.append(f"saddi {_register(operands[0])}, {_register(operands[1])}, 0")
            continue
        if mnemonic == "nop":
            lines.append("nop")
            continue
        if mnemonic == "j":
            lines.extend(_jump(operands[0]))
            continue
        if mnemonic == "jal":
            lines.extend(_jump(operands[1], rd=operands[0]))
            continue
        if mnemonic == "jalr":
            lines.extend(_jump_reg(operands[1], operands[2], rd=operands[0]))
            continue
        if mnemonic == "jr":
            imm = operands[1] if len(operands) > 1 else "0"
            lines.extend(_jump_reg(operands[0], imm))
            continue
        if mnemonic in BRANCH_STEMS:
            lines.extend(_branch(mnemonic, operands[0], operands[1], operands[2]))
            continue
        mapped = SCALAR_MNEMONICS.get(mnemonic, mnemonic)
        if len(operands) == 3:
            lines.append(f"{mapped} {_register(operands[0])}, {_register(operands[1])}, {operands[2] if '(' in operands[1] else _register(operands[2])}")
            continue
        if len(operands) == 2:
            if "(" in operands[1]:
                lines.append(f"{mapped} {_register(operands[0])}, {_memory_operand(operands[1])}")
            else:
                lines.append(f"{mapped} {_register(operands[0])}, {_register(operands[1])}")
            continue
        raise ValueError(f"Unsupported statement in {ctx.stem}: {stmt}")
    return lines


def _rv32_source_lines(stem: str) -> list[str]:
    rv32_path = UPSTREAM_ROOT / "rv32ui" / f"{stem}.S"
    text = rv32_path.read_text()
    match = INCLUDE_ALIAS_PATTERN.search(text)
    if match is not None:
        return (UPSTREAM_ROOT / "rv64ui" / f"{match.group('stem')}.S").read_text().splitlines()
    return text.splitlines()


def _logical_lines(stem: str) -> list[str]:
    lines = _rv32_source_lines(stem)
    in_code = False
    skip_depth = 0
    logical: list[str] = []
    buffer = ""
    for raw in lines:
        raw_stripped = raw.strip()
        if raw_stripped == "RVTEST_CODE_BEGIN":
            in_code = True
            continue
        if not in_code:
            continue
        if raw_stripped == "TEST_PASSFAIL":
            break
        if raw_stripped.startswith("#if __riscv_xlen == 64"):
            skip_depth += 1
            continue
        if raw_stripped == "#endif":
            skip_depth -= 1
            continue
        if skip_depth:
            continue
        if raw_stripped.startswith("#"):
            continue
        stripped = raw.split("#", 1)[0].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(".option") or stripped.startswith(".align"):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        if buffer:
            logical.append((buffer + stripped).strip())
            buffer = ""
        else:
            logical.append(stripped)
    return logical


def _render_rr_file(stem: str) -> str:
    ctx = TestContext(stem)
    body: list[str] = []
    for line in _logical_lines(stem):
        match = MACRO_PATTERN.match(line)
        if match is None:
            continue
        name = match.group("name")
        args = _split_args(match.group("args"))
        if name == "TEST_RR_OP":
            body.extend(_rr_case(int(args[0]), stem, _mask_expr(args[2]), _mask_expr(args[3]), _mask_expr(args[4])))
        elif name == "TEST_RR_SRC1_EQ_DEST":
            body.extend(_rr_src1_eq_dest_case(int(args[0]), stem, _mask_expr(args[2]), _mask_expr(args[3]), _mask_expr(args[4])))
        elif name == "TEST_RR_SRC2_EQ_DEST":
            body.extend(_rr_src2_eq_dest_case(int(args[0]), stem, _mask_expr(args[2]), _mask_expr(args[3]), _mask_expr(args[4])))
        elif name == "TEST_RR_SRC12_EQ_DEST":
            body.extend(_rr_src12_eq_dest_case(int(args[0]), stem, _mask_expr(args[2]), _mask_expr(args[3])))
        elif name == "TEST_RR_DEST_BYPASS":
            body.extend(_rr_dest_bypass_case(ctx, int(args[0]), stem, _mask_expr(args[3]), _mask_expr(args[4]), _mask_expr(args[5]), int(args[1])))
        elif name == "TEST_RR_SRC12_BYPASS":
            body.extend(_rr_src_bypass_case(ctx, int(args[0]), stem, _mask_expr(args[4]), _mask_expr(args[5]), _mask_expr(args[6]), int(args[1]), int(args[2]), reverse=False))
        elif name == "TEST_RR_SRC21_BYPASS":
            body.extend(_rr_src_bypass_case(ctx, int(args[0]), stem, _mask_expr(args[4]), _mask_expr(args[5]), _mask_expr(args[6]), int(args[1]), int(args[2]), reverse=True))
        elif name == "TEST_RR_ZEROSRC1":
            body.extend(_rr_zero_case(int(args[0]), stem, _mask_expr(args[2]), _mask_expr(args[3]), "src1"))
        elif name == "TEST_RR_ZEROSRC2":
            body.extend(_rr_zero_case(int(args[0]), stem, _mask_expr(args[2]), _mask_expr(args[3]), "src2"))
        elif name == "TEST_RR_ZEROSRC12":
            body.extend(_rr_zero_case(int(args[0]), stem, _mask_expr(args[2]), None, "src12"))
        elif name == "TEST_RR_ZERODEST":
            body.extend(_rr_zero_case(int(args[0]), stem, 0, _mask_expr(args[3]), "dest"))
    return _render_program(stem, body)


def _render_imm_file(stem: str) -> str:
    ctx = TestContext(stem)
    body: list[str] = []
    for line in _logical_lines(stem):
        match = MACRO_PATTERN.match(line)
        if match is None:
            continue
        name = match.group("name")
        args = _split_args(match.group("args"))
        if name == "TEST_IMM_OP":
            body.extend(_imm_case(int(args[0]), stem, _mask_expr(args[2]), _mask_expr(args[3]), _sext_imm(args[4])))
        elif name == "TEST_IMM_SRC1_EQ_DEST":
            body.extend(_imm_src1_eq_dest_case(int(args[0]), stem, _mask_expr(args[2]), _mask_expr(args[3]), _sext_imm(args[4])))
        elif name == "TEST_IMM_DEST_BYPASS":
            body.extend(_imm_dest_bypass_case(ctx, int(args[0]), stem, _mask_expr(args[3]), _mask_expr(args[4]), _sext_imm(args[5]), int(args[1])))
        elif name == "TEST_IMM_SRC1_BYPASS":
            body.extend(_imm_src1_bypass_case(ctx, int(args[0]), stem, _mask_expr(args[3]), _mask_expr(args[4]), _sext_imm(args[5]), int(args[1])))
        elif name == "TEST_IMM_ZEROSRC1":
            body.extend(_imm_zero_case(int(args[0]), stem, _mask_expr(args[2]), None, _sext_imm(args[3]), "zerosrc1"))
        elif name == "TEST_IMM_ZERODEST":
            body.extend(_imm_zero_case(int(args[0]), stem, 0, _mask_expr(args[2]), _sext_imm(args[3]), "zerodest"))
    return _render_program(stem, body)


def _render_branch_file(stem: str) -> str:
    ctx = TestContext(stem)
    body: list[str] = [
        f"# Penguin note: upstream {stem}.S contains a no-delay-slot test case that is",
        "# architecturally incompatible with Penguin's two executed delay slots, so the",
        "# imported file keeps the instruction semantic cases and leaves delay-slot",
        "# behavior to the dedicated Penguin control-flow tests.",
        "",
    ]
    for line in _logical_lines(stem):
        match = MACRO_PATTERN.match(line)
        if match is None:
            continue
        name = match.group("name")
        args = _split_args(match.group("args"))
        if name == "TEST_BR2_OP_TAKEN":
            body.extend(_branch_taken_case(ctx, int(args[0]), stem, _mask_expr(args[2]), _mask_expr(args[3])))
        elif name == "TEST_BR2_OP_NOTTAKEN":
            body.extend(_branch_not_taken_case(ctx, int(args[0]), stem, _mask_expr(args[2]), _mask_expr(args[3])))
        elif name == "TEST_BR2_SRC12_BYPASS":
            body.extend(_branch_bypass_case(ctx, int(args[0]), stem, _mask_expr(args[4]), _mask_expr(args[5]), int(args[1]), int(args[2])))
        elif name == "TEST_CASE":
            continue
    return _render_program(stem, body)


def _render_load_file(stem: str) -> str:
    ctx = TestContext(stem)
    body: list[str] = []
    for line in _logical_lines(stem):
        match = MACRO_PATTERN.match(line)
        if match is None:
            continue
        name = match.group("name")
        args = _split_args(match.group("args"))
        if name == "TEST_LD_OP":
            body.extend(_load_case(stem, int(args[0]), args[1], _mask_expr(args[2]), int(args[3], 0), args[4]))
        elif name == "TEST_LD_DEST_BYPASS":
            body.extend(_load_dest_bypass_case(ctx, int(args[0]), args[2], _mask_expr(args[3]), int(args[4], 0), args[5], int(args[1])))
        elif name == "TEST_LD_SRC1_BYPASS":
            body.extend(_load_src1_bypass_case(ctx, int(args[0]), args[2], _mask_expr(args[3]), int(args[4], 0), args[5], int(args[1])))
        elif name == "TEST_CASE":
            body.extend([*_test_case_prelude(int(args[0])), *_translate_code_block(ctx, int(args[0]), args[3]), *_test_case_epilogue(args[1], _mask_expr(args[2]))])
    return _render_program(stem, body)


def _render_store_file(stem: str) -> str:
    ctx = TestContext(stem)
    body: list[str] = []
    trailer: list[str] = []
    for line in _logical_lines(stem):
        match = MACRO_PATTERN.match(line)
        if match is None:
            if line.startswith("li ") or line.startswith("la ") or line.startswith("sb ") or line.startswith("sh "):
                trailer.extend(_translate_code_block(ctx, 999, line))
            continue
        name = match.group("name")
        args = _split_args(match.group("args"))
        if name == "TEST_ST_OP":
            body.extend(_store_case(stem, int(args[0]), args[1], args[2], _mask_expr(args[3]), int(args[4], 0), args[5]))
        elif name == "TEST_ST_SRC12_BYPASS":
            body.extend(_store_bypass_case(ctx, int(args[0]), args[3], args[4], _mask_expr(args[5]), int(args[6], 0), args[7], int(args[1]), int(args[2]), reverse=False))
        elif name == "TEST_ST_SRC21_BYPASS":
            body.extend(_store_bypass_case(ctx, int(args[0]), args[3], args[4], _mask_expr(args[5]), int(args[6], 0), args[7], int(args[1]), int(args[2]), reverse=True))
        elif name == "TEST_CASE":
            body.extend([*_test_case_prelude(int(args[0])), *_translate_code_block(ctx, int(args[0]), args[3]), *_test_case_epilogue(args[1], _mask_expr(args[2]))])
    body.extend(trailer)
    return _render_program(stem, body)


def _render_lui_file() -> str:
    body = [
        *_test_case_prelude(2),
        "slui x1, 0x00000",
        *_test_case_epilogue("x1", _mask_expr("0x00000000")),
        *_test_case_prelude(3),
        "slui x1, 0xfffff",
        "ssrai x1, x1, 1",
        *_test_case_epilogue("x1", _mask_expr("0xfffff800")),
        *_test_case_prelude(4),
        "slui x1, 0x7ffff",
        "ssrai x1, x1, 20",
        *_test_case_epilogue("x1", _mask_expr("0x000007ff")),
        *_test_case_prelude(5),
        "slui x1, 0x80000",
        "ssrai x1, x1, 20",
        *_test_case_epilogue("x1", _mask_expr("0xfffff800")),
        *_test_case_prelude(6),
        "slui x0, 0x80000",
        *_test_case_epilogue("x0", 0),
    ]
    return _render_program("lui", body)


def _render_auipc_file() -> str:
    body = [
        "test_2:",
        "    saddi x3, x0, 2",
        "    li x10, label_2_target + 9992",
        "    sjal x11, label_2_target",
        "    nop",
        "    nop",
        "label_2_target:",
        "    ssub x10, x10, x11",
        # Penguin uses a word-indexed pc and jal writes pc + 1, so the two
        # required delay slots shift the upstream byte-PC expectation by 6.
        *_indent(_test_case_epilogue("x10", _mask_expr("9994"))),
        "test_3:",
        "    saddi x3, x0, 3",
        "    li x10, label_3_target - 10008",
        "    sjal x11, label_3_target",
        "    nop",
        "    nop",
        "label_3_target:",
        "    ssub x10, x10, x11",
        *_indent(_test_case_epilogue("x10", _mask_expr("-10006"))),
    ]
    return _render_program("auipc", body, already_indented=True)


def _render_simple_file() -> str:
    return _render_program(
        "simple",
        [
            "test_2:",
            "    saddi x3, x0, 2",
            "    sjal x0, test_pass",
            "    nop",
            "    nop",
        ],
        already_indented=True,
    )


def _render_jal_file() -> str:
    body = [
        "# Penguin note: upstream jal.S contains a no-delay-slot case that does not",
        "# apply to Penguin. Dedicated delay-slot tests cover that behavior separately.",
        "",
        "test_2:",
        "    saddi x3, x0, 2",
        "    saddi x1, x0, 0",
        "    sjal x4, target_2",
        "linkaddr_2:",
        "    nop",
        "    nop",
        "    sjal x0, fail",
        "    nop",
        "    nop",
        "target_2:",
        "    li x2, linkaddr_2",
        "    sbne x2, x4, fail",
        "    nop",
        "    nop",
    ]
    return _render_program("jal", body, already_indented=True)


def _render_jalr_file() -> str:
    body = [
        "# Penguin note: upstream jalr.S contains a no-delay-slot case that does not",
        "# apply to Penguin. Dedicated delay-slot tests cover that behavior separately.",
        "",
        "test_2:",
        "    saddi x3, x0, 2",
        "    saddi x5, x0, 0",
        "    li x6, target_2",
        "    sjalr x5, x6, 0",
        "linkaddr_2:",
        "    nop",
        "    nop",
        "    sjal x0, fail",
        "    nop",
        "    nop",
        "target_2:",
        "    li x6, linkaddr_2",
        "    sbne x5, x6, fail",
        "    nop",
        "    nop",
        "test_3:",
        "    saddi x3, x0, 3",
        "    li x5, target_3",
        "    sjalr x5, x5, 0",
        "linkaddr_3:",
        "    nop",
        "    nop",
        "    sjal x0, fail",
        "    nop",
        "    nop",
        "target_3:",
        "    li x6, linkaddr_3",
        "    sbne x5, x6, fail",
        "    nop",
        "    nop",
        "test_4:",
        "    saddi x3, x0, 4",
        "    saddi x4, x0, 0",
        "jalr_bypass_4_loop:",
        "    li x6, jalr_bypass_4_target",
        "    sjalr x13, x6, 0",
        "jalr_bypass_4_return:",
        "    nop",
        "    nop",
        "    sjal x0, fail",
        "    nop",
        "    nop",
        "jalr_bypass_4_target:",
        "    saddi x4, x4, 1",
        "    saddi x5, x0, 2",
        "    sbne x4, x5, jalr_bypass_4_loop",
        "    nop",
        "    nop",
        "test_5:",
        "    saddi x3, x0, 5",
        "    saddi x4, x0, 0",
        "jalr_bypass_5_loop:",
        "    li x6, jalr_bypass_5_target",
        "    nop",
        "    sjalr x13, x6, 0",
        "jalr_bypass_5_return:",
        "    nop",
        "    nop",
        "    sjal x0, fail",
        "    nop",
        "    nop",
        "jalr_bypass_5_target:",
        "    saddi x4, x4, 1",
        "    saddi x5, x0, 2",
        "    sbne x4, x5, jalr_bypass_5_loop",
        "    nop",
        "    nop",
        "test_6:",
        "    saddi x3, x0, 6",
        "    saddi x4, x0, 0",
        "jalr_bypass_6_loop:",
        "    li x6, jalr_bypass_6_target",
        "    nop",
        "    nop",
        "    sjalr x13, x6, 0",
        "jalr_bypass_6_return:",
        "    nop",
        "    nop",
        "    sjal x0, fail",
        "    nop",
        "    nop",
        "jalr_bypass_6_target:",
        "    saddi x4, x4, 1",
        "    saddi x5, x0, 2",
        "    sbne x4, x5, jalr_bypass_6_loop",
        "    nop",
        "    nop",
    ]
    return _render_program("jalr", body, already_indented=True)


def _render_ld_st_file(stem: str) -> str:
    body: list[str] = []
    trailer: list[str] = []
    for line in _logical_lines(stem):
        match = MACRO_PATTERN.match(line)
        if match is None:
            if line.startswith("li ") or line.startswith("la ") or line.startswith("sb ") or line.startswith("lb "):
                trailer.extend(_translate_code_block(TestContext(stem), 999, line))
            continue
        name = match.group("name")
        args = _split_args(match.group("args"))
        if name == "TEST_LD_ST_BYPASS":
            body.extend(_ld_st_bypass_case(int(args[0]), args[1], args[2], _mask_expr(args[3]), int(args[4], 0), args[5]))
        elif name == "TEST_ST_LD_BYPASS":
            body.extend(_st_ld_bypass_case(int(args[0]), args[1], args[2], _mask_expr(args[3]), int(args[4], 0), args[5]))
    body.extend(trailer)
    return _render_program(stem, body)


def _render_program(stem: str, body: list[str], *, already_indented: bool = False) -> str:
    lines = [
        f"# Flattened from upstream riscv-tests rv32ui/{stem}.S",
        "#",
        "# Penguin test-result convention for imported riscv-tests programs:",
        "#   x10 = 0 on pass, 1 on fail",
        "#   x3  = failing upstream test number on fail, 0 on pass",
        "#   secall terminates the test and returns control to the host harness",
        "",
    ]
    if already_indented:
        lines.extend(body)
    else:
        lines.extend(_indent(body))
    if lines[-1] != "":
        lines.append("")
    lines.extend(
        [
            "test_pass:",
            "    saddi x3, x0, 0",
            "    saddi x10, x0, 0",
            "    secall",
            "",
            "fail:",
            "    saddi x10, x0, 1",
            "    secall",
            "",
        ]
    )
    return "\n".join(lines)


def render_program(stem: str) -> str:
    if stem in RR_STEMS:
        return _render_rr_file(stem)
    if stem in IMM_STEMS:
        return _render_imm_file(stem)
    if stem in BRANCH_STEMS:
        return _render_branch_file(stem)
    if stem in LOAD_STEMS:
        return _render_load_file(stem)
    if stem in STORE_STEMS:
        return _render_store_file(stem)
    if stem == "lui":
        return _render_lui_file()
    if stem == "auipc":
        return _render_auipc_file()
    if stem == "simple":
        return _render_simple_file()
    if stem == "jal":
        return _render_jal_file()
    if stem == "jalr":
        return _render_jalr_file()
    if stem in {"ld_st", "st_ld"}:
        return _render_ld_st_file(stem)
    raise ValueError(f"Unsupported stem: {stem}")


def main() -> None:
    missing = [
        stem
        for stem in SUPPORTED_STEMS
        if not (UPSTREAM_ROOT / "rv32ui" / f"{stem}.S").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing upstream rv32ui sources: {missing}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for stem in SUPPORTED_STEMS:
        (OUTPUT_ROOT / f"{stem}.S").write_text(render_program(stem))
    print(f"Generated {len(SUPPORTED_STEMS)} flattened rv32ui programs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
