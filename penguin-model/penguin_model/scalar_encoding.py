"""RV32I-compatible binary encoding helpers for Penguin scalar instructions."""

from __future__ import annotations

from .instructions import (
    BType,
    DelayType,
    EmptyType,
    IType,
    Instruction,
    JType,
    RType,
    SType,
    UType,
    VPUBinaryType,
)

_MNEMONIC_ALIASES = {
    "slui": "lui",
    "sauipc": "auipc",
    "sjal": "jal",
    "sjalr": "jalr",
    "sbeq": "beq",
    "sbne": "bne",
    "sblt": "blt",
    "sbge": "bge",
    "sbltu": "bltu",
    "sbgeu": "bgeu",
    "slb": "lb",
    "slh": "lh",
    "slw": "lw",
    "slbu": "lbu",
    "slhu": "lhu",
    "sld": "lw",
    "ssb": "sb",
    "ssh": "sh",
    "ssw": "sw",
    "sst": "sw",
    "saddi": "addi",
    "sslti": "slti",
    "ssltiu": "sltiu",
    "sxori": "xori",
    "sori": "ori",
    "sandi": "andi",
    "sslli": "slli",
    "ssrli": "srli",
    "ssrai": "srai",
    "sadd": "add",
    "ssub": "sub",
    "ssll": "sll",
    "sslt": "slt",
    "ssltu": "sltu",
    "sxor": "xor",
    "ssrl": "srl",
    "ssra": "sra",
    "sor": "or",
    "sand": "and",
    "sfence": "fence",
    "secall": "ecall",
    "sebreak": "ebreak",
}

LOAD_FUNCT3 = {
    "lb": 0b000,
    "lh": 0b001,
    "lw": 0b010,
    "lbu": 0b100,
    "lhu": 0b101,
}
STORE_FUNCT3 = {
    "sb": 0b000,
    "sh": 0b001,
    "sw": 0b010,
}

OPCODE_LOAD = 0b0000011
OPCODE_MISC_MEM = 0b0001111
OPCODE_OP_IMM = 0b0010011
OPCODE_AUIPC = 0b0010111
OPCODE_STORE = 0b0100011
OPCODE_OP = 0b0110011
OPCODE_LUI = 0b0110111
OPCODE_BRANCH = 0b1100011
OPCODE_JALR = 0b1100111
OPCODE_JAL = 0b1101111
OPCODE_SYSTEM = 0b1110011
OPCODE_CUSTOM_0 = 0b0001011


def _mask_u32(value: int) -> int:
    return value & 0xFFFF_FFFF


def _check_range(name: str, value: int, bits: int) -> None:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if not minimum <= value <= maximum:
        raise ValueError(f"{name}={value} does not fit signed {bits}-bit field")


def _check_alignment(name: str, value: int, align: int) -> None:
    if value % align != 0:
        raise ValueError(f"{name}={value} is not aligned to {align} bytes")


def _encode_r_type(opcode: int, funct3: int, funct7: int, params: RType) -> int:
    return (
        ((funct7 & 0x7F) << 25)
        | ((params.rs2 & 0x1F) << 20)
        | ((params.rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((params.rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def _encode_i_type(opcode: int, funct3: int, imm: int, rs1: int, rd: int) -> int:
    return (
        (((imm & 0xFFF) << 20))
        | ((rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def _encode_s_type(opcode: int, funct3: int, params: SType) -> int:
    imm = params.imm & 0xFFF
    return (
        (((imm >> 5) & 0x7F) << 25)
        | ((params.rs2 & 0x1F) << 20)
        | ((params.rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | ((imm & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def _encode_b_type(opcode: int, funct3: int, params: BType) -> int:
    imm = params.imm & 0x1FFF
    return (
        (((imm >> 12) & 0x1) << 31)
        | (((imm >> 5) & 0x3F) << 25)
        | ((params.rs2 & 0x1F) << 20)
        | ((params.rs1 & 0x1F) << 15)
        | ((funct3 & 0x7) << 12)
        | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 0x1) << 7)
        | (opcode & 0x7F)
    )


def _encode_u_type(opcode: int, params: UType) -> int:
    return (((params.imm & 0xFFFFF) << 12) | ((params.rd & 0x1F) << 7) | (opcode & 0x7F))


def _encode_j_type(opcode: int, params: JType) -> int:
    imm = params.imm & 0x1F_FFFF
    return (
        (((imm >> 20) & 0x1) << 31)
        | (((imm >> 1) & 0x3FF) << 21)
        | (((imm >> 11) & 0x1) << 20)
        | (((imm >> 12) & 0xFF) << 12)
        | ((params.rd & 0x1F) << 7)
        | (opcode & 0x7F)
    )


def encode_scalar_instruction(instruction: Instruction) -> int:
    """Encode one Penguin scalar instruction into a 32-bit machine word."""

    mnemonic = _MNEMONIC_ALIASES.get(instruction.mnemonic, instruction.mnemonic)
    params = instruction.params

    if mnemonic == "lui":
        if not isinstance(params, UType):
            raise TypeError("lui expects UType operands")
        return _mask_u32(_encode_u_type(OPCODE_LUI, params))

    if mnemonic == "auipc":
        if not isinstance(params, UType):
            raise TypeError("auipc expects UType operands")
        return _mask_u32(_encode_u_type(OPCODE_AUIPC, params))

    if mnemonic == "jal":
        if not isinstance(params, JType):
            raise TypeError("jal expects JType operands")
        byte_imm = params.imm * 4
        _check_range("imm", byte_imm, 21)
        _check_alignment("imm", byte_imm, 2)
        return _mask_u32(_encode_j_type(OPCODE_JAL, JType(rd=params.rd, imm=byte_imm)))

    if mnemonic == "jalr":
        if not isinstance(params, IType):
            raise TypeError("jalr expects IType operands")
        byte_imm = params.imm * 4
        _check_range("imm", byte_imm, 12)
        return _mask_u32(_encode_i_type(OPCODE_JALR, 0b000, byte_imm, params.rs1, params.rd))

    if mnemonic in {"beq", "bne", "blt", "bge", "bltu", "bgeu"}:
        if not isinstance(params, BType):
            raise TypeError(f"{mnemonic} expects BType operands")
        byte_imm = params.imm * 4
        _check_range("imm", byte_imm, 13)
        _check_alignment("imm", byte_imm, 2)
        funct3 = {
            "beq": 0b000,
            "bne": 0b001,
            "blt": 0b100,
            "bge": 0b101,
            "bltu": 0b110,
            "bgeu": 0b111,
        }[mnemonic]
        return _mask_u32(
            _encode_b_type(
                OPCODE_BRANCH,
                funct3,
                BType(rs1=params.rs1, rs2=params.rs2, imm=byte_imm),
            )
        )

    if mnemonic in LOAD_FUNCT3:
        if not isinstance(params, IType):
            raise TypeError(f"{mnemonic} expects IType operands")
        _check_range("imm", params.imm, 12)
        return _mask_u32(
            _encode_i_type(OPCODE_LOAD, LOAD_FUNCT3[mnemonic], params.imm, params.rs1, params.rd)
        )

    if mnemonic in STORE_FUNCT3:
        if not isinstance(params, SType):
            raise TypeError(f"{mnemonic} expects SType operands")
        _check_range("imm", params.imm, 12)
        return _mask_u32(_encode_s_type(OPCODE_STORE, STORE_FUNCT3[mnemonic], params))

    if mnemonic in {"addi", "slti", "sltiu", "xori", "ori", "andi"}:
        if not isinstance(params, IType):
            raise TypeError(f"{mnemonic} expects IType operands")
        _check_range("imm", params.imm, 12)
        funct3 = {
            "addi": 0b000,
            "slti": 0b010,
            "sltiu": 0b011,
            "xori": 0b100,
            "ori": 0b110,
            "andi": 0b111,
        }[mnemonic]
        return _mask_u32(_encode_i_type(OPCODE_OP_IMM, funct3, params.imm, params.rs1, params.rd))

    if mnemonic in {"slli", "srli", "srai"}:
        if not isinstance(params, IType):
            raise TypeError(f"{mnemonic} expects IType operands")
        shamt = params.imm & 0x1F
        funct7 = 0b0100000 if mnemonic == "srai" else 0b0000000
        funct3 = 0b001 if mnemonic == "slli" else 0b101
        imm12 = (funct7 << 5) | shamt
        return _mask_u32(_encode_i_type(OPCODE_OP_IMM, funct3, imm12, params.rs1, params.rd))

    if mnemonic in {"add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and"}:
        if not isinstance(params, RType):
            raise TypeError(f"{mnemonic} expects RType operands")
        funct3 = {
            "add": 0b000,
            "sub": 0b000,
            "sll": 0b001,
            "slt": 0b010,
            "sltu": 0b011,
            "xor": 0b100,
            "srl": 0b101,
            "sra": 0b101,
            "or": 0b110,
            "and": 0b111,
        }[mnemonic]
        funct7 = 0b0100000 if mnemonic in {"sub", "sra"} else 0b0000000
        return _mask_u32(_encode_r_type(OPCODE_OP, funct3, funct7, params))

    if mnemonic == "vadd":
        if not isinstance(params, VPUBinaryType):
            raise TypeError("vadd expects VPUBinaryType operands")
        if params.md > 31 or params.ms1 > 31 or params.ms2 > 31:
            raise ValueError("preliminary RTL VPU only supports m0-m31")
        return _mask_u32(
            _encode_r_type(
                OPCODE_CUSTOM_0,
                0b000,
                0b0000000,
                RType(rd=params.md, rs1=params.ms1, rs2=params.ms2),
            )
        )

    if mnemonic == "fence":
        if not isinstance(params, EmptyType):
            raise TypeError("fence expects EmptyType operands")
        return _mask_u32(_encode_i_type(OPCODE_MISC_MEM, 0b000, 0, 0, 0))

    if mnemonic == "delay":
        if not isinstance(params, DelayType):
            raise TypeError("delay expects DelayType operands")
        if params.cycles < 0 or params.cycles > 0xFFF:
            raise ValueError("delay immediate must be in [0, 4095]")
        return _mask_u32(_encode_i_type(OPCODE_SYSTEM, 0b000, params.cycles, 0, 0))

    if mnemonic == "ecall":
        if not isinstance(params, EmptyType):
            raise TypeError("ecall expects EmptyType operands")
        return _mask_u32(_encode_i_type(OPCODE_SYSTEM, 0b000, 0x000, 0, 0))

    if mnemonic == "ebreak":
        if not isinstance(params, EmptyType):
            raise TypeError("ebreak expects EmptyType operands")
        return _mask_u32(_encode_i_type(OPCODE_SYSTEM, 0b000, 0x001, 0, 0))

    raise KeyError(f"unsupported scalar encoding mnemonic: {mnemonic}")


__all__ = ["encode_scalar_instruction"]
