"""RV32I-compatible binary encoding helpers for Penguin scalar instructions."""

from __future__ import annotations

from .instructions import BType, EmptyType, IType, Instruction, JType, RType, SType, UType, VPUBinaryType

LOAD_FUNCT3 = {
    "slb": 0b000,
    "slh": 0b001,
    "slw": 0b010,
    "slbu": 0b100,
    "slhu": 0b101,
    "lb": 0b000,
    "lh": 0b001,
    "lw": 0b010,
    "lbu": 0b100,
    "lhu": 0b101,
    "sld": 0b010,
}
STORE_FUNCT3 = {
    "ssb": 0b000,
    "ssh": 0b001,
    "ssw": 0b010,
    "sb": 0b000,
    "sh": 0b001,
    "sw": 0b010,
    "sst": 0b010,
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

    mnemonic = instruction.mnemonic
    params = instruction.params

    if mnemonic == "slui":
        if not isinstance(params, UType):
            raise TypeError("slui expects UType operands")
        return _mask_u32(_encode_u_type(OPCODE_LUI, params))

    if mnemonic == "sauipc":
        if not isinstance(params, UType):
            raise TypeError("sauipc expects UType operands")
        return _mask_u32(_encode_u_type(OPCODE_AUIPC, params))

    if mnemonic == "sjal":
        if not isinstance(params, JType):
            raise TypeError("sjal expects JType operands")
        _check_range("imm", params.imm, 21)
        _check_alignment("imm", params.imm, 2)
        return _mask_u32(_encode_j_type(OPCODE_JAL, params))

    if mnemonic == "sjalr":
        if not isinstance(params, IType):
            raise TypeError("sjalr expects IType operands")
        _check_range("imm", params.imm, 12)
        return _mask_u32(_encode_i_type(OPCODE_JALR, 0b000, params.imm, params.rs1, params.rd))

    if mnemonic in {"sbeq", "sbne", "sblt", "sbge", "sbltu", "sbgeu"}:
        if not isinstance(params, BType):
            raise TypeError(f"{mnemonic} expects BType operands")
        _check_range("imm", params.imm, 13)
        _check_alignment("imm", params.imm, 2)
        funct3 = {
            "sbeq": 0b000,
            "sbne": 0b001,
            "sblt": 0b100,
            "sbge": 0b101,
            "sbltu": 0b110,
            "sbgeu": 0b111,
        }[mnemonic]
        return _mask_u32(_encode_b_type(OPCODE_BRANCH, funct3, params))

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

    if mnemonic in {"saddi", "sslti", "ssltiu", "sxori", "sori", "sandi"}:
        if not isinstance(params, IType):
            raise TypeError(f"{mnemonic} expects IType operands")
        _check_range("imm", params.imm, 12)
        funct3 = {
            "saddi": 0b000,
            "sslti": 0b010,
            "ssltiu": 0b011,
            "sxori": 0b100,
            "sori": 0b110,
            "sandi": 0b111,
        }[mnemonic]
        return _mask_u32(_encode_i_type(OPCODE_OP_IMM, funct3, params.imm, params.rs1, params.rd))

    if mnemonic in {"sslli", "ssrli", "ssrai"}:
        if not isinstance(params, IType):
            raise TypeError(f"{mnemonic} expects IType operands")
        shamt = params.imm & 0x1F
        funct7 = 0b0100000 if mnemonic == "ssrai" else 0b0000000
        funct3 = 0b001 if mnemonic == "sslli" else 0b101
        imm12 = (funct7 << 5) | shamt
        return _mask_u32(_encode_i_type(OPCODE_OP_IMM, funct3, imm12, params.rs1, params.rd))

    if mnemonic in {"sadd", "ssub", "ssll", "sslt", "ssltu", "sxor", "ssrl", "ssra", "sor", "sand"}:
        if not isinstance(params, RType):
            raise TypeError(f"{mnemonic} expects RType operands")
        funct3 = {
            "sadd": 0b000,
            "ssub": 0b000,
            "ssll": 0b001,
            "sslt": 0b010,
            "ssltu": 0b011,
            "sxor": 0b100,
            "ssrl": 0b101,
            "ssra": 0b101,
            "sor": 0b110,
            "sand": 0b111,
        }[mnemonic]
        funct7 = 0b0100000 if mnemonic in {"ssub", "ssra"} else 0b0000000
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

    if mnemonic == "sfence":
        if not isinstance(params, EmptyType):
            raise TypeError("sfence expects EmptyType operands")
        return _mask_u32(_encode_i_type(OPCODE_MISC_MEM, 0b000, 0, 0, 0))

    if mnemonic == "secall":
        if not isinstance(params, EmptyType):
            raise TypeError("secall expects EmptyType operands")
        return _mask_u32(_encode_i_type(OPCODE_SYSTEM, 0b000, 0x000, 0, 0))

    if mnemonic == "sebreak":
        if not isinstance(params, EmptyType):
            raise TypeError("sebreak expects EmptyType operands")
        return _mask_u32(_encode_i_type(OPCODE_SYSTEM, 0b000, 0x001, 0, 0))

    raise KeyError(f"unsupported scalar encoding mnemonic: {mnemonic}")


__all__ = ["encode_scalar_instruction"]
