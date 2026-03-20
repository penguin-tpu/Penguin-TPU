"""RV32I-compatible binary encoding helpers for Penguin scalar instructions."""

from __future__ import annotations

from .instructions import (
    BType,
    DMAControlType,
    DelayType,
    EmptyType,
    IType,
    Instruction,
    JType,
    MXUAccumulatorType,
    MXUMatmulAccType,
    MXUMatmulType,
    RType,
    ScaleImmType,
    SType,
    UType,
    VectorImmType,
    VPUBinaryType,
    VPUUnaryType,
    WeightTensorType,
    XLUUnaryType,
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
    "vadd": "vadd.bf16",
    "vsub": "vsub.bf16",
    "vmul": "vmul.bf16",
    "vmax": "vmax.bf16",
    "vmin": "vmin.bf16",
    "vrecip": "vrecip.bf16",
    "transpose.xlu": "vtrpose.xlu",
    "reduce.max.xlu": "vreduce.max.xlu",
    "reduce.sum.xlu": "vreduce.sum.xlu",
    "vmatpush.mxu0": "vmatpush.weight.mxu0",
    "vmatpush.mxu1": "vmatpush.weight.mxu1",
    "vmatpush.fp8.weight.mxu0": "vmatpush.weight.mxu0",
    "vmatpush.fp8.weight.mxu1": "vmatpush.weight.mxu1",
    "vmatpush.acc.bf16.mxu0": "vmatpush.bf16.acc.mxu0",
    "vmatpush.acc.bf16.mxu1": "vmatpush.bf16.acc.mxu1",
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
OPCODE_VLS = 0b0000111
OPCODE_VPU = 0b1010111
OPCODE_VI = 0b1011111
OPCODE_XLU = 0b1101011
OPCODE_MXU = 0b1110111
OPCODE_DMA_TRANSFER = 0b1111011
OPCODE_DMA_CONTROL = 0b1111111


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


def _encode_vr_type(opcode: int, funct7: int, vd: int, vs1: int, vs2: int) -> int:
    for name, value in (("vd", vd), ("vs1", vs1), ("vs2", vs2)):
        if value < 0 or value > 0x3F:
            raise ValueError(f"{name}={value} does not fit 6-bit tensor field")
    return (
        ((funct7 & 0x7F) << 25)
        | ((vs2 & 0x3F) << 19)
        | ((vs1 & 0x3F) << 13)
        | ((vd & 0x3F) << 7)
        | (opcode & 0x7F)
    )


def _encode_vi_type(opcode: int, funct3: int, vd: int, imm16: int) -> int:
    if vd < 0 or vd > 0x3F:
        raise ValueError(f"vd={vd} does not fit 6-bit tensor field")
    return (
        ((imm16 & 0xFFFF) << 16)
        | ((funct3 & 0x7) << 13)
        | ((vd & 0x3F) << 7)
        | (opcode & 0x7F)
    )


def _encode_vm_type(opcode: int, funct7: int, vs1: int, wsel: int) -> int:
    if vs1 < 0 or vs1 > 0x3F:
        raise ValueError(f"vs1={vs1} does not fit 6-bit tensor field")
    if wsel < 0 or wsel > 0x1:
        raise ValueError(f"wsel={wsel} does not fit 1-bit weight selector")
    return (
        ((funct7 & 0x7F) << 25)
        | ((vs1 & 0x3F) << 13)
        | ((wsel & 0x1) << 7)
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

    if mnemonic == "seli":
        if not isinstance(params, ScaleImmType):
            raise TypeError("seli expects ScaleImmType operands")
        if params.imm < 0 or params.imm > 0xFF:
            raise ValueError("seli immediate must be in [0, 255]")
        return _mask_u32(_encode_i_type(OPCODE_LOAD, 0b111, params.imm, 0, params.ed))

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

    if mnemonic in {"vadd.bf16", "vsub.bf16", "vmin.bf16", "vmax.bf16", "vmul.bf16"}:
        if not isinstance(params, VPUBinaryType):
            raise TypeError(f"{mnemonic} expects VPUBinaryType operands")
        funct7 = {
            "vadd.bf16": 0b0000000,
            "vsub.bf16": 0b0000010,
            "vmin.bf16": 0b0000100,
            "vmax.bf16": 0b0000110,
            "vmul.bf16": 0b0100100,
        }[mnemonic]
        return _mask_u32(_encode_vr_type(OPCODE_VPU, funct7, params.md, params.ms1, params.ms2))

    if mnemonic in {"vredsum.bf16", "vmov", "vrecip.bf16", "vexp", "vrelu"}:
        if not isinstance(params, VPUUnaryType):
            raise TypeError(f"{mnemonic} expects VPUUnaryType operands")
        funct7 = {
            "vredsum.bf16": 0b0000001,
            "vmov": 0b1000000,
            "vrecip.bf16": 0b1000001,
            "vexp": 0b1000010,
            "vrelu": 0b1000100,
        }[mnemonic]
        return _mask_u32(_encode_vr_type(OPCODE_VPU, funct7, params.md, params.ms, 0))

    if mnemonic in {"vli.all", "vli.row", "vli.col", "vli.one"}:
        if not isinstance(params, VectorImmType):
            raise TypeError(f"{mnemonic} expects VectorImmType operands")
        funct3 = {
            "vli.all": 0b000,
            "vli.row": 0b001,
            "vli.col": 0b010,
            "vli.one": 0b011,
        }[mnemonic]
        return _mask_u32(_encode_vi_type(OPCODE_VI, funct3, params.md, params.imm))

    if mnemonic in {"vtrpose.xlu", "vreduce.max.xlu", "vreduce.sum.xlu"}:
        if not isinstance(params, XLUUnaryType):
            raise TypeError(f"{mnemonic} expects XLUUnaryType operands")
        funct7 = {
            "vtrpose.xlu": 0b0000000,
            "vreduce.max.xlu": 0b0000001,
            "vreduce.sum.xlu": 0b0000010,
        }[mnemonic]
        return _mask_u32(_encode_vr_type(OPCODE_XLU, funct7, params.md, params.ms, 0))

    if mnemonic in {"vmatpush.weight.mxu0", "vmatpush.weight.mxu1"}:
        if not isinstance(params, WeightTensorType):
            raise TypeError(f"{mnemonic} expects WeightTensorType operands")
        funct7 = 0b0000000 if mnemonic.endswith("mxu0") else 0b0000001
        return _mask_u32(_encode_vr_type(OPCODE_MXU, funct7, params.slot, params.ms, 0))

    if mnemonic in {"vmatpush.bf16.acc.mxu0", "vmatpush.bf16.acc.mxu1"}:
        if not isinstance(params, MXUAccumulatorType):
            raise TypeError(f"{mnemonic} expects MXUAccumulatorType operands")
        funct7 = 0b0000100 if mnemonic.endswith("mxu0") else 0b0000101
        return _mask_u32(_encode_vr_type(OPCODE_MXU, funct7, 0, params.mreg, 0))

    if mnemonic in {"vmatpop.fp8.acc.mxu0", "vmatpop.fp8.acc.mxu1"}:
        if not isinstance(params, MXUAccumulatorType):
            raise TypeError(f"{mnemonic} expects MXUAccumulatorType operands")
        funct7 = 0b0000110 if mnemonic.endswith("mxu0") else 0b0000111
        return _mask_u32(_encode_vr_type(OPCODE_MXU, funct7, params.mreg, 0, 0))

    if mnemonic in {"vmatpop.bf16.acc.mxu0", "vmatpop.bf16.acc.mxu1"}:
        if not isinstance(params, MXUAccumulatorType):
            raise TypeError(f"{mnemonic} expects MXUAccumulatorType operands")
        funct7 = 0b0001000 if mnemonic.endswith("mxu0") else 0b0001001
        return _mask_u32(_encode_vr_type(OPCODE_MXU, funct7, params.mreg, 0, 0))

    if mnemonic in {"vmatmul.mxu0", "vmatmul.mxu1"}:
        if not isinstance(params, MXUMatmulType):
            raise TypeError(f"{mnemonic} expects MXUMatmulType operands")
        funct7 = 0b0001010 if mnemonic.endswith("mxu0") else 0b0001011
        return _mask_u32(_encode_vm_type(OPCODE_MXU, funct7, params.ms, params.ws))

    if mnemonic in {"vmatmul.acc.mxu0", "vmatmul.acc.mxu1"}:
        if not isinstance(params, MXUMatmulAccType):
            raise TypeError(f"{mnemonic} expects MXUMatmulAccType operands")
        funct7 = 0b0001100 if mnemonic.endswith("mxu0") else 0b0001101
        return _mask_u32(_encode_vm_type(OPCODE_MXU, funct7, params.ms, params.ws))

    if mnemonic == "fence":
        if not isinstance(params, EmptyType):
            raise TypeError("fence expects EmptyType operands")
        return _mask_u32(_encode_i_type(OPCODE_MISC_MEM, 0b000, 0, 0, 0))

    if mnemonic == "delay":
        if not isinstance(params, DelayType):
            raise TypeError("delay expects DelayType operands")
        if params.cycles < 0 or params.cycles > 0xFFF:
            raise ValueError("delay immediate must be in [0, 4095]")
        return _mask_u32(_encode_i_type(OPCODE_JALR, 0b001, params.cycles, 0, 0))

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
