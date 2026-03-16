"""Specification-driven MXU tests for the Penguin functional model."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from typing import Any

import pytest
import torch

from penguin_model import (
    ALL_INSTRUCTION_SPECS,
    INSTRUCTION_LATENCY,
    MATMUL_LATENCY_CYCLES,
    MXU_PUSH_LATENCY_CYCLES,
    TENSOR_INSTRUCTION_SPECS,
    VLOAD_LATENCY_CYCLES,
    VSTORE_LATENCY_CYCLES,
    ArchState,
    Instruction,
    PenguinCore,
    StopReason,
    TensorMemType,
)
from penguin_model.testbench import (
    DRAM_BASE,
    TEST_CORE_CONFIG,
    TEST_DRAM_SIZE,
    TEST_IMEM_SIZE,
    TEST_VMEM_SIZE,
    VMEM_BASE,
)

MREG_ROWS = 64
MREG_ROW_BYTES = 32
MREG_FP8_COLS = 32
MREG_BF16_COLS = 16
MREG_BYTES = 2048
WEIGHT_TILE_ROWS = 32
WEIGHT_TILE_COLS = 16
WEIGHT_SLOT_BYTES = 512

REQUIRED_MXU_MNEMONICS = {
    "mxu.push.mxu0",
    "mxu.push.mxu1",
    "matmul.mxu0",
    "matmul.mxu1",
    "matmul.add.mxu0",
    "matmul.add.mxu1",
}


def _fresh_state() -> ArchState:
    return ArchState.from_config(TEST_CORE_CONFIG)


def _fresh_core() -> PenguinCore:
    return PenguinCore(config=TEST_CORE_CONFIG)


def _require_mxu_support() -> None:
    missing = sorted(REQUIRED_MXU_MNEMONICS - set(TENSOR_INSTRUCTION_SPECS))
    if missing:
        pytest.xfail(f"MXU mnemonics are not registered yet: {', '.join(missing)}")


def _require_fp8_dtype() -> torch.dtype:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.xfail("PyTorch float8_e4m3fn support is required for MXU reference vectors")
    return dtype


def _field_type_name(field: Any) -> str:
    field_type = field.type
    return field_type if isinstance(field_type, str) else getattr(field_type, "__name__", "")


def _register_operand(field: Any, register_name: str) -> Any:
    type_name = _field_type_name(field)
    if type_name == "str" or isinstance(field.default, str):
        return register_name
    return int(register_name[1:])


def _weight_slot_operand(field: Any, slot_name: str) -> Any:
    type_name = _field_type_name(field)
    if type_name == "str" or isinstance(field.default, str):
        return slot_name
    return int(slot_name[1:])


def _build_mxu_params(mnemonic: str, **operands: Any) -> Any:
    params_type = ALL_INSTRUCTION_SPECS[mnemonic].params_type
    if not is_dataclass(params_type):
        pytest.xfail(f"{mnemonic} params type is not a dataclass: {params_type!r}")

    values: dict[str, Any] = {}
    for field in fields(params_type):
        name = field.name
        if name in {"rd", "md", "mdst", "dest", "dest_mreg", "m_dest", "mrd"}:
            values[name] = _register_operand(field, operands["dest"])
        elif name in {"rs", "ms", "msrc", "src", "src_mreg", "m_src", "mra", "lhs"}:
            values[name] = _register_operand(field, operands["src"])
        elif name in {"mpartial", "partial", "partial_mreg", "m_partial", "acc", "accum", "mp"}:
            values[name] = _register_operand(field, operands["partial"])
        elif name in {"wsel", "wsrc", "slot", "weight_slot", "wslot", "ws"}:
            values[name] = _weight_slot_operand(field, operands["slot"])
        elif name in {"rs1", "base", "addr_rs", "vmem_rs", "base_rs", "address_rs"}:
            values[name] = operands["rs1"]
        elif name in {"imm", "offset"}:
            values[name] = operands.get("imm", 0)
        elif field.default is not MISSING:
            continue
        else:
            pytest.xfail(f"Unrecognized {mnemonic} operand field {name!r}")

    return params_type(**values)


def _make_instruction(mnemonic: str, **operands: Any) -> Instruction:
    _require_mxu_support()
    return Instruction(mnemonic, _build_mxu_params(mnemonic, **operands))


def _as_byte_tensor(data: bytes) -> torch.Tensor:
    return torch.tensor(list(data), dtype=torch.uint8)


def _value_to_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, torch.Tensor):
        return bytes(int(item) for item in value.flatten().tolist())
    if isinstance(value, list):
        return bytes(int(item) & 0xFF for item in value)
    if isinstance(value, tuple):
        return bytes(int(item) & 0xFF for item in value)
    pytest.xfail(f"Unsupported tensor-register storage value type: {type(value)!r}")


def _tensor_register_container(state: ArchState) -> Any:
    for name in ("tensor_regs", "tensor_registers", "mregs", "mreg"):
        container = getattr(state, name, None)
        if container is not None:
            return container
    pytest.xfail("Tensor register file is not exposed on the architectural state yet")


def _write_mreg_bytes(state: ArchState, index: int, data: bytes) -> None:
    if len(data) != MREG_BYTES:
        raise ValueError(f"Tensor register image must be {MREG_BYTES} bytes")

    for name in ("write_mreg_bytes", "set_mreg_bytes", "store_mreg_bytes"):
        method = getattr(state, name, None)
        if method is not None:
            method(index, data)
            return

    container = _tensor_register_container(state)
    key_options = (index, f"m{index}")
    for key in key_options:
        try:
            current = container[key]
        except Exception:
            continue

        if isinstance(current, torch.Tensor):
            byte_tensor = _as_byte_tensor(data)
            if current.shape == byte_tensor.shape:
                current.copy_(byte_tensor)
            else:
                container[key] = byte_tensor
            return
        if isinstance(current, (bytes, bytearray, list, tuple)):
            container[key] = type(current)(data)
            return
        container[key] = data
        return

    pytest.xfail("Tensor register file exists but tests cannot write tensor-register bytes yet")


def _read_mreg_bytes(state: ArchState, index: int) -> bytes:
    for name in ("read_mreg_bytes", "get_mreg_bytes", "load_mreg_bytes"):
        method = getattr(state, name, None)
        if method is not None:
            return _value_to_bytes(method(index))

    container = _tensor_register_container(state)
    for key in (index, f"m{index}"):
        try:
            return _value_to_bytes(container[key])
        except Exception:
            continue

    pytest.xfail("Tensor register file exists but tests cannot read tensor-register bytes yet")


def _pack_fp8_tile(values: torch.Tensor) -> bytes:
    fp8_dtype = _require_fp8_dtype()
    if tuple(values.shape) != (MREG_ROWS, MREG_FP8_COLS):
        raise ValueError("Activation tile must be 64x32 FP8 elements")
    packed = values.to(dtype=fp8_dtype).contiguous().view(torch.uint8)
    return bytes(int(item) for item in packed.flatten().tolist())


def _pack_weight_tile(values: torch.Tensor) -> bytes:
    fp8_dtype = _require_fp8_dtype()
    if tuple(values.shape) != (WEIGHT_TILE_ROWS, WEIGHT_TILE_COLS):
        raise ValueError("Weight tile must be 32x16 FP8 elements")
    packed = values.to(dtype=fp8_dtype).contiguous().view(torch.uint8)
    return bytes(int(item) for item in packed.flatten().tolist())


def _pack_bf16_tile(values: torch.Tensor) -> bytes:
    if tuple(values.shape) != (MREG_ROWS, MREG_BF16_COLS):
        raise ValueError("Result tile must be 64x16 BF16 elements")
    packed = values.to(dtype=torch.bfloat16).contiguous().view(torch.uint8)
    return bytes(int(item) for item in packed.flatten().tolist())


def _reference_matmul(activation_tile: torch.Tensor, weight_tile: torch.Tensor) -> torch.Tensor:
    fp8_dtype = _require_fp8_dtype()
    activation_fp8 = activation_tile.to(dtype=fp8_dtype).to(dtype=torch.float32)
    weight_fp8 = weight_tile.to(dtype=fp8_dtype).to(dtype=torch.float32)
    return (activation_fp8 @ weight_fp8).to(dtype=torch.bfloat16)


def _small_activation_tile() -> torch.Tensor:
    tile = torch.zeros((MREG_ROWS, MREG_FP8_COLS), dtype=torch.float32)
    tile[:2, :3] = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, -2.0, 3.0],
        ],
        dtype=torch.float32,
    )
    return tile


def _weight_tile_a() -> torch.Tensor:
    tile = torch.zeros((WEIGHT_TILE_ROWS, WEIGHT_TILE_COLS), dtype=torch.float32)
    tile[:3, :2] = torch.tensor(
        [
            [1.0, -1.0],
            [0.5, 2.0],
            [-3.0, 0.25],
        ],
        dtype=torch.float32,
    )
    return tile


def _weight_tile_b() -> torch.Tensor:
    tile = torch.zeros((WEIGHT_TILE_ROWS, WEIGHT_TILE_COLS), dtype=torch.float32)
    tile[:3, :2] = torch.tensor(
        [
            [2.0, 1.0],
            [-1.5, 0.5],
            [0.25, -2.0],
        ],
        dtype=torch.float32,
    )
    return tile


def _partial_tile() -> torch.Tensor:
    tile = torch.zeros((MREG_ROWS, MREG_BF16_COLS), dtype=torch.float32)
    tile[:2, :2] = torch.tensor(
        [
            [1.0, -2.0],
            [0.5, 3.0],
        ],
        dtype=torch.float32,
    )
    return tile.to(dtype=torch.bfloat16)


def _assert_program_completed(core: PenguinCore, perf: Any, *, instructions: int) -> None:
    assert core.state.stop_reason == StopReason.PROGRAM_END
    assert perf.instructions == instructions


def test_mxu_instruction_family_registers_once_support_lands() -> None:
    _require_mxu_support()
    assert REQUIRED_MXU_MNEMONICS <= set(TENSOR_INSTRUCTION_SPECS)


def test_tensor_instruction_latency_view_exposes_tensor_ops() -> None:
    _require_mxu_support()
    assert INSTRUCTION_LATENCY["vload"] == VLOAD_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["vstore"] == VSTORE_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["mxu.push.mxu0"] == MXU_PUSH_LATENCY_CYCLES
    assert INSTRUCTION_LATENCY["matmul.mxu1"] == MATMUL_LATENCY_CYCLES


def test_vload_and_vstore_round_trip_tensor_image_and_perf_counters() -> None:
    _require_mxu_support()
    core = _fresh_core()
    image = bytes(((index * 17) + 3) & 0xFF for index in range(MREG_BYTES))

    core.state.vmem.write(VMEM_BASE + 0x80, _as_byte_tensor(image))
    core.state.write_xreg(1, VMEM_BASE + 0x80)
    core.state.write_xreg(2, VMEM_BASE + 0x800)

    perf = core.execute(
        [
            Instruction("vload", TensorMemType(mreg=5, rs1=1, imm=0)),
            Instruction("vstore", TensorMemType(mreg=5, rs1=2, imm=0)),
        ]
    )

    _assert_program_completed(core, perf, instructions=2)
    assert _read_mreg_bytes(core.state, 5) == image
    assert bytes(core.state.vmem.read(VMEM_BASE + 0x800, MREG_BYTES).tolist()) == image
    assert perf.cycles == VLOAD_LATENCY_CYCLES + VSTORE_LATENCY_CYCLES
    assert perf.bytes_read == MREG_BYTES
    assert perf.bytes_written == MREG_BYTES
    assert perf.instructions_by_opcode == {"vload": 1, "vstore": 1}


def test_mxu0_push_and_matmul_produce_expected_bf16_tile() -> None:
    _require_mxu_support()
    core = _fresh_core()
    activation = _small_activation_tile()
    weights = _weight_tile_a()
    expected = _pack_bf16_tile(_reference_matmul(activation, weights))

    _write_mreg_bytes(core.state, 1, _pack_fp8_tile(activation))
    core.state.vmem.write(VMEM_BASE + 0x100, _as_byte_tensor(_pack_weight_tile(weights)))
    core.state.write_xreg(1, VMEM_BASE + 0x100)

    perf = core.execute(
        [
            _make_instruction("mxu.push.mxu0", slot="w0", rs1=1, imm=0),
            _make_instruction("matmul.mxu0", dest="m2", src="m1", slot="w0"),
        ]
    )

    _assert_program_completed(core, perf, instructions=2)
    assert _read_mreg_bytes(core.state, 2) == expected


def test_matmul_add_uses_explicit_partial_tensor_operand() -> None:
    _require_mxu_support()
    core = _fresh_core()
    activation = _small_activation_tile()
    weights = _weight_tile_a()
    partial = _partial_tile()
    expected = _pack_bf16_tile(_reference_matmul(activation, weights) + partial)

    _write_mreg_bytes(core.state, 1, _pack_fp8_tile(activation))
    _write_mreg_bytes(core.state, 3, _pack_bf16_tile(partial))
    _write_mreg_bytes(core.state, 4, _pack_bf16_tile(torch.full((MREG_ROWS, MREG_BF16_COLS), 7.0)))
    core.state.vmem.write(VMEM_BASE + 0x140, _as_byte_tensor(_pack_weight_tile(weights)))
    core.state.write_xreg(1, VMEM_BASE + 0x140)

    perf = core.execute(
        [
            _make_instruction("mxu.push.mxu0", slot="w0", rs1=1, imm=0),
            _make_instruction("matmul.add.mxu0", dest="m4", src="m1", slot="w0", partial="m3"),
        ]
    )

    _assert_program_completed(core, perf, instructions=2)
    assert _read_mreg_bytes(core.state, 4) == expected


def test_fresh_matmul_overwrites_destination_tile() -> None:
    _require_mxu_support()
    core = _fresh_core()
    activation = _small_activation_tile()
    weights = _weight_tile_a()
    expected = _pack_bf16_tile(_reference_matmul(activation, weights))

    _write_mreg_bytes(core.state, 1, _pack_fp8_tile(activation))
    _write_mreg_bytes(core.state, 2, _pack_bf16_tile(torch.full((MREG_ROWS, MREG_BF16_COLS), 5.0)))
    core.state.vmem.write(VMEM_BASE + 0x180, _as_byte_tensor(_pack_weight_tile(weights)))
    core.state.write_xreg(1, VMEM_BASE + 0x180)

    perf = core.execute(
        [
            _make_instruction("mxu.push.mxu0", slot="w0", rs1=1, imm=0),
            _make_instruction("matmul.mxu0", dest="m2", src="m1", slot="w0"),
        ]
    )

    _assert_program_completed(core, perf, instructions=2)
    assert _read_mreg_bytes(core.state, 2) == expected


def test_mxu_weight_slots_w0_and_w1_are_independent() -> None:
    _require_mxu_support()
    core = _fresh_core()
    activation = _small_activation_tile()
    weights_w0 = _weight_tile_a()
    weights_w1 = _weight_tile_b()
    expected_w0 = _pack_bf16_tile(_reference_matmul(activation, weights_w0))
    expected_w1 = _pack_bf16_tile(_reference_matmul(activation, weights_w1))

    _write_mreg_bytes(core.state, 1, _pack_fp8_tile(activation))
    core.state.vmem.write(VMEM_BASE + 0x200, _as_byte_tensor(_pack_weight_tile(weights_w0)))
    core.state.vmem.write(VMEM_BASE + 0x240, _as_byte_tensor(_pack_weight_tile(weights_w1)))
    core.state.write_xreg(1, VMEM_BASE + 0x200)
    core.state.write_xreg(2, VMEM_BASE + 0x240)

    perf = core.execute(
        [
            _make_instruction("mxu.push.mxu0", slot="w0", rs1=1, imm=0),
            _make_instruction("mxu.push.mxu0", slot="w1", rs1=2, imm=0),
            _make_instruction("matmul.mxu0", dest="m2", src="m1", slot="w0"),
            _make_instruction("matmul.mxu0", dest="m3", src="m1", slot="w1"),
        ]
    )

    _assert_program_completed(core, perf, instructions=4)
    assert _read_mreg_bytes(core.state, 2) == expected_w0
    assert _read_mreg_bytes(core.state, 3) == expected_w1
    assert expected_w0 != expected_w1


def test_mxu0_and_mxu1_keep_distinct_weight_state() -> None:
    _require_mxu_support()
    core = _fresh_core()
    activation = _small_activation_tile()
    weights_mxu0 = _weight_tile_a()
    weights_mxu1 = _weight_tile_b()
    expected_mxu0 = _pack_bf16_tile(_reference_matmul(activation, weights_mxu0))
    expected_mxu1 = _pack_bf16_tile(_reference_matmul(activation, weights_mxu1))

    _write_mreg_bytes(core.state, 1, _pack_fp8_tile(activation))
    core.state.vmem.write(VMEM_BASE + 0x280, _as_byte_tensor(_pack_weight_tile(weights_mxu0)))
    core.state.vmem.write(VMEM_BASE + 0x2C0, _as_byte_tensor(_pack_weight_tile(weights_mxu1)))
    core.state.write_xreg(1, VMEM_BASE + 0x280)
    core.state.write_xreg(2, VMEM_BASE + 0x2C0)

    perf = core.execute(
        [
            _make_instruction("mxu.push.mxu0", slot="w0", rs1=1, imm=0),
            _make_instruction("mxu.push.mxu1", slot="w0", rs1=2, imm=0),
            _make_instruction("matmul.mxu0", dest="m2", src="m1", slot="w0"),
            _make_instruction("matmul.mxu1", dest="m3", src="m1", slot="w0"),
        ]
    )

    _assert_program_completed(core, perf, instructions=4)
    assert _read_mreg_bytes(core.state, 2) == expected_mxu0
    assert _read_mreg_bytes(core.state, 3) == expected_mxu1
    assert expected_mxu0 != expected_mxu1


def test_mxu_push_uses_vmem_not_dram_as_weight_source() -> None:
    _require_mxu_support()
    core = _fresh_core()
    activation = _small_activation_tile()
    weights_vmem = _weight_tile_a()
    weights_dram = _weight_tile_b()
    expected_vmem = _pack_bf16_tile(_reference_matmul(activation, weights_vmem))

    _write_mreg_bytes(core.state, 1, _pack_fp8_tile(activation))
    core.state.vmem.write(VMEM_BASE + 0x300, _as_byte_tensor(_pack_weight_tile(weights_vmem)))
    core.state.dram.write(DRAM_BASE + 0x300, _as_byte_tensor(_pack_weight_tile(weights_dram)))
    core.state.write_xreg(1, VMEM_BASE + 0x300)

    perf = core.execute(
        [
            _make_instruction("mxu.push.mxu0", slot="w0", rs1=1, imm=0),
            _make_instruction("matmul.mxu0", dest="m2", src="m1", slot="w0"),
        ]
    )

    _assert_program_completed(core, perf, instructions=2)
    assert _read_mreg_bytes(core.state, 2) == expected_vmem


def test_mxu_push_rejects_misaligned_vmem_address() -> None:
    _require_mxu_support()
    core = _fresh_core()
    activation = _small_activation_tile()

    _write_mreg_bytes(core.state, 1, _pack_fp8_tile(activation))
    _write_mreg_bytes(core.state, 2, _pack_bf16_tile(torch.zeros((MREG_ROWS, MREG_BF16_COLS))))
    core.state.write_xreg(1, VMEM_BASE + 0x101)

    perf = core.execute(
        [
            _make_instruction("mxu.push.mxu0", slot="w0", rs1=1, imm=0),
            _make_instruction("matmul.mxu0", dest="m2", src="m1", slot="w0"),
        ]
    )

    assert perf.instructions >= 1
    assert core.state.stop_reason is not None
    assert _read_mreg_bytes(core.state, 2) == _pack_bf16_tile(
        torch.zeros((MREG_ROWS, MREG_BF16_COLS))
    )


def test_vload_rejects_misaligned_vmem_address_without_mutating_destination() -> None:
    _require_mxu_support()
    core = _fresh_core()
    original = bytes(((index * 5) + 1) & 0xFF for index in range(MREG_BYTES))

    _write_mreg_bytes(core.state, 7, original)
    core.state.write_xreg(1, VMEM_BASE + 0x81)

    perf = core.execute([Instruction("vload", TensorMemType(mreg=7, rs1=1, imm=0))])

    assert perf.instructions == 1
    assert core.state.stop_reason == StopReason.TENSOR_MEMORY_MISALIGNED
    assert _read_mreg_bytes(core.state, 7) == original


def test_mxu_perf_counters_track_tensor_latencies_and_opcode_histogram() -> None:
    _require_mxu_support()
    core = _fresh_core()
    activation = _small_activation_tile()
    weights = _weight_tile_a()

    _write_mreg_bytes(core.state, 1, _pack_fp8_tile(activation))
    core.state.vmem.write(VMEM_BASE + 0x340, _as_byte_tensor(_pack_weight_tile(weights)))
    core.state.write_xreg(1, VMEM_BASE + 0x340)

    perf = core.execute(
        [
            _make_instruction("mxu.push.mxu1", slot="w1", rs1=1, imm=0),
            _make_instruction("matmul.mxu1", dest="m2", src="m1", slot="w1"),
        ]
    )

    _assert_program_completed(core, perf, instructions=2)
    assert perf.cycles == MXU_PUSH_LATENCY_CYCLES + MATMUL_LATENCY_CYCLES
    assert perf.bytes_read == WEIGHT_SLOT_BYTES
    assert perf.bytes_written == 0
    assert perf.instructions_by_opcode == {"mxu.push.mxu1": 1, "matmul.mxu1": 1}
