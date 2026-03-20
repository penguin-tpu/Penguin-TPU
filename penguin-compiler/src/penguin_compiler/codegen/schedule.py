"""Static dependency scheduler for fixed Penguin assembly programs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re

from penguin_model import (
    ALL_INSTRUCTION_SPECS,
    BType,
    DEFAULT_PENGUIN_CORE_CONFIG,
    DMAType,
    DelayType,
    EmptyType,
    IType,
    IMEM_BASE,
    Instruction,
    JType,
    MXUAccumulatorType,
    MXUMatmulAccType,
    MXUMatmulType,
    PenguinCoreConfig,
    RType,
    SType,
    ScaleImmType,
    ScaleMemType,
    TensorMemType,
    UType,
    VPUBinaryType,
    VPUUnaryType,
    WeightMemType,
    WeightTensorType,
    XLUTransposeType,
    assemble_text,
)
from penguin_model.isa import SCALAR_LOAD_MNEMONICS

_LABEL_RE = re.compile(r"(?P<label>[A-Za-z_][A-Za-z0-9_]*)\s*:")


@dataclass(frozen=True, slots=True)
class ScheduledSourceInstruction:
    """One parsed instruction plus its attached comments/labels."""

    prologue_lines: tuple[str, ...]
    labels: tuple[str, ...]
    instruction: Instruction
    control_target_label: str | None = None


def schedule_assembly_text(
    source: str,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
    base_address: int = IMEM_BASE,
    source_name: str = "<memory>",
) -> str:
    """Insert `delay N` instructions so software-visible dependencies are explicit."""

    records, trailing_lines = _parse_source_structure(
        source,
        base_address=base_address,
        source_name=source_name,
    )
    ready_cycle: dict[tuple[str, int], int] = defaultdict(int)
    next_issue_cycle = 0
    rendered: list[str] = []

    for record in records:
        rendered.extend(record.prologue_lines)
        rendered.extend(f"{label}:" for label in record.labels)

        instruction = record.instruction
        if isinstance(instruction.params, DelayType):
            continue

        required_issue_cycle = next_issue_cycle
        for resource in _read_resources(instruction):
            required_issue_cycle = max(required_issue_cycle, ready_cycle[resource])

        if required_issue_cycle > next_issue_cycle:
            delay_cycles = required_issue_cycle - next_issue_cycle - 1
            if delay_cycles < 0:
                raise AssertionError("delay scheduling produced a negative cycle count")
            rendered.append(f"    delay {delay_cycles}")
            next_issue_cycle = required_issue_cycle

        rendered.append(
            f"    {_format_instruction(instruction, control_target_label=record.control_target_label)}"
        )

        ready_at = next_issue_cycle + _instruction_latency_cycles(instruction, config) + 1
        for resource in _write_resources(instruction):
            ready_cycle[resource] = max(ready_cycle[resource], ready_at)
        next_issue_cycle += 1

    rendered.extend(trailing_lines)
    return "\n".join(rendered).rstrip() + "\n"


def schedule_assembly_file(
    path: str | Path,
    *,
    config: PenguinCoreConfig = DEFAULT_PENGUIN_CORE_CONFIG,
    base_address: int = IMEM_BASE,
) -> str:
    source_path = Path(path)
    return schedule_assembly_text(
        source_path.read_text(),
        config=config,
        base_address=base_address,
        source_name=str(source_path),
    )


def _parse_source_structure(
    source: str,
    *,
    base_address: int,
    source_name: str,
) -> tuple[list[ScheduledSourceInstruction], tuple[str, ...]]:
    structured_records: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    pending_prologue: list[str] = []
    pending_labels: list[str] = []

    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        code = raw_line.split("#", 1)[0].strip()
        if not code:
            pending_prologue.append(raw_line.rstrip())
            continue

        while True:
            match = _LABEL_RE.match(code)
            if match is None:
                break
            pending_labels.append(match.group("label"))
            code = code[match.end() :].strip()
            if not code:
                break

        if not code:
            continue

        structured_records.append((tuple(pending_prologue), tuple(pending_labels)))
        pending_prologue = []
        pending_labels = []

    program = assemble_text(source, source_name=source_name, base_address=base_address)
    if len(program) != len(structured_records):
        raise ValueError(
            f"{source_name}: parsed {len(structured_records)} source instructions but assembled {len(program)}"
        )

    address_to_label = {
        address: name
        for name, address in sorted(program.labels.items(), key=lambda item: item[1])
    }
    records: list[ScheduledSourceInstruction] = []
    for index, ((prologue_lines, labels), instruction) in enumerate(
        zip(structured_records, program, strict=True)
    ):
        pc = base_address + index
        control_target_label: str | None = None
        if isinstance(instruction.params, BType | JType):
            control_target_label = address_to_label.get(pc + instruction.params.imm)
        records.append(
            ScheduledSourceInstruction(
                prologue_lines=prologue_lines,
                labels=labels,
                instruction=instruction,
                control_target_label=control_target_label,
            )
        )
    return records, tuple(pending_prologue + [f"{label}:" for label in pending_labels])


def _instruction_latency_cycles(
    instruction: Instruction,
    config: PenguinCoreConfig,
) -> int:
    mnemonic = instruction.mnemonic
    params = instruction.params
    if mnemonic == "vload":
        return config.vload_latency_cycles
    if mnemonic == "vstore":
        return config.vstore_latency_cycles
    if mnemonic.startswith("vmatpush.weight."):
        return config.vmatpush_weight_latency_cycles
    if mnemonic.startswith("vload.weight."):
        return config.vload_weight_latency_cycles
    if mnemonic.startswith(("vmatpush.acc.bf16.", "vmatpush.bf16.acc.")):
        return config.vmatpush_acc_latency_cycles
    if mnemonic.startswith("vmatpush.acc.fp8."):
        return config.vmatpop_acc_fp8_latency_cycles
    if mnemonic.startswith("vmatpop.bf16.acc."):
        return config.vmatpop_acc_bf16_latency_cycles
    if mnemonic.startswith("vmatpop.fp8.acc."):
        return config.vmatpop_acc_fp8_latency_cycles
    if mnemonic.startswith("vmatmul"):
        return config.matmul_latency_cycles
    if isinstance(params, VPUBinaryType):
        return config.vpu_simple_op_latency_cycles
    if isinstance(params, VPUUnaryType):
        if mnemonic in {"vexp", "vrecip", "vrecip.bf16"}:
            return config.vpu_non_pipelineable_op_latency_cycles
        return config.vpu_simple_op_latency_cycles
    if isinstance(params, XLUTransposeType):
        return config.xlu_transpose_latency_cycles
    return ALL_INSTRUCTION_SPECS[mnemonic].latency


def _read_resources(instruction: Instruction) -> tuple[tuple[str, int], ...]:
    params = instruction.params
    mnemonic = instruction.mnemonic

    if isinstance(params, RType):
        return _xregs(params.rs1, params.rs2)
    if isinstance(params, IType):
        resources = list(_xregs(params.rs1))
        if mnemonic in SCALAR_LOAD_MNEMONICS:
            resources.append(("vmem", 0))
        return tuple(resources)
    if isinstance(params, SType):
        return _xregs(params.rs1, params.rs2)
    if isinstance(params, BType):
        return _xregs(params.rs1, params.rs2)
    if isinstance(params, UType | JType | EmptyType | DelayType):
        return ()
    if isinstance(params, DMAType):
        resources = list(_xregs(params.rd, params.rs1, params.rs2))
        if mnemonic.startswith("dma.store."):
            resources.append(("vmem", 0))
        return tuple(resources)
    if isinstance(params, ScaleImmType):
        return ()
    if isinstance(params, ScaleMemType):
        return _xregs(params.rs1) + (("vmem", 0),)
    if isinstance(params, TensorMemType):
        resources = list(_xregs(params.rs1))
        if mnemonic == "vload":
            resources.append(("vmem", 0))
        if mnemonic == "vstore":
            resources.append(("mreg", params.mreg))
        return tuple(resources)
    if isinstance(params, WeightMemType):
        return _xregs(params.rs1) + (("vmem", 0),)
    if isinstance(params, WeightTensorType):
        return (("mreg", params.ms),)
    if isinstance(params, MXUAccumulatorType):
        mxu = _mxu_index(mnemonic)
        if mnemonic.startswith(("vmatpush.acc.bf16.", "vmatpush.bf16.acc.")):
            return (("mreg", params.mreg), ("mreg", params.mreg + 1))
        return (("accum", mxu),)
    if isinstance(params, MXUMatmulType):
        return (("mreg", params.ms), ("weight", _weight_resource_index(mnemonic, params.ws)))
    if isinstance(params, MXUMatmulAccType):
        mxu = _mxu_index(mnemonic)
        return (
            ("mreg", params.ms),
            ("weight", _weight_resource_index(mnemonic, params.ws)),
            ("accum", mxu),
        )
    if isinstance(params, VPUBinaryType):
        return (("mreg", params.ms1), ("mreg", params.ms2))
    if isinstance(params, VPUUnaryType):
        return (("mreg", params.ms),)
    if isinstance(params, XLUTransposeType):
        return (("mreg", params.ms),)
    raise TypeError(f"Unsupported instruction params for read scheduling: {type(params).__name__}")


def _write_resources(instruction: Instruction) -> tuple[tuple[str, int], ...]:
    params = instruction.params
    mnemonic = instruction.mnemonic

    if isinstance(params, RType | IType | UType | JType):
        return _xregs(_rd_index(params))
    if isinstance(params, BType | SType | EmptyType | DelayType | DMAType):
        if isinstance(params, SType):
            return (("vmem", 0),)
        return ()
    if isinstance(params, ScaleImmType | ScaleMemType):
        return (("ereg", params.ed),)
    if isinstance(params, TensorMemType):
        if mnemonic == "vload":
            return (("mreg", params.mreg),)
        return (("vmem", 0),)
    if isinstance(params, WeightMemType | WeightTensorType):
        return (("weight", _weight_resource_index(mnemonic, params.slot)),)
    if isinstance(params, MXUAccumulatorType):
        mxu = _mxu_index(mnemonic)
        if mnemonic.startswith(("vmatpush.acc.bf16.", "vmatpush.bf16.acc.")):
            return (("accum", mxu),)
        if mnemonic.startswith("vmatpop.bf16.acc."):
            return (("mreg", params.mreg), ("mreg", params.mreg + 1))
        return (("mreg", params.mreg),)
    if isinstance(params, MXUMatmulType | MXUMatmulAccType):
        return (("accum", _mxu_index(mnemonic)),)
    if isinstance(params, VPUBinaryType | VPUUnaryType | XLUTransposeType):
        return (("mreg", params.md),)
    raise TypeError(f"Unsupported instruction params for write scheduling: {type(params).__name__}")


def _xregs(*indices: int) -> tuple[tuple[str, int], ...]:
    return tuple(("xreg", index) for index in indices if index != 0)


def _rd_index(params: RType | IType | UType | JType) -> int:
    return params.rd


def _mxu_index(mnemonic: str) -> int:
    return 0 if mnemonic.endswith("mxu0") else 1


def _weight_resource_index(mnemonic: str, slot: int) -> int:
    return _mxu_index(mnemonic) * 100 + slot


def _format_instruction(
    instruction: Instruction,
    *,
    control_target_label: str | None = None,
) -> str:
    mnemonic = instruction.mnemonic
    params = instruction.params
    if isinstance(params, RType):
        return f"{mnemonic} x{params.rd}, x{params.rs1}, x{params.rs2}"
    if isinstance(params, IType):
        if mnemonic in SCALAR_LOAD_MNEMONICS:
            return f"{mnemonic} x{params.rd}, {params.imm}(x{params.rs1})"
        if mnemonic == "jalr":
            return f"{mnemonic} x{params.rd}, x{params.rs1}, {params.imm}"
        return f"{mnemonic} x{params.rd}, x{params.rs1}, {params.imm}"
    if isinstance(params, SType):
        return f"{mnemonic} x{params.rs2}, {params.imm}(x{params.rs1})"
    if isinstance(params, BType):
        target = control_target_label if control_target_label is not None else params.imm
        return f"{mnemonic} x{params.rs1}, x{params.rs2}, {target}"
    if isinstance(params, UType):
        return f"{mnemonic} x{params.rd}, {params.imm}"
    if isinstance(params, JType):
        target = control_target_label if control_target_label is not None else params.imm
        return f"{mnemonic} x{params.rd}, {target}"
    if isinstance(params, EmptyType):
        return mnemonic
    if isinstance(params, DelayType):
        return f"{mnemonic} {params.cycles}"
    if isinstance(params, DMAType):
        return f"{mnemonic} x{params.rd}, x{params.rs1}, x{params.rs2}"
    if isinstance(params, ScaleImmType):
        return f"{mnemonic} e{params.ed}, {params.imm}"
    if isinstance(params, ScaleMemType):
        return f"{mnemonic} e{params.ed}, {params.imm}(x{params.rs1})"
    if isinstance(params, TensorMemType):
        return f"{mnemonic} m{params.mreg}, {params.imm}(x{params.rs1})"
    if isinstance(params, WeightMemType):
        return f"{mnemonic} w{params.slot}, x{params.rs1}"
    if isinstance(params, WeightTensorType):
        return f"{mnemonic} w{params.slot}, m{params.ms}"
    if isinstance(params, MXUAccumulatorType):
        return f"{mnemonic} m{params.mreg}"
    if isinstance(params, MXUMatmulType | MXUMatmulAccType):
        return f"{mnemonic} m{params.ms}, w{params.ws}"
    if isinstance(params, VPUBinaryType):
        return f"{mnemonic} m{params.md}, m{params.ms1}, m{params.ms2}"
    if isinstance(params, VPUUnaryType):
        return f"{mnemonic} m{params.md}, m{params.ms}"
    if isinstance(params, XLUTransposeType):
        return f"{mnemonic} m{params.md}, m{params.ms}"
    raise TypeError(f"Unsupported instruction params for formatting: {type(params).__name__}")


__all__ = ["schedule_assembly_file", "schedule_assembly_text"]
