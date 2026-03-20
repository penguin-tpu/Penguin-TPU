from __future__ import annotations

from penguin_compiler import schedule_assembly_text
from penguin_model import (
    DelayType,
    IMEM_BASE,
    IType,
    Instruction,
    MXUAccumulatorType,
    MXUMatmulType,
    TensorMemType,
    VMEM_BASE,
    WeightTensorType,
    assemble_text,
)


def test_scheduler_inserts_delay_between_tensor_producer_and_consumer() -> None:
    scheduled = schedule_assembly_text(
        """
        li x1, VMEM_BASE + 0x0000
        li x2, VMEM_BASE + 0x1000
        vload m1, 0(x1)
        vload m3, 0(x2)
        vmatpush.weight.mxu0 w0, m3
        vmatmul.mxu0 a0, m1, w0
        vmatpop.bf16.acc.mxu0 m2, a0
        """,
        base_address=IMEM_BASE,
    )

    program = assemble_text(scheduled, base_address=IMEM_BASE)

    assert list(program) == [
        Instruction("addi", IType(rd=1, rs1=0, imm=VMEM_BASE + 0x0000)),
        Instruction("addi", IType(rd=2, rs1=0, imm=VMEM_BASE + 0x1000)),
        Instruction("vload", TensorMemType(mreg=1, rs1=1, imm=0)),
        Instruction("vload", TensorMemType(mreg=3, rs1=2, imm=0)),
        Instruction("delay", DelayType(cycles=63)),
        Instruction("vmatpush.weight.mxu0", WeightTensorType(slot=0, ms=3)),
        Instruction("delay", DelayType(cycles=63)),
        Instruction("vmatmul.mxu0", MXUMatmulType(ms=1, ws=0, acc=0)),
        Instruction("delay", DelayType(cycles=63)),
        Instruction("vmatpop.bf16.acc.mxu0", MXUAccumulatorType(mreg=2, acc=0)),
    ]


def test_exported_model_stage_bundle_contains_scheduled_delay(tmp_path) -> None:
    from penguin_compiler import deterministic_hidden, export_pytorch_model_package, make_fixed_gemma_attention

    package = export_pytorch_model_package(
        make_fixed_gemma_attention(),
        deterministic_hidden(),
        tmp_path / "scheduled_package",
    )

    q_proj_program = package.stage_bundles["q_proj"] / "gemma_linear_64x32.S"
    assert "delay " in q_proj_program.read_text()
