"""Directed-program helpers for Penguin scalar model tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .arch_state import ArchState
from .core import PenguinCore
from .instructions import BType, EmptyType, IType, Instruction, JType, RType, SType, UType
from .memory import DRAM_BASE, VMEM_BASE

TEST_DRAM_SIZE = 4 * 1024
TEST_VMEM_SIZE = 4 * 1024
TEST_IMEM_SIZE = 1 * 1024

LabelKind = Literal["label"]


@dataclass(frozen=True, slots=True)
class _Label:
    kind: LabelKind
    name: str


@dataclass(frozen=True, slots=True)
class _PendingBranch:
    mnemonic: str
    rs1: int
    rs2: int
    target: str


@dataclass(frozen=True, slots=True)
class _PendingJump:
    mnemonic: str
    rd: int
    target: str


@dataclass(frozen=True, slots=True)
class _PendingImmediateLoad:
    rd: int
    target: str
    offset: int


ProgramItem = Instruction | _Label | _PendingBranch | _PendingJump | _PendingImmediateLoad


def fresh_arch_state() -> ArchState:
    return ArchState.with_memory_sizes(
        dram_size=TEST_DRAM_SIZE,
        vmem_size=TEST_VMEM_SIZE,
        imem_size=TEST_IMEM_SIZE,
    )


def preload_words(memory, words: dict[int, int]) -> None:
    for offset, value in words.items():
        memory.store_u32(memory.base + offset, value)


def run_scalar_program(
    program: list[Instruction],
    *,
    start_pc: int = 0,
    max_instructions: int | None = None,
    vmem_words: dict[int, int] | None = None,
    dram_words: dict[int, int] | None = None,
) -> tuple[PenguinCore, object]:
    state = fresh_arch_state()
    if vmem_words is not None:
        preload_words(state.vmem, vmem_words)
    if dram_words is not None:
        preload_words(state.dram, dram_words)

    core = PenguinCore(state=state)
    perf = core.execute(program, start_pc=start_pc, max_instructions=max_instructions)
    return core, perf


class ScalarProgramBuilder:
    """Small label-aware builder for directed scalar tests."""

    def __init__(self) -> None:
        self._items: list[ProgramItem] = []

    def label(self, name: str) -> None:
        self._items.append(_Label(kind="label", name=name))

    def empty(self, mnemonic: str) -> None:
        self._items.append(Instruction(mnemonic, EmptyType()))

    def nop(self) -> None:
        self._items.append(Instruction("saddi", IType(rd=0, rs1=0, imm=0)))

    def delay_slots(self) -> None:
        self.nop()
        self.nop()

    def r(self, mnemonic: str, *, rd: int, rs1: int, rs2: int) -> None:
        self._items.append(Instruction(mnemonic, RType(rd=rd, rs1=rs1, rs2=rs2)))

    def i(self, mnemonic: str, *, rd: int, rs1: int, imm: int) -> None:
        self._items.append(Instruction(mnemonic, IType(rd=rd, rs1=rs1, imm=imm)))

    def s(self, mnemonic: str, *, rs1: int, rs2: int, imm: int) -> None:
        self._items.append(Instruction(mnemonic, SType(rs1=rs1, rs2=rs2, imm=imm)))

    def u(self, mnemonic: str, *, rd: int, imm: int) -> None:
        self._items.append(Instruction(mnemonic, UType(rd=rd, imm=imm)))

    def branch(self, mnemonic: str, *, rs1: int, rs2: int, target: str) -> None:
        self._items.append(_PendingBranch(mnemonic=mnemonic, rs1=rs1, rs2=rs2, target=target))

    def jal(self, *, rd: int, target: str) -> None:
        self._items.append(_PendingJump(mnemonic="sjal", rd=rd, target=target))

    def li_label(self, *, rd: int, target: str, offset: int = 0) -> None:
        self._items.append(_PendingImmediateLoad(rd=rd, target=target, offset=offset))

    def build_with_labels(self) -> tuple[list[Instruction], dict[str, int]]:
        label_pc: dict[str, int] = {}
        pc = 0
        for item in self._items:
            if isinstance(item, _Label):
                if item.name in label_pc:
                    raise ValueError(f"Duplicate label {item.name}")
                label_pc[item.name] = pc
                continue
            pc += 4

        program: list[Instruction] = []
        pc = 0
        for item in self._items:
            if isinstance(item, _Label):
                continue
            if isinstance(item, Instruction):
                program.append(item)
            elif isinstance(item, _PendingBranch):
                if item.target not in label_pc:
                    raise ValueError(f"Unknown label {item.target}")
                program.append(
                    Instruction(
                        item.mnemonic,
                        BType(
                            rs1=item.rs1,
                            rs2=item.rs2,
                            imm=label_pc[item.target] - pc,
                        ),
                    )
                )
            elif isinstance(item, _PendingJump):
                if item.target not in label_pc:
                    raise ValueError(f"Unknown label {item.target}")
                program.append(
                    Instruction(
                        item.mnemonic,
                        JType(rd=item.rd, imm=label_pc[item.target] - pc),
                    )
                )
            elif isinstance(item, _PendingImmediateLoad):
                if item.target not in label_pc:
                    raise ValueError(f"Unknown label {item.target}")
                program.append(
                    Instruction(
                        "saddi",
                        IType(
                            rd=item.rd,
                            rs1=0,
                            imm=label_pc[item.target] + item.offset,
                        ),
                    )
                )
            else:
                raise TypeError(f"Unsupported program item: {item!r}")
            pc += 4

        return program, label_pc

    def build(self) -> list[Instruction]:
        program, _ = self.build_with_labels()
        return program


__all__ = [
    "DRAM_BASE",
    "ScalarProgramBuilder",
    "TEST_DRAM_SIZE",
    "TEST_IMEM_SIZE",
    "TEST_VMEM_SIZE",
    "VMEM_BASE",
    "fresh_arch_state",
    "preload_words",
    "run_scalar_program",
]
