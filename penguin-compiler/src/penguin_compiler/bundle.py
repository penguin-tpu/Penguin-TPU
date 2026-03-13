"""Shared bundle definitions for compiler outputs."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutableBundle:
    """File-level contract between compiler, model, RTL, and FPGA loaders."""

    program: Path
    manifest: Path
    constants: Path

