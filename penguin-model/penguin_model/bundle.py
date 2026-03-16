"""Bundle definitions for model-side loaders."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutableBundle:
    """File-level contract consumed by the model and hardware flows."""

    program: Path
    manifest: Path
    constants: Path
