# Penguin-TPU

Reduced-scope repository scaffold for a hyperspecialized accelerator project.

The repository is centered on one simple path:

single PyTorch model -> `penguin-compiler` -> executable package -> `penguin-model` / RTL / FPGA

## Simplified Layout

```text
penguin-compiler/  Direct model-to-assembly Python package
penguin-model/     Reference/performance-model Python package
rtl/               Hardware sources
tests/     Unit tests, cocotb tests, golden vectors, regressions
docs/      Architecture notes and specs
configs/   Versioned hardware/compiler presets
examples/  Tiny models and sample programs
scripts/   Repo-level helper scripts
pyproject.toml     Root uv workspace
```

## Scope Rules

To keep the first version manageable:

- support one specific PyTorch model first
- skip a general IR and export assembly directly
- separate the compile flow and perf model flow into two top-level Python packages
- support a tiny operator set first: GEMM, add, ReLU, data movement
- let both packages interact through an executable package: assembly + JSON manifest + binary constants
- keep one hardware target first: a single RTL core and one FPGA board
- keep one verification loop first: framework -> penguin-model -> RTL

Manage both Python packages with `uv` workspace commands from the repo root.

Typical workflow:

- `uv sync`
- `uv run penguin-compile`
- `uv run penguin-model`

Detailed structure and validation guidance is in
`docs/architecture/repo-plan.md`.
