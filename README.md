# Penguin-TPU

Reduced-scope repository scaffold for a hyperspecialized accelerator project.

The repository is centered on one simple path:

single PyTorch model -> `penguin-compiler` -> executable package -> `penguin-model` / RTL / FPGA

## Simplified Layout

```text
penguin-compiler/  Direct model-to-assembly Python package
penguin-model/     Reference/performance-model Python package
rtl/               Hardware sources
tests/             Unit tests, cocotb tests, golden vectors, regressions
docs/specs/        Formal architecture and microarchitecture specs
configs/           Versioned hardware/compiler presets
examples/          Tiny models and sample programs
scripts/           Repo-level helper scripts
pyproject.toml     Root uv workspace
```

## Scope Rules

To keep the first version manageable:

- first implement a function-complete RISC-V-like integer scalar core
- then establish the proper memory structure around that scalar core
- then build the testing and validation system around the scalar and memory path
- only after that add matrix processing and other accelerator-specific features
- support one specific PyTorch model first
- skip a general IR and export assembly directly
- separate the compile flow and perf model flow into two top-level Python packages
- let both packages interact through an executable package: assembly + JSON manifest + binary constants
- keep one hardware target first: a single RTL core and one FPGA board
- keep one verification loop first: framework -> penguin-model -> RTL

Manage both Python packages with `uv` workspace commands from the repo root.

Typical workflow:

- `uv sync`
- `uv run penguin-compile`
- `uv run penguin-model`

Scope rules, package responsibilities, verification strategy, and PPA tracking are
documented in `AGENTS.md`. Running project state is tracked in `SOUL.md`.

## Design References

This project draws on several public resources for architectural context, but implements
a deliberately scaled-down subset — one model, one small compute array, static shapes,
no general compiler IR, no multi-chip support. See `AGENTS.md` for detailed notes on
each reference and how it relates to Penguin's scope.

- [Modeling Google TPU](https://github.com/T-K-233/Modeling-Google-TPU) — TPU v5e functional model
- [NPU Model](https://github.com/ucb-ee194-tapeout/npu_model) — tick-based NPU performance model
- [From JAX to VLIW](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation) — TPU compiler stack walkthrough
- [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) — TPU hardware and roofline primer
