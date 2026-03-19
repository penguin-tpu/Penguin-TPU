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

## Current Status

The repository is no longer just a scaffold. The current software baseline includes:

- a working scalar functional/performance model with 2-delay-slot control flow
- async channelized DMA between DRAM and VMEM
- whole-register tensor staging with `vload`, `vstore`, and `mxu.push.*`
- MXU matmul and BF16 VPU elementwise execution in `penguin-model`
- executable-bundle support with `program.S`, `program.symbols.json5`,
  `manifest.json5`, and `constants.bin`
- a real `penguin-compile bundle ...` CLI for packaging assembly into runnable bundles
- a real `penguin-model` CLI for running either mapped `.S` programs or bundle
  directories, with optional JSON trace dumping
- checked-in scalar and tensor example programs under `tests/vectors/programs/`

Still deferred for a later milestone:

- direct PyTorch-to-Penguin model lowering
- manifest-level mapping of `constants.bin` into runtime DRAM/VMEM addresses

## Executable Bundle Shape

The shared software/hardware artifact is a small directory bundle:

```text
program.S              Assembly source
program.symbols.json5  IMEM mapping plus input/output/scratch symbols
manifest.json5         Bundle metadata and entry symbol
constants.bin          Packed constant payload
```

Sidecar symbol tables are also checked in next to the test/example `.S` sources in
`tests/vectors/programs/`. The model loaders use those sidecars to assemble programs at
their mapped IMEM addresses.

## Useful Commands

Common verification and example commands from the repo root:

- `uv sync`
- `uv run pytest`
- `uv run penguin-compile bundle --program tests/vectors/programs/scalar/examples/scalar_matmul.S --output-dir /tmp/penguin-bundle`
- `uv run penguin-model --program tests/vectors/programs/scalar/examples/scalar_matmul.S`
- `uv run python examples/matmul_simple.py`
- `uv run python examples/matmul_large.py --trace outputs/examples/matmul_large_trace.json`
- `uv run python examples/linear_large.py --trace outputs/examples/linear_large_trace.json`
- `uv run python scripts/generate_program_symbol_tables.py`

Generated Perfetto traces now live under `outputs/examples/` for example runs and
`outputs/tests/` for pytest program-execution runs.

## Design References

This project draws on several public resources for architectural context, but implements
a deliberately scaled-down subset — one model, one small compute array, static shapes,
no general compiler IR, no multi-chip support. See `AGENTS.md` for detailed notes on
each reference and how it relates to Penguin's scope.

- [Modeling Google TPU](https://github.com/T-K-233/Modeling-Google-TPU) — TPU v5e functional model
- [NPU Model](https://github.com/ucb-ee194-tapeout/npu_model) — tick-based NPU performance model
- [From JAX to VLIW](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation) — TPU compiler stack walkthrough
- [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) — TPU hardware and roofline primer
