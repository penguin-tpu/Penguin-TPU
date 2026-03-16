# penguin-model

Python package for the executable architectural model:

- load Penguin assembly program bundles
- load checked-in `.S` programs plus adjacent `*.symbols.json5` sidecars
- execute reference instruction semantics
- report output tensors
- estimate rough cycle counts and utilization counters
- dump JSON traces for cycle-level verification

The model consumes the same bundle shape emitted by `penguin-compiler`.

## Current Surfaces

The current model includes:

- scalar integer execution with 2 architected branch/jump delay slots
- channelized async DMA between DRAM and VMEM
- whole-register tensor staging with `vload`, `vstore`, and `mxu.push.*`
- MXU matmul execution
- BF16 VPU elementwise ops: `vadd`, `vmul`, `vmax`, `vmin`, `vrelu`, `vmov`
- bundle and symbol-table loaders:
  - `load_executable_bundle(...)`
  - `load_mapped_program(...)`
  - `load_program_symbol_table(...)`
  - `preload_loaded_bundle_symbols(...)`
- the `penguin-model` CLI for running mapped programs or bundle directories

## Typical Usage

Run the full verification suite:

```bash
uv run pytest
```

Run a checked-in example workload:

```bash
uv run python examples/matmul_large.py --trace outputs/examples/matmul_large_trace.json
```

Generate the current PI0 workload roofline report and plot:

```bash
uv run python examples/roofline_report.py --output outputs/examples/penguin_roofline.png
```

Example traces default to `outputs/examples/`. Under pytest, direct program executions
also emit JSON traces under `outputs/tests/`.

Load a checked-in program from source with its mapped IMEM base:

```python
from penguin_model import PenguinCore, load_mapped_program

program = load_mapped_program("tests/vectors/programs/tensor/examples/matmul_large.S")
core = PenguinCore()
perf = core.execute(program)
```

`load_mapped_program(...)` reads the adjacent `*.symbols.json5` sidecar, assembles the
program at the symbol table's IMEM base address, and validates the program size against
the sidecar entry.

Run a bundle directory through the CLI:

```bash
uv run penguin-model --bundle /tmp/penguin-matmul-large --trace /tmp/penguin-trace.json
```

When a bundle contains file-backed symbol payloads, the model loader can stage them into
their mapped DRAM/VMEM/IMEM regions before execution through
`preload_loaded_bundle_symbols(...)`.

Known deferred bundle work is tracked in `TODO.md`.
