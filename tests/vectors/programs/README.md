# Test Programs

This directory holds checked-in assembly source used by the scalar model tests and
examples.

Each `.S` file has a sidecar `*.symbols.json5` file in the same directory. The sidecar
tracks the program IMEM mapping and, for runnable examples, the input/output/scratch
memory regions used by the workload.

The Python model loaders consume these sidecars directly through
`penguin_model.load_mapped_program(...)` and `penguin_model.load_program_symbol_table(...)`.
The verification suite also checks that every checked-in `.S` file has a matching
sidecar and that the recorded program size matches the assembled instruction stream.

- `scalar/` contains the current scalar-only ISA programs.
- `scalar/riscv_isa/` contains flattened imports of upstream `riscv-tests` programs.
- `tensor/` contains tensor/MXU programs used by tests and examples.
- `directed/`, `workloads/`, `performance/`, and `model/` split the programs by test
  purpose.
- `examples/` holds runnable example programs consumed outside the test suite, including
  the fixed-shape Gemma stage programs used to assemble staged executable bundles for
  attention, MLP, and decoder-block examples.

Imported `riscv-tests` programs use a simple host-return convention instead of privilege
mode traps: they terminate with `secall`, write `x10=0` on pass or `x10=1` on fail, and
leave the failing upstream test number in `x3` when they fail.

The checked-in `scalar/riscv_isa/` set covers the supported upstream `rv32ui`
instruction files plus the load/store interaction tests. Two upstream files are
intentionally excluded because they do not match Penguin's current architecture:

- `fence_i.S` depends on self-modifying code and instruction-cache coherence
- `ma_data.S` expects architectural support for misaligned scalar accesses instead of
  Penguin's fatal-stop behavior

Regenerate the scalar sources with:

```bash
uv run python scripts/generate_scalar_programs.py
```

Regenerate the symbol-table sidecars with:

```bash
uv run python scripts/generate_program_symbol_tables.py
```
