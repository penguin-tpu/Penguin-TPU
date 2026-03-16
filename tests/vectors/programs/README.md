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
- `tensor/` contains tensor/MXU programs used by tests and examples.
- `directed/`, `workloads/`, `performance/`, and `model/` split the programs by test
  purpose.
- `examples/` holds runnable example programs consumed outside the test suite.

Regenerate the scalar sources with:

```bash
uv run python scripts/generate_scalar_programs.py
```

Regenerate the symbol-table sidecars with:

```bash
uv run python scripts/generate_program_symbol_tables.py
```
