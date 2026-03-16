# Test Programs

This directory holds checked-in assembly source used by the scalar model tests and
examples.

- `scalar/` contains the current scalar-only ISA programs.
- `tensor/` contains tensor/MXU programs used by tests and examples.
- `directed/`, `workloads/`, `performance/`, and `model/` split the programs by test
  purpose.
- `examples/` holds runnable example programs consumed outside the test suite.

Regenerate the scalar sources with:

```bash
uv run python scripts/generate_scalar_programs.py
```
