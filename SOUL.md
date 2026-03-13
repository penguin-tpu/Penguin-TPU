# Penguin-TPU Soul

## Current State

This repository has moved beyond pure scaffolding.

The first real `penguin-model` execution slice now exists:

- Torch-backed byte-addressed memory
- explicit scalar instruction classes with Python semantics
- a minimal execution core with integer and float register files
- performance counters for cycles, bytes, and per-opcode counts
- a scalar float32 matmul builder that uses the implemented instruction subset
- tests that verify `lw`, `sw`, and matmul correctness against PyTorch

Recent project policy update:

- `docs/specs/` is now explicitly reserved for formal architecture and
  microarchitecture specification documents.
- All repository documents should be human-facing except `AGENTS.md` and `SOUL.md`.
- Agents are expected to create git checkpoint commits at meaningful stable points.
- `SOUL.md` should track intentions and unresolved questions alongside technical state.

Current checkpoint intention:

- create the initial framework commit once the repository scaffold, package split,
  agent guidance, and planning documents are coherent
- treat this checkpoint as a baseline for future implementation work, not as proof of
  working functionality

What exists now:

- top-level `uv` workspace in `pyproject.toml`
- two top-level Python package shells:
  - `penguin-compiler`
  - `penguin-model`
- package metadata and placeholder CLI entry points
- a working scalar functional model in `penguin-model` for:
  - `lw`
  - `sw`
  - `flw`
  - `fsw`
  - `fmul.s`
  - `fadd.s`
- top-level docs:
  - `README.md`
  - `docs/specs/scalar-functional-subset.md`
- hardware directory split under `rtl/`:
  - `rtl/penguin_tpu`
  - `rtl/vivado_ips`
- top-level `tests/` directory structure for vectors, unit tests, cocotb, and regressions

What does not exist yet:

- actual compiler logic
- assembly parser / bundle-driven model execution logic
- executable package loader/writer implementation
- ISA specification
- `manifest.json` schema
- tensor layout specification
- golden vectors
- working RTL testbench flow
- FPGA bring-up scripts

## Progress Summary

The project has converged on a deliberate architectural stance:

- keep scope narrow
- target one fixed model first
- avoid a general IR
- separate compiler and model into two Python packages
- let both packages communicate through an executable package

This is a good decision for getting a first vertical slice working quickly.

## Ground Truth About The Code

`penguin-compiler` currently contains only:

- package metadata
- a minimal `ExecutableBundle` dataclass
- a CLI stub that exits with “not implemented yet”

`penguin-model` now contains:

- Torch-backed main memory modeled after the referenced Google TPU Python memory style
- executable instruction semantics for a minimal scalar subset
- a small execution core and simple performance counters
- a scalar matmul program builder used for the first functional tests
- a CLI that no longer lies about total package emptiness, but still does not load bundles

There is no shared implementation beyond those stubs.

## Immediate TODOs

### 1. Freeze the executable package contract

Define exactly:

- assembly file naming
- `manifest.json` schema
- constants blob format
- input/output blob conventions
- address and memory-region encoding

Without this, the compiler, model, and RTL will drift.

### 2. Write the missing specs

Add files under `docs/specs/` for:

- architecture specification
- microarchitecture specification
- ISA semantics
- tensor layout and packing
- memory map
- configuration parameters visible to software and hardware

### 3. Extend the first vertical slice

The smallest software-side slice now exists. The next step is to stop using only
Python-built programs and move toward the real contract:

- define textual assembly syntax
- add program parsing/loading
- define bundle metadata for memory regions
- place one golden matmul vector under `tests/vectors/`
- keep matching PyTorch as the source of truth

### 4. Decide how shared code should live

Right now `ExecutableBundle` is duplicated in both Python packages. That is acceptable as
a temporary scaffold, but it will become a maintenance problem if both copies evolve.

Open question:

- keep the duplication for loose coupling
- or introduce a tiny shared package later, such as `penguin-common`

For now, do not introduce a third package unless the interface is actually stabilizing.

### 5. Add real tests before adding real features

Priority order:

- package unit tests
- golden-vector tests
- model-vs-reference tests
- RTL cocotb tests

## Design Caveats

### No general IR

This is intentional, but brittle.

It is a good choice if:

- the model topology is fixed
- shapes are static
- the compiler is really a specialized exporter

It becomes risky if:

- the supported model family grows
- layer ordering changes often
- kernel scheduling becomes nontrivial

If that happens, revisit the no-IR decision, but not before a working vertical slice.

### Executable package is still underspecified

Everyone agrees on the rough shape:

- assembly
- JSON manifest
- binary constants

But the exact schema is missing. This is the single highest-risk ambiguity in the repo.

### Formal specs are now a first-class deliverable

The project now explicitly expects `docs/specs/` to hold formal arch/uarch specs rather
than casual notes. That is the right direction, but it raises the bar on discipline:

- terminology needs to be consistent
- architecture-visible behavior and implementation details need clear separation
- the executable package format must line up with those specs

### Performance modeling is still placeholder-grade

The model now records instruction counts, cycle counts, and byte traffic, but the cycle
latencies are simple placeholders chosen for early bring-up. They are useful for smoke
tests and relative accounting, not for trustworthy PPA conclusions.

### Hardware tree exists, but its implementation status is unknown

The repo has:

- `rtl/penguin_tpu`
- `rtl/vivado_ips`

The planning docs are aligned to that structure, but this memory file does not assume
the hardware internals are complete or validated. A future agent should inspect those
directories before making hardware claims.

## External Design References

Four external resources are documented in `AGENTS.md` and summarized in `README.md`.
They cover TPU functional modeling, NPU performance modeling, TPU compiler internals, and
TPU hardware organization. All are design references only — Penguin is a scaled-down
design and does not need all features from any of them. When adopting ideas from these
references, scope them to Penguin's immediate vertical slice.

## MVP Milestones

These are the near-term targets, in order. Do not skip ahead.

### Milestone 1 — not started

Make one hand-written assembly program run in both `penguin-model` and RTL.

### Milestone 2 — not started

Export one specific PyTorch model into that assembly format and match PyTorch outputs in
`penguin-model`.

### Milestone 3 — not started

Run the same compiled artifact on RTL through cocotb.

### Milestone 4 — not started

Deploy the same artifact on one FPGA board and collect basic timing and utilization
reports.

Only after these four milestones should you consider adding a general IR, expanding
operator coverage, adding JAX support, or splitting the simulator into richer functional
and performance submodels.

## Questions And Doubts

- Should the initial architectural subset keep both integer word loads/stores and float
  loads/stores, or should the eventual ISA surface expose only one memory-access form?
- Should assembly syntax follow RISC-V conventions literally, or remain only
  RISC-V-inspired while the project is still narrowing scope?
- Should `manifest.json` describe only static constants, or also runtime input/output buffers?
- Will inputs be fixed test vectors or variable runtime payloads?
- Is the assembly textual only, or should a binary encoding be emitted early as well?
- Should `penguin-model` parse assembly directly, or should both sides consume a lower-level encoded format derived from it?
- How much of Vivado/IP generation should live in-repo versus being regenerated externally?
- Does the hardware plan assume one clock domain and one memory space, or are there already multiple domains/interfaces in mind?

## Recommended Next Move

The highest-leverage next step is not more scaffolding. It is one concrete spec file for
the executable package plus one tiny end-to-end example that exercises it.

If a future agent needs a starting sequence, use this:

1. Add `docs/specs/executable-package.md`.
2. Define textual assembly syntax for the implemented scalar subset.
3. Define one example package under `tests/vectors/`.
4. Implement a loader/parser in `penguin-model`.
5. Make `penguin-compile` emit that same format.
6. Only then connect the format to RTL.
