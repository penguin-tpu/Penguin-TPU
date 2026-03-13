# Penguin-TPU Agent Guide

## Mission

This repository is for a reduced-scope, hyperspecialized accelerator project.

The intended end-to-end path is:

single PyTorch model -> `penguin-compiler` -> executable package -> `penguin-model` -> RTL -> FPGA

This is intentionally not a general-purpose compiler stack. The current design choice is
to skip a full IR and directly export one fixed model or one very narrow model family
into Penguin assembly plus metadata and packed constants.

## Project Goals

- compile one specific PyTorch model into a Penguin executable package
- execute that package in a Python reference/performance model
- execute the same package in RTL
- eventually deploy the same package to FPGA
- verify correctness first, then track coarse PPA metrics

## Repo Rules

- Documents in this repository must be written for human users unless the file is
  `AGENTS.md` or `SOUL.md`.
- `docs/specs/` is reserved for formal specification documents, especially
  architecture and microarchitecture specifications. Those documents should read like
  real arch/uarch specs, not rough notes.
- `AGENTS.md` is the agent onboarding guide and operating manual for the repo.
- `SOUL.md` is the running project memory. Record major changes, intentions, open
  questions, caveats, and state transitions there.
- If `SOUL.md` becomes too large, compact it by summarizing stale details and removing
  information that no longer matters.

## Version Control Policy

- Manage git history actively.
- When the repository reaches a stable checkpoint worth keeping, create a git commit.
- A checkpoint usually means one of:
  - a meaningful slice of functionality is implemented
  - a major bug is fixed
  - an interface or spec is stabilized
  - an important reorganization is complete and coherent
- Use your own identity as the commit author when creating those commits.
- Do not create noisy checkpoint commits for tiny, unstable edits.
- Record the intention of the checkpoint and any unresolved questions in `SOUL.md`
  before or alongside the work.

## Top-Level Structure

```text
pyproject.toml          Root `uv` workspace
README.md               Human-facing repo summary
SOUL.md                 Current state, TODOs, caveats, questions
docs/                   Architecture notes and specs
penguin-compiler/       Python package for direct model-to-assembly export
penguin-model/          Python package for reference/performance execution
rtl/                    Hardware sources
tests/                  Cross-stack vectors, unit tests, cocotb, regressions
configs/                Hardware or tool configuration presets
examples/               Tiny example programs and models
scripts/                Repo helpers
```

## Package Responsibilities

### `penguin-compiler`

Owns the direct software export path.

Expected responsibilities:

- inspect one known PyTorch topology
- validate shapes and parameters
- map layers to fixed assembly templates
- emit the executable package

Current package layout:

```text
penguin-compiler/
  pyproject.toml
  src/penguin_compiler/
  tests/
```

### `penguin-model`

Owns the executable architectural model and rough performance model.

Expected responsibilities:

- load the executable package
- execute assembly semantics
- produce output tensors
- estimate cycle counts and simple counters

Current package layout:

```text
penguin-model/
  pyproject.toml
  src/penguin_model/
  tests/
```

### `rtl`

Owns the hardware side.

Current intended substructure:

```text
rtl/
  penguin_tpu/   Core RTL and local hardware integration
  vivado_ips/    Vendor IP wrappers or generated IP collateral
```

## Executable Package Contract

The main shared artifact is the executable package. Treat this as the interface between
software and hardware.

Minimum contents:

- assembly program
- `manifest.json`
- binary constants blob

Optional, depending on the workflow:

- fixed input blob
- expected output blob
- expected cycle count or checksum for regression tests

Assembly is the executable boundary, but assembly alone is not enough. The manifest is
required to describe memory layout, tensor metadata, addresses, and runtime expectations.

## Specs And Documentation

Use `docs/` as the human-facing documentation tree.

Expected documentation split:

- `docs/specs/`: formal architecture and microarchitecture specifications
- `docs/architecture/`: higher-level repository and subsystem design documents
- `README.md`: concise project overview and usage entry point

Specification documents under `docs/specs/` should eventually cover:

- architecture-visible ISA behavior
- memory map and memory consistency assumptions
- tensor layout and packing rules
- microarchitectural organization and key implementation-visible constraints
- configuration parameters shared by software and hardware

## Scope Constraints

Keep these constraints unless the user explicitly broadens scope:

- one specific PyTorch model first
- static shapes only
- no general IR
- small operator set first: GEMM, add, ReLU, data movement
- one hardware configuration first
- one board target first

## Verification Strategy

The intended correctness loop is:

PyTorch eager output -> `penguin-model` output -> RTL output -> FPGA output

Planned verification organization:

- `penguin-compiler/tests/`: exporter and assembly emission tests
- `penguin-model/tests/`: ISA/model tests
- `tests/unit/`: shared or system-local unit tests
- `tests/cocotb/`: RTL tests driven by the same executable package
- `tests/vectors/`: assembly, manifests, constants, inputs, expected outputs
- `tests/regressions/`: small benchmark suites and summary results

## High-Level Commands

Run all commands from the repo root unless there is a good reason not to.

Current workspace commands:

- `uv sync`
- `uv run penguin-compile`
- `uv run penguin-model`

Useful inspection commands:

- `find . -maxdepth 3 -type f | sort`
- `rg -n "TODO|FIXME|XXX" .`
- `git status --short`
- `git log --oneline -n 10`
- `sed -n '1,220p' README.md`
- `sed -n '1,320p' docs/architecture/repo-plan.md`
- `sed -n '1,260p' AGENTS.md`
- `sed -n '1,320p' SOUL.md`

Expected future commands once implementation exists:

- `uv run pytest`
- `uv run pytest penguin-compiler/tests`
- `uv run pytest penguin-model/tests`
- `uv run pytest tests/unit`
- `uv run pytest tests/cocotb`

## Important Current Truths

- The repository structure and package boundaries are mostly planned, not yet implemented.
- The CLIs `penguin-compile` and `penguin-model` currently exist as stubs and exit with
  “not implemented yet”.
- There is not yet a finalized ISA spec or manifest schema in `docs/specs/`.
- There is not yet a real compiler, model executor, regression harness, or RTL test flow.
- Git checkpoints should be created intentionally when the repo reaches a meaningful
  stable state, not for every minor edit.

## How To Work In This Repo

When adding code, prefer this order:

1. Write or refine the spec in `docs/specs/` if the interface is not settled.
2. Implement the smallest vertical slice in `penguin-compiler` and `penguin-model`.
3. Add a golden vector under `tests/vectors/`.
4. Add tests before broadening functionality.
5. Only then connect the same artifact into RTL.

When reaching a stable checkpoint:

1. Update `SOUL.md` with what changed, why it matters, and what is still unresolved.
2. Review `git status`.
3. Create a git commit if the change is cohesive and worth future reference.

If you need project status, unresolved caveats, or open design questions, read `SOUL.md`
next.
