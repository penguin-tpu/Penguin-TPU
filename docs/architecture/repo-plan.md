# Penguin-TPU Repository Plan

## 1. Reduced Scope Goal

The first version should optimize for one working vertical slice, not full generality.

Target path:

single PyTorch model -> `penguin-compiler` -> executable package -> `penguin-model` / RTL / FPGA

This plan assumes the hardware is intentionally hyperspecialized for one model or one
very narrow model family. Under that assumption, a full compiler IR is unnecessary
overhead.

Deliberately defer for later:

- JAX support
- a general graph IR
- a separate performance model
- multiple hardware backends
- large operator coverage
- aggressive optimization passes
- power analysis beyond coarse estimates

## 2. Simplified Directory Structure

```text
Penguin-TPU/
  README.md
  pyproject.toml
  penguin-compiler/
    pyproject.toml
    src/penguin_compiler/
    tests/
  penguin-model/
    pyproject.toml
    src/penguin_model/
    tests/
  rtl/
    penguin_tpu/     Core RTL and local hardware integration
    vivado_ips/      Vendor IP wrappers or generated IP collateral
  tests/
    unit/            Python and RTL-local tests
    cocotb/          End-to-end RTL tests driven from Python
    vectors/
      programs/      Assembly programs
      data/          Packed weights, activations, golden outputs
      manifests/     Program metadata and memory layout
    regressions/     Small workload suites and result summaries
  docs/
    architecture/    Repo plan and architecture notes
    specs/           ISA and programming model specs
  configs/           Hardware configuration presets
  examples/          Tiny models and hand-written programs
  scripts/           Helper scripts for build and regression
```

This keeps the project small, but still makes one useful separation:

- `penguin-compiler/` owns translation from PyTorch into executable artifacts
- `penguin-model/` owns executable architectural and performance interpretation of those artifacts

That is a good split because compiler evolution and model calibration tend to change for
different reasons.

## 3. What Each Combined Module Owns

### `penguin-compiler/`

Keep this as a narrow model adapter, not a general compiler framework:

- accept one known model topology
- validate shapes and parameters
- map each layer to one fixed assembly template
- emit a stable program bundle

Recommended package shape:

- `src/penguin_compiler/export/`: PyTorch model introspection
- `src/penguin_compiler/codegen/`: assembly emission
- `src/penguin_compiler/pack/`: weight and tensor packing
- `src/penguin_compiler/cli/`: export tools

### `penguin-model/`

This package is the executable spec and rough performance estimator.

Recommended package shape:

- `src/penguin_model/isa/`: instruction semantics
- `src/penguin_model/model/`: program execution
- `src/penguin_model/trace/`: traces, counters, reporting
- `src/penguin_model/cli/`: run and profile tools

This keeps the performance model independent from export logic while still reusing the
same artifacts. It also makes it easier to test and version the perf model separately.

### `rtl/`

This is all implementation-side logic.

- `penguin_tpu/`: synthesizable core and local wrappers
- `vivado_ips/`: vendor IP wrappers or generated IP collateral

The FPGA flow should reuse the RTL directly. Avoid creating a separate hardware
architecture layer unless the design grows enough to require it.

### `tests/`

All verification lives here so the same inputs and harnesses are reused across layers.

- `unit/`: compiler and simulator unit tests
- `cocotb/`: run assembly programs on RTL and compare against the simulator
- `vectors/`: fixed tensors, constants, assembly, and expected outputs
- `regressions/`: small benchmark suites and PPA summaries

## 4. Simplified Interaction Model

The simplest maintainable interaction is artifact-driven.

### Shared artifacts

The compiler package should emit one executable package:

- assembly text
- `manifest.json`
- binary constant data

The same artifacts are consumed by:

- `penguin-model/` for golden execution and cycle estimation
- `tests/cocotb/` for RTL execution
- board bring-up logic around `rtl/` for on-hardware execution

This avoids building separate loaders or test formats for each layer.

### Why assembly is the right boundary

Yes, assembly is the right executable boundary for this project, because:

- it is easy to inspect and diff
- it matches what RTL should execute
- it keeps the perf model and hardware path aligned

But assembly alone is too weak as a full interface. The practical boundary should be:

- assembly program
- packed binary constants and optional fixed IO data
- JSON manifest describing layout, addresses, and expected outputs

Think of the assembly as the core contract and the manifest as the glue that keeps every
consumer consistent.

### One source of truth

Keep only three core contracts in `docs/specs/`:

1. ISA behavior
2. tensor and memory layout
3. configuration parameters visible to both software and hardware

Everything else should derive from those contracts.

There is intentionally no standalone IR contract in this version.

## 5. Scope Constraints For The First Version

To avoid an unbounded project, keep the MVP intentionally narrow.

### Model scope

- support one specific PyTorch model
- support static shapes only
- keep layer ordering fixed
- reject unsupported variants early instead of trying to be flexible

### Operator set

- GEMM
- elementwise add
- ReLU
- explicit load/store or DMA-style data movement

### Datatypes

- one activation type first, such as int8
- one accumulation type first, such as int32

### Hardware

- one compute array shape
- one scratchpad organization
- one FPGA board target

These limits are not architectural weaknesses. They are how you get a complete,
debuggable system early.

## 6. Simplified Verification And Validation Plan

Use one main correctness loop:

framework output -> simulator output -> RTL output -> FPGA output

### A. Software-side verification

Verify `penguin-compiler` with:

- fixed unit tests for each supported layer mapping
- deterministic assembly output tests
- manifest validation tests
- comparison of `penguin-model` outputs against PyTorch eager outputs

The main goal is to prove that the direct exporter preserves model meaning.

### B. RTL verification

Use one primary RTL strategy:

- cocotb + pytest

Feed the exact compiler-generated artifacts into RTL tests and compare against the
`penguin-model` outputs. This is enough to get high value without building a heavyweight
verification environment too early.

Use a few targeted assertions in RTL for:

- FIFO overflow/underflow
- valid/ready protocol correctness
- illegal instruction detection

Formal verification can wait until the design stabilizes.

### C. FPGA validation

The board run should reuse the same test vectors already passing in simulation.

At first, validate only:

- final tensor correctness
- end-to-end latency
- coarse hardware counters if available

Do not build deep on-chip debug infrastructure until the simulator and RTL path are
already trustworthy.

## 7. Simplified PPA Plan

Keep PPA tracking lightweight at first.

### Track only these metrics initially

- cycle count from the simulator
- RTL latency for selected programs
- post-synthesis LUT, FF, BRAM, DSP usage
- post-route Fmax

Defer detailed power optimization until the architecture and compiler are stable.

### Practical workflow

1. Estimate cycles in `penguin-model/`.
2. Compare against RTL for a few representative programs.
3. Run Vivado on one board target.
4. Save results under `tests/regressions/`.
5. Reject changes that break correctness or exceed simple area/timing budgets.

That is enough to guide meaningful design decisions without overbuilding infrastructure.

## 8. Recommended MVP Milestones

### Milestone 1

Make one hand-written assembly program run in both `penguin-model` and RTL.

### Milestone 2

Export one specific PyTorch model into that assembly format and match PyTorch outputs in
`penguin-model`.

### Milestone 3

Run the same compiled artifact on RTL through cocotb.

### Milestone 4

Deploy the same artifact on one FPGA board and collect basic timing and utilization
reports.

Only after these four milestones should you consider adding a general IR, expanding
operator coverage, adding JAX support, or splitting the simulator into richer
functional and performance submodels.
