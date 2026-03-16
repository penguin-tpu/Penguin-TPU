# Penguin-TPU Agent Guide

## Mission

This repository is for a reduced-scope, hyperspecialized accelerator project.

The intended end-to-end path is:

single PyTorch model -> `penguin-compiler` -> executable package -> `penguin-model` -> RTL -> FPGA

This is intentionally not a general-purpose compiler stack. The current design choice is
to skip a full IR and directly export one fixed model or one very narrow model family
into Penguin assembly plus metadata and packed constants.

## Project Goals

- implement a function-complete RISC-V-like integer scalar core first
- establish a proper memory structure around that scalar core
- build the testing and validation system before broadening accelerator features
- only then add matrix processing and other Penguin-specific acceleration features
- compile one specific PyTorch model into a Penguin executable package
- execute that package in a Python reference/performance model
- execute the same package in RTL
- eventually deploy the same package to FPGA
- verify correctness first, then track coarse PPA metrics

## Design References

Penguin-TPU is a scaled-down, hyperspecialized accelerator. It does not aim to replicate
the full feature set of any production TPU or NPU. The following external resources
informed the architecture and software modeling approach. They are inspiration and
context, not specifications to follow literally.

- [Modeling Google TPU](https://github.com/T-K-233/Modeling-Google-TPU) — a best-effort
  functional model of Google TPU v5e-1, with instruction semantics inferred from LLO
  compiler output. Useful for understanding TPU instruction-level behavior, but Penguin
  targets a much simpler ISA and a smaller compute array.

- [NPU Model (ucb-ee194-tapeout)](https://github.com/ucb-ee194-tapeout/npu_model) — a
  tick-based, execution-driven performance model for an NPU architecture. Its pipeline
  structure (IFU → IDU → EXUs), claim-based handshaking, and Perfetto trace generation
  are relevant references for the `penguin-model` package. Penguin uses a narrower
  operator set and does not need all of the configurability exposed there.

- [From JAX to VLIW: Tracing a Computation Through the TPU Compiler Stack](https://patricktoulme.substack.com/p/from-jax-to-vliw-tracing-a-computation)
  — a walkthrough of the full HLO → LLO → VLIW compilation path on a real TPU v6e.
  Covers fusion, layout assignment, memory space annotation, async DMA scheduling, and
  VLIW bundle packing. Penguin deliberately skips most of this complexity: there is no
  general HLO/LLO pipeline, no VLIW scheduling, and no multi-fusion orchestration. Still
  valuable for understanding what a mature TPU compiler does and why certain architectural
  features exist.

- [How to Think About TPUs (JAX Scaling Book, Part 2)](https://jax-ml.github.io/scaling-book/tpus/)
  — detailed explanation of TPU hardware organization: MXU systolic arrays, VPU, VMEM,
  HBM, ICI networking, and roofline analysis. Penguin implements a single small systolic
  array with a simple scratchpad, not the full memory hierarchy or network topology
  described here. The roofline reasoning and arithmetic intensity discussion are directly
  applicable to sizing decisions in Penguin.

Features from these references should only be adopted when they serve the immediate
vertical slice.

For Python coding style, follow the conventions used in
[Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite).

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
- If an agent task lasts more than 3 minutes, then after finishing the task, invoke the
  `slackbot` skill to send a short Slack message summarizing what was completed.

## Version Control Policy

- Manage git history actively.
- When the repository reaches a stable checkpoint worth keeping, create a git commit.
- A checkpoint usually means one of:
  - a meaningful slice of functionality is implemented
  - a major bug is fixed
  - an interface or spec is stabilized
  - an important reorganization is complete and coherent
- Use your own identity as the commit author when creating those commits, not the user's, so that agent-authored changes are distinguishable from human-authored ones, and you get the credits.
- Do not create noisy checkpoint commits for tiny, unstable edits.
- Record the intention of the checkpoint and any unresolved questions in `SOUL.md`
  before or alongside the work.

## Top-Level Structure

```text
pyproject.toml          Root `uv` workspace
README.md               Human-facing repo summary
AGENTS.md               Agent onboarding, long-term planning, scope rules
SOUL.md                 Running project memory, TODOs, caveats, questions
docs/specs/             Formal architecture and microarchitecture specs
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

Owns the direct software export path. Keep this as a narrow model adapter, not a general
compiler framework.

Expected responsibilities:

- accept one known model topology
- validate shapes and parameters
- map each layer to one fixed assembly template
- emit the executable package

Recommended internal layout:

```text
penguin-compiler/
  pyproject.toml
  src/penguin_compiler/
    export/    PyTorch model introspection
    codegen/   assembly emission
    pack/      weight and tensor packing
    cli/       export tools
  tests/
```

### `penguin-model`

Owns the executable architectural model and rough performance model.

Expected responsibilities:

- load the executable package
- execute assembly semantics
- produce output tensors
- estimate cycle counts and simple counters

Recommended internal layout:

```text
penguin-model/
  pyproject.toml
  src/penguin_model/
    isa/       instruction semantics
    model/     program execution
    trace/     traces, counters, reporting
    cli/       run and profile tools
  tests/
```

### `rtl`

Owns the hardware side. The FPGA flow should reuse the RTL directly. Avoid creating a
separate hardware architecture layer unless the design grows enough to require it.

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

Assembly is the right executable boundary because it is easy to inspect and diff, it
matches what RTL should execute, and it keeps the perf model and hardware path aligned.
But assembly alone is not enough. The manifest is required to describe memory layout,
tensor metadata, addresses, and runtime expectations.

The same artifacts are consumed by:

- `penguin-model/` for golden execution and cycle estimation
- `tests/cocotb/` for RTL execution
- board bring-up logic around `rtl/` for on-hardware execution

This avoids building separate loaders or test formats for each layer.

Keep only three core contracts in `docs/specs/`:

1. ISA behavior
2. tensor and memory layout
3. configuration parameters visible to both software and hardware

Everything else should derive from those contracts. There is intentionally no standalone
IR contract in this version.

## Specs And Documentation

Use `docs/specs/` for formal specification documents. These should read like real
arch/uarch specs, not rough notes. `README.md` serves as the concise project overview and
usage entry point. Long-term planning lives in `AGENTS.md`; running state in `SOUL.md`.

Specification documents under `docs/specs/` should eventually cover:

- architecture-visible ISA behavior
- memory map and memory consistency assumptions
- tensor layout and packing rules
- microarchitectural organization and key implementation-visible constraints
- configuration parameters shared by software and hardware

## Scope Constraints

Keep these constraints unless the user explicitly broadens scope:

- scalar-core bring-up comes before matrix acceleration
- the first hardware goal is a function-complete RISC-V-like integer scalar core
- the next system goal is a proper memory structure around that core
- the next infrastructure goal is a real testing and validation loop
- matrix processing is a later layer on top of the validated scalar/memory base
- one specific PyTorch model first, static shapes only, fixed layer ordering
- reject unsupported variants early instead of trying to be flexible
- no general IR
- one activation datatype first (such as int8), one accumulation type first (such as int32)
- one compute array shape, one scratchpad organization
- one hardware configuration first, one FPGA board target first

Deliberately defer until after a working vertical slice:

- JAX support
- a general graph IR
- multiple hardware backends
- large operator coverage
- aggressive optimization passes
- power analysis beyond coarse estimates

## Verification Strategy

The intended correctness loop is:

PyTorch eager output -> `penguin-model` output -> RTL output -> FPGA output

Recommended implementation order:

1. verify scalar ISA semantics
2. verify memory behavior and memory-system invariants
3. verify the scalar core in RTL
4. verify the executable-package flow end to end
5. only then extend the same harnesses to matrix instructions and accelerator features

Planned verification organization:

- `penguin-compiler/tests/`: exporter and assembly emission tests
- `penguin-model/tests/`: ISA/model tests
- `tests/unit/`: shared or system-local unit tests
- `tests/cocotb/`: RTL tests driven by the same executable package
- `tests/vectors/`: assembly, manifests, constants, inputs, expected outputs
- `tests/regressions/`: small benchmark suites and summary results

### Software-side verification

Verify `penguin-compiler` with fixed unit tests for each supported layer mapping,
deterministic assembly output tests, manifest validation, and comparison of
`penguin-model` outputs against PyTorch eager outputs. The main goal is to prove that
the exporter preserves model meaning.

### RTL verification

Primary strategy is cocotb + pytest. Feed the exact compiler-generated artifacts into
RTL tests and compare against `penguin-model` outputs. Use targeted assertions for FIFO
overflow/underflow, valid/ready protocol correctness, and illegal instruction detection.
Formal verification can wait until the design stabilizes.

### FPGA validation

The board run should reuse the same test vectors already passing in simulation. Validate
final tensor correctness, end-to-end latency, and coarse hardware counters if available.
Do not build deep on-chip debug infrastructure until the simulator and RTL path are
already trustworthy.

## PPA Tracking

Keep PPA tracking lightweight at first.

Track only:

- cycle count from the simulator
- RTL latency for selected programs
- post-synthesis LUT, FF, BRAM, DSP usage
- post-route Fmax

Defer detailed power optimization until the architecture and compiler are stable.

Practical workflow:

1. Estimate cycles in `penguin-model/`.
2. Compare against RTL for a few representative programs.
3. Run Vivado on one board target.
4. Save results under `tests/regressions/`.
5. Reject changes that break correctness or exceed simple area/timing budgets.

## SlackBot Tool

Use the external `SlackBot` package when a task needs to send a short status update,
checkpoint notification, or other simple message to Slack from the command line.

Install it into this repo's `uv` environment from GitHub:

- `uv add git+https://github.com/penguin-tpu/SlackBot.git`
- `uv sync`

Primary commands:

- `uv run slackbot --help`
- `uv run slackbot --setup`
- `uv run slackbot --channel C01234567 --message "Penguin-TPU run finished"`
- `uv run slackbot --channel C01234567 --message "Regression passed" --thread-ts 12345.6789`
- `uv run slackbot --channel C01234567 --dry-run`

Setup notes:

- On first real use, run `uv run slackbot --setup`.
- The tool will prompt for the Slack `Bot User OAuth Token`, which usually starts with `xoxb-`.
- Do not use `App ID`, `Client ID`, `Client Secret`, `Signing Secret`, or `Verification Token` for message posting.
- The Slack app needs the `chat:write` bot scope.
- If posting to public channels without inviting the bot first, it may also need `chat:write.public`.

Channel notes:

- Prefer the Slack channel ID, such as `C01234567`, over a channel name.
- A simple way to find it is from the Slack web URL: `https://app.slack.com/client/TXXXXXXX/CXXXXXXX`.
- The `CXXXXXXX` segment is the channel ID.

Recommended use in this repo:

- Use SlackBot for long-running regressions, FPGA runs, or checkpoint notifications.
- Keep messages concise and operational: what ran, whether it passed, and where logs or artifacts live.
- Do not hardcode secrets in source files, docs, or committed config.

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
- `sed -n '1,320p' AGENTS.md`
- `sed -n '1,320p' SOUL.md`

Expected future commands once implementation exists:

- `uv run pytest`
- `uv run pytest penguin-compiler/tests`
- `uv run pytest penguin-model/tests`
- `uv run pytest tests/unit`
- `uv run pytest tests/cocotb`

## Important Current Truths

- The repository structure and package boundaries are now implemented enough to support
  a working software-only vertical slice.
- `penguin-compile` currently packages assembly plus sidecar metadata into executable
  bundles; it is not yet a direct PyTorch model exporter.
- `penguin-model` can execute mapped `.S` programs and executable bundle directories from
  the CLI.
- There is not yet a finalized ISA spec or manifest schema in `docs/specs/`.
- There is not yet a full direct-model compiler export path or RTL test flow.
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
3. Create a git commit if the change is cohesive and worth future reference. Create the commit with your name as author.

If you need project status, unresolved caveats, or open design questions, read `SOUL.md`
next.
