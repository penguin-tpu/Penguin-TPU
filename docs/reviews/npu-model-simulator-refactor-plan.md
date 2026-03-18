# NPU Model Simulator Refactor Plan

## Purpose

This document summarizes the relevant structure of the upstream
`ucb-ee194-tapeout/npu_model` repository and lays out a plan for refactoring
`penguin-model` so the simulator organization more closely matches that
reference. The goal is structural convergence on the upstream execution model,
not a literal copy of its ISA, memory map, or datatype choices.

Reference files reviewed:

- `npu_model/simulation.py`
- `npu_model/hardware/core.py`
- `npu_model/hardware/ifu.py`
- `npu_model/hardware/idu.py`
- `npu_model/hardware/stage_data.py`
- `npu_model/hardware/exu.py`
- `npu_model/hardware/mxu.py`
- `npu_model/hardware/dma.py`
- `npu_model/hardware/vpu.py`
- `npu_model/hardware/arch_state.py`
- `npu_model/software/program.py`
- `npu_model/software/instruction.py`
- `npu_model/logging/logger.py`

## Upstream Repo Organization

The upstream repo is organized around a small set of execution-driven modeling
concepts:

- `Simulation` is a thin orchestration wrapper.
- `Core` owns architectural state plus pipeline submodules.
- `IFU` and `IDU` are explicit modules.
- Each execution unit is an object with its own `tick()` and local state.
- `StageData` is the fundamental handshake primitive between stages.
- `Program` is a simple indexed instruction container.
- `Instruction` is static program data.
- `Uop` is the dynamic in-flight instruction instance.
- `Logger` is a lightweight stage-timeline recorder.

This means the upstream execution model is decomposed by hardware role rather
than by one monolithic simulator object.

## Upstream Execution Model

### Simulation Layer

`simulation.py` is intentionally small:

- instantiate logger
- instantiate core
- load program
- run `core.tick()` until completion or max cycles
- flush pending completions
- close trace
- summarize stats

The important point is that the simulation wrapper is not where timing or
execution policy lives. It only drives the hardware model.

### Core Composition

`hardware/core.py` shows the key top-level pattern:

- create `ArchState`
- create a list of execution units from config
- create `IFU`
- create `IDU`
- on each tick, run in reverse pipeline order:
  - EXUs
  - IDU
  - IFU

That downstream-first order is the core behavioral contract. It allows a
completion in an execution unit to become visible before decode makes its
same-cycle decisions, and it lets downstream claim state determine whether
upstream stages advance or stall.

### StageData Handshake

`hardware/stage_data.py` is the most important structural primitive.

Its contract is:

- upstream calls `prepare(...)`
- downstream calls `claim()`
- upstream checks `should_stall()`
- `peek()` lets a stage inspect data without consuming it

This is a one-entry elastic buffer. It is the reason the upstream IFU/IDU code
is simple and hardware-like. The stage object owns the valid bit, not the core.

### IFU Behavior

`hardware/ifu.py` models an IFU with one output register:

- if the output was not claimed, the IFU stalls
- when the IFU first observes that stall, it ends the fetch stage for the
  waiting instruction immediately
- if not stalled, it fetches one instruction and starts the fetch stage
- PC advances only when a new instruction is fetched

Important consequences:

- fetch is one cycle when the next stage claims it promptly
- if a fetched instruction is buffered but decode is blocked, fetch does not
  stay visually active forever
- the stall appears as a gap between fetch and decode, not as an overlong fetch
  bar

### IDU Behavior

`hardware/idu.py` makes decode explicit:

- it can hold one current `uop`
- it claims from IFU only when it does not already hold one
- it adds decode-time dispatch delay if the instruction requires it
- it routes output through one `StageData` buffer per execution unit
- backpressure is modeled by checking whether the target EXU output buffer is
  already occupied
- barrier-like ops are handled in decode rather than in execute

Structurally, this is closer to RTL than a direct "decode and execute in one
function" model.

### Execution Unit Pattern

`hardware/exu.py`, `hardware/mxu.py`, `hardware/dma.py`, and `hardware/vpu.py`
share the same broad pattern:

- each EXU owns its own `tick()`
- each EXU claims from one `StageData` input coming from IDU
- each EXU maintains explicit in-flight state
- completion logging is deferred to the following tick
- stage transition logging happens locally inside the EXU

The scalar EXU is the simplest case:

- retire pending completion from the previous cycle
- claim one new uop
- start execute immediately
- execute semantics immediately
- defer execute-end / retire logging to the next cycle

The MXU/VPU pattern is the long-latency version:

- keep one in-flight instruction
- count down `execute_delay`
- execute semantics when delay expires
- claim IDU input only when the EXU actually accepts the instruction

The DMA unit uses a queue instead of a single in-flight op, which is the main
exception. The structural point still holds: the functional unit owns its own
buffering and countdown.

### ArchState Role

`hardware/arch_state.py` is a centralized storage object:

- scalar register file
- matrix register file
- weight-back buffers
- main memory
- barrier / DMA flags
- PC / NPC

It is a direct shared backing store for all stages and execution units. The
upstream model keeps less separation between architectural state and micro-state
than Penguin currently does.

### Program / Dynamic Instruction Split

The upstream `Program` and `Instruction` objects are deliberately simple:

- `Program` is an indexable instruction list plus memory initialization regions
- `Instruction` is a mnemonic plus raw argument dictionary plus optional delay
- `Uop` is the dynamic per-instance instruction object with unique ID and live
  countdown fields

That split is useful because it cleanly separates:

- static program representation
- dynamic pipeline instance state

## What Matters Most To Reuse

The most valuable ideas to adopt from upstream are structural, not algorithmic:

1. A dedicated `Simulation` wrapper separate from `Core`.
2. A decomposed `Core` that owns `IFU`, `IDU`, and EXU modules.
3. Explicit `StageData` buffers for inter-stage claim-based handshaking.
4. Local per-unit `tick()` methods rather than a single monolithic issue engine.
5. A distinct dynamic instruction object for in-flight state.
6. Functional-unit-local completion scheduling and logging.

The less valuable parts to copy literally are:

- the exact upstream ISA API
- the single-memory backing model
- the simplified DMA flag scheme
- upstream rough edges and TODO-level shortcuts

## Current Penguin Differences

Today `penguin-model` already matches part of the upstream intent, but the
structure is still much more monolithic.

Current Penguin state:

- `Sim` lives inside [core.py](/home/tk/Desktop/Penguin-TPU/penguin-model/penguin_model/core.py)
- frontend, issue, scoreboarding, async completion scheduling, stop conditions,
  and trace timing all live in that one file
- architectural state and memory are richer and more realistic than upstream
- the instruction representation is typed and ISA-specific, not raw-dictionary
  based
- tensor, VPU, XLU, and DMA timing are already parameterized through
  [core_config.py](/home/tk/Desktop/Penguin-TPU/penguin-model/penguin_model/core_config.py)

This means Penguin is already ahead of upstream in some functionality, but it is
behind upstream in hardware-like modular decomposition.

## Refactor Target

The target refactor should make Penguin look structurally like:

```text
simulation.py
  Simulation

hardware/
  core.py
  ifu.py
  idu.py
  stage_data.py
  exu.py
  salu.py
  dma.py
  tmem.py
  mxu.py
  vpu.py
  xlu.py
```

That does not require matching the upstream package names exactly, but it should
match the same architectural separation of responsibilities.

## Proposed Penguin Refactor Plan

### Phase 1: Separate Orchestration From Hardware

Add a thin `simulation.py` wrapper that owns:

- `PenguinCoreConfig`
- `TraceLogger`
- `SimCore` or `Core`
- program loading
- run loop
- stats summary helpers

Keep `Sim` as the user-facing entry point only if it becomes this wrapper. The
current "core plus simulation loop" hybrid should be split.

### Phase 2: Introduce StageData

Add a local `stage_data.py` with the same conceptual contract as upstream:

- `prepare`
- `claim`
- `peek`
- `should_stall`
- `is_valid`
- `reset`

Use it for:

- IFU -> IDU
- IDU -> SALU
- IDU -> DMA
- IDU -> TMEM
- IDU -> MXU0
- IDU -> MXU1
- IDU -> VPU
- IDU -> XLU

This is the key step that removes the ad hoc `_if_slot` / `_id_slot` ownership
 from `core.py`.

### Phase 3: Split IFU

Move fetch logic into an `ifu.py` module that owns:

- fetch PC / next PC consumption
- one buffered output `StageData`
- fetch trace start/end rules
- frontend stall handling

The current corrected fetch behavior should be preserved exactly:

- one-cycle fetch when promptly claimed
- fetch closes on first unclaimed cycle
- no new fetch when output remains buffered

### Phase 4: Split IDU

Move decode / issue into `idu.py`:

- one currently-held dynamic instruction
- per-EXU output stage buffers
- operand-ready and scoreboard checks
- decode-only ops such as `dma.wait`
- barrier / fence semantics
- dispatch-delay handling if Penguin later needs it

This is where most of the current `core.py` branching should move.

### Phase 5: Introduce a Penguin Dynamic Uop

Keep the current typed static `Instruction`, but add a dynamic `Uop`-like object
containing:

- unique instruction ID
- static `Instruction`
- fetch / decode / execute bookkeeping
- per-uop countdown fields
- any per-uop branch-shadow or completion metadata

This lets the model preserve the current typed ISA while adopting the upstream
"static instruction vs dynamic in-flight uop" split.

### Phase 6: Split Execution Units

Create an EXU base class and unit-specific subclasses.

Recommended split:

- `ScalarExecutionUnit`
- `DmaExecutionUnit`
- `TensorMemoryExecutionUnit`
- `MatrixExecutionUnit`
- `VectorExecutionUnit`
- `TransposeExecutionUnit`

The important upstream pattern to copy is:

- each unit claims from one IDU output
- each unit owns its own in-flight queue / slot
- each unit is responsible for `D -> E -> retire` logging
- each unit decides when a claimed instruction becomes architecturally visible

### Phase 7: Move Async Completion Scheduling Into Units

The current `_schedule_action(...)` model in `core.py` should shrink or disappear.

Instead:

- scalar EXU uses a one-cycle deferred completion slot
- MXU/VPU/XLU keep explicit in-flight entries with countdown
- DMA owns its own queue and transfer lifecycle
- any VMEM-side bandwidth delays sit in the TMEM / DMA units rather than in the
  top-level core

This will make unit-local timing easier to reason about and closer to the
upstream design.

### Phase 8: Keep Penguin-Specific Enhancements

Do not regress these current Penguin features while refactoring:

- `PenguinCoreConfig` and its hierarchical parameterization
- separate DRAM / VMEM / IMEM modeling
- typed instruction parameter objects
- richer stop reasons
- current JSON trace lane set
- current DMA multi-channel behavior
- current operand scoreboards across `x`, `e`, `m`, weight slots, and VMEM

These are improvements over upstream and should be preserved behind the new
module boundaries.

## Concrete File Moves

Likely moves out of the current
[core.py](/home/tk/Desktop/Penguin-TPU/penguin-model/penguin_model/core.py):

- `_PipelineSlot` -> dynamic `uop.py` or `instruction.py`
- fetch trace handling -> `ifu.py`
- decode / operand-ready / destination reservation logic -> `idu.py`
- DMA issue/transfer stage handling -> `dma.py`
- MXU/VPU/XLU completion callbacks -> unit-local modules
- top-level run loop / trace setup / auto-trace plumbing -> `simulation.py`

The remaining top-level `Core` should mainly:

- instantiate submodules
- wire buffers
- call `tick()` downstream-first
- expose `load_program`, `reset`, `is_finished`, and stop/drain helpers

## Risks And Watchpoints

### Scoreboard Placement

Penguin currently uses centralized scoreboards inside `core.py`. If we move to a
more upstream-like structure, we need to decide whether scoreboards live:

- in `IDU`
- in `Core`
- or in a dedicated pipeline-control helper

The upstream repo is simple enough that this problem barely exists; Penguin is
not.

### DMA Wait Contract

`dma.wait` in Penguin already has precise decode-fence semantics. The refactor
must preserve:

- decode-resident waiting
- no execute-stage entry
- per-channel independence
- one buffered IFU instruction while decode is fenced

### Trace Compatibility

The current Penguin trace is richer than upstream:

- more lanes
- memory-region events
- free-running cycle counter

The refactor should preserve current trace outputs even if the internal module
structure becomes more upstream-like.

### Stop / Step-Limit Semantics

The current model has explicit `StopReason` handling and `max_instructions`
support. Those need to be re-specified carefully once instruction progress lives
in multiple modules.

## Recommended Implementation Order

1. Add `Simulation` wrapper without changing behavior.
2. Add `StageData` and migrate IFU/IDU boundaries first.
3. Split IFU and IDU out of `core.py`.
4. Introduce dynamic `Uop`.
5. Split SALU and DMA units.
6. Split TMEM / MXU / VPU / XLU units.
7. Remove obsolete top-level scheduling helpers from `core.py`.
8. Re-baseline traces and cycle tests after each phase.

## Success Criteria

The refactor should be considered successful when:

- `core.py` becomes mostly wiring and tick ordering
- IFU, IDU, and each major EXU are separate modules
- inter-stage movement is mediated by `StageData`
- dynamic in-flight instruction state is explicit
- current Penguin regressions stay green
- the resulting organization is recognizably closer to upstream `npu_model`
  while still preserving Penguin-specific architectural behavior
