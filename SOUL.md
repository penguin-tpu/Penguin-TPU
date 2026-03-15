# Penguin-TPU Soul

## Current State

The repository is no longer just a scaffold. The design direction is now stable enough
to treat the current spec set as the baseline for implementation.

What exists now:

- a `uv` workspace with two Python packages:
  - `penguin-compiler`
  - `penguin-model`
- a working scalar integer functional model in `penguin-model`
- scalar tests covering the current RV32I-derived scalar subset
- a directed scalar-program testbench for the functional/perf model, including
  label-resolved self-checking programs inspired by `riscv-tests` `rv32ui`
- a GitHub Actions CI workflow that installs the `uv` workspace and runs
  the full Python test suite on pushes and pull requests
- formal baseline specs:
  - [scalar-functional-subset.md](/home/tk/Desktop/Penguin-TPU/docs/specs/scalar-functional-subset.md)
  - [architecture-spec.md](/home/tk/Desktop/Penguin-TPU/docs/specs/architecture-spec.md)
  - [microarchitecture-spec.md](/home/tk/Desktop/Penguin-TPU/docs/specs/microarchitecture-spec.md)
  - [memory-organization-spec.md](/home/tk/Desktop/Penguin-TPU/docs/specs/memory-organization-spec.md)
  - [configuration-parameters-spec.md](/home/tk/Desktop/Penguin-TPU/docs/specs/configuration-parameters-spec.md)
  - [upstream-npu-spec-merge-review.md](/home/tk/Desktop/Penguin-TPU/docs/reviews/upstream-npu-spec-merge-review.md)

What does not exist yet:

- actual `penguin-compiler` export logic
- executable package loader/writer implementation
- tensor ISA implementation in the model
- RTL testbench flow
- FPGA bring-up flow

## Frozen Baseline

These are the design choices that should now be treated as intentional baseline
decisions, not open brainstorming.

### Execution model

- single-issue frontend
- fixed-width 32-bit instructions
- one instruction launch per cycle
- long-chime tensor instructions
- statically scheduled machine
- on-chip execution is deterministic
- off-chip memory activity is asynchronous
- branches and jumps have 2 architecturally visible delay slots
- architectural state explicitly includes host-visible execution control/status, DMA busy
  state, and the shared memory-base CSR
- structural conflicts are resolved by stalls/arbitration, not by partial architectural
  completion
- Penguin does not support general trap or interrupt recovery; errors halt and report
  status

Reasoning:

- keeps the machine explainable
- minimizes frontend complexity
- makes compiler scheduling meaningful
- keeps async behavior limited to off-chip traffic

### Tensor architecture

- 64 tensor registers: `m0..m63`
- each register is `64 rows x 32 bytes`
- one flat tensor register file shared across data types
- whole-register tensor operations only
- full-connectivity tensor crossbar between registers and functional units
- one architecturally visible XLU for transpose work
- XLU scope is now frozen to transpose only

Reasoning:

- simple software-visible storage model
- no type-specific tensor storage classes
- no sub-tile window semantics in the first ISA cut
- no architectural bank partitioning rules
- dedicated transpose hardware avoids forcing layout-reordering work into VMEM or VPU
- broader TPU-style cross-lane functionality is intentionally deferred

### MXU architecture

- two MXUs:
  - `mxu0`: systolic-array-based
  - `mxu1`: inner-product-tree-based
- both are architecturally visible
- both can execute concurrently, but only one new instruction may issue per cycle
- weight-stationary dataflow
- each MXU has distinct `w0` and `w1` weight-slot state
- MXU does pure matmul/partial-sum accumulation only
- no bias/residual/activation fusion in the MXU

Reasoning:

- comparison between `mxu0` and `mxu1` is a design goal
- weight residency is explicit and software-visible
- matrix hardware stays focused on matrix work
- fusion is deferred to VPU/software scheduling

### MXU numerical contract

- `FP8_e4m3 x FP8_e4m3 -> BF16`
- output-only scaling per workload-level matmul
- optional BF16-to-FP8 writeback
- BF16-to-FP8 uses round-to-nearest-even with saturation on overflow

Reasoning:

- low-precision multiplicands keep matrix hardware efficient
- BF16 accumulation is a practical first target
- output-only scaling is simpler than per-input scaling

### VPU contract

- VPU reads directly from `m` registers
- VPU writes only to `m` registers
- no local operand buffers
- whole-register operations only
- first opcode floor:
  - `vadd`
  - `vmul`
  - `vmax`
  - `vmin`
  - `vrelu`
  - `vmov`

Reasoning:

- keeps compute separate from memory movement
- gives a small but useful first VPU floor
- avoids hidden data movement semantics

### XLU contract

- one architecturally visible XLU
- XLU reads directly from `m` registers
- XLU writes only to `m` registers
- whole-register operations only
- intended for transpose work

Reasoning:

- transpose is important enough to be a first-class accelerator operation
- keeps tensor reordering out of MXU and out of DMA
- avoids awkward software-only transpose sequences for common layouts

### Memory organization

- DRAM: backing data storage
- DRAM base is `0x8000_0000`
- DRAM size is `16 GiB`
- IMEM: instruction memory
- IMEM base is `0x0010_0000`
- IMEM size is `32 KiB`
- VMEM: on-chip tensor/vector data memory
- VMEM base is `0x0800_0000`
- VMEM size is `1 MiB`
- IMEM and VMEM are byte-addressed
- IMEM fetch is 4-byte aligned
- tensor registers access VMEM only
- DMA is the only DRAM <-> VMEM path
- DMA moves unit-stride raw bytes only
- DMA addresses are 32-byte aligned and DMA sizes are multiples of 32 bytes
- DMA is asynchronous and fenced by channel
- DRAM latency is currently modeled as 10 cycles in the functional model
- first revision exposes 8 symmetric DMA channels
- `vload` / `vstore` are blocking VMEM <-> `m` transfers
- `mxu.push.*` is a blocking VMEM -> `w*` transfer
- `vload` / `vstore` transfer one full 2048-byte tensor register image
- `mxu.push.*` transfers one full 512-byte weight tile
- `vload` / `vstore` / `mxu.push.*` all require 32-byte-aligned VMEM addresses
- DMA, `vload` / `vstore`, and `mxu.push.*` all use scalar-register indirect addressing
- one shared memory-base CSR extends addressing beyond the 32-bit scalar range
- IMEM is populated before execution by host-side software or firmware
- `dma.wait.chN` returns immediately if the channel is already idle
- DMA completion order across channels is not guaranteed by issue order
- each DMA channel supports only one outstanding operation
- software provides tiled tensor layout and tile-level zero padding

Reasoning:

- one narrow async boundary is easier to implement and validate
- DMA stays simple because it is byte-oriented, not tensor-aware
- whole-register on-chip transfers match the whole-tile tensor model
- scalar-directed addressing preserves the one-word instruction rule

## Design Intent

The project is deliberately ordered this way:

1. scalar core first
2. memory structure next
3. testing/regression infrastructure next
4. tensor accelerator features after the scalar-plus-memory base is solid

This ordering still stands. The tensor specs are now far enough along that they can guide
implementation, but they should not cause the project to skip the scalar and memory
bring-up steps.

## Implementation Reality

The codebase still lags the tensor specs significantly.

Implemented today:

- scalar functional model
- scalar tests
- reusable scalar directed-program builder and runner for model tests
- directed scalar ALU, branch/jump, and load/store program tests with perf checks
- GitHub CI for automatic `uv run pytest` coverage
- trace logging
- separate DRAM / VMEM / IMEM memory regions in the functional model
- fixed architectural base addresses for IMEM, VMEM, and DRAM in the functional model
- `ArchState` now owns `dram`, `vmem`, `imem`, and DMA channels directly rather than
  nesting them under `state.memories`
- the old `MemorySystem` wrapper has been removed; memory-region construction now goes
  through `ArchState.with_memory_sizes(...)`
- scalar control flow now implements the spec-defined 2 branch/jump delay slots
- a younger control-transfer instruction in a delay slot now overrides any older pending
  redirect in both the spec and the model
- VMEM-only scalar `sld` / `sst`
- DMA channel issue/wait behavior for DRAM <-> VMEM byte transfers
- DMA completion timing modeled with 10-cycle DRAM latency
- trace and execution modeling now separate `EXU.SALU` from `EXU.DMA`
- DRAM backing in the functional model is page-backed so the 16 GiB region is modeled
  sparsely rather than allocated densely
- scalar verification is now grouped across ISA/unit tests, directed scalar programs,
  workload tests, and locked performance regression tests
- the scalar test surface now covers control-flow delay slots, shift-mask behavior,
  DMA edge cases, VMEM-only data paths, workload-style address generation/copy/reduction,
  and performance counters for representative scalar kernels
- a text assembly parser now loads checked-in `.S` programs into the Python model
- scalar tests and the runnable scalar example now execute checked-in assembly sources
  from `tests/vectors/programs/` instead of constructing instruction lists in Python
- `scripts/generate_scalar_programs.py` now regenerates the current scalar test/example
  assembly corpus deterministically
- the formal tensor-side specs were tightened after a direct review of the upstream
  `ucb-ee194-tapeout/npu_model/npu_spec` documents
- that merge pass added explicit execution-state inventory, host launch/reset semantics,
  DMA visibility/order rules, and a deliberate no-partial-retirement rule
- the Saturn Microarchitecture Manual was noted as an additional future reference,
  especially for frontend/fault/memory-ordering questions that Penguin has not fully
  frozen yet
- user review then resolved several previously open items: host CSR block semantics,
  host-only early `MEM_BASE` programming, single-outstanding DMA channels, 32-byte
  VMEM tensor alignment, transpose-only XLU scope, tile-level zero padding, and the
  no-recovery error model

Not yet implemented:

- tensor instructions
- MXU/VPU/XLU execution
- executable package
- compiler lowering

## Immediate Next Steps

1. Define the executable package and manifest.
2. Add formal tensor layout/packing spec, especially padding and tail-handling rules.
3. Add a first binary/text assembly encoding spec for 32-bit instructions.
4. Define the host control and CSR map for launch, halt, done, and error reporting.
5. Implement tensor transfer instructions: `vload`, `vstore`, `mxu.push.*`.
6. Implement the first tensor-side functional stubs for `matmul.*`, XLU transpose, and
   the initial VPU op floor.

## Remaining Open Items

Only a small set of important questions remain:

- exact DMA instruction shapes
- exact `vload` / `vstore` encodings
- exact host-side CSR-region base address and field layout
- exact DMA instruction shapes for tensor-era programs
- exact XLU transpose instruction encoding and sequencing
- whether VMEM is logically unified or internally partitioned by traffic class
- final tensor ISA encoding details
- halt behavior while long-chime tensor operations are in flight

## Checkpoint Note

The current document set should be treated as the first coherent baseline for the tensor
architecture and memory organization. Future changes should update the formal specs and
then reflect the delta here, rather than letting `SOUL.md` become a second competing
specification.
