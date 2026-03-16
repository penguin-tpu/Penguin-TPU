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
- initial VPU elementwise functional/performance modeling for `vadd`, `vmul`, `vmax`,
  `vmin`, `vrelu`, and `vmov`
- executable-package manifest/symbol-table support on both sides of the software
  boundary:
  - `penguin-compiler` can now write bundle directories with `program.S`,
    a sidecar `program.symbols.json5`, `manifest.json5`, `constants.bin`, and
    file-backed input/output payloads
  - `penguin-model` can now load that manifest, resolve symbol memory mappings, and
    execute programs from a nonzero IMEM base address
  - `penguin-model` can also stage file-backed bundle payloads into mapped DRAM/VMEM/IMEM
    regions before execution
- working package CLIs now exist on both sides of the software boundary:
  - `penguin-compile bundle ...` packages assembly plus symbol/payload metadata into a
    runnable bundle
  - `penguin-model --program ...` executes mapped `.S` programs
  - `penguin-model --bundle ...` executes bundle directories and can emit JSON traces
- checked-in programs under `tests/vectors/programs/` now also carry adjacent
  `*.symbols.json5` sidecars, and the shared Python loaders consume those sidecars when
  running scalar/tensor example programs from source files
- deterministic pseudo-random power-on initialization for DRAM, VMEM, scalar registers,
  tensor registers, and MXU weight-slot state in the Python model
- repo-wide verification now also locks:
  - all checked-in assembly programs under `tests/vectors/programs/` have valid
    `*.symbols.json5` sidecars whose recorded sizes match the assembled instruction
    streams
  - bundle-loader error handling for non-program entry symbols and program-size mismatch
  - assembler parsing of symbol/label expressions inside tensor-memory operands
  - VPU in-place destination aliasing semantics against BF16/PyTorch reference behavior
- user-facing READMEs now describe the real current software surfaces rather than the
  original scaffold-only state:
  - current example entry points
  - executable bundle layout
  - sidecar symbol-table loading flow
  - the current CLI surfaces and their remaining deferred scope
- formal baseline specs:
  - [scalar-functional-subset.md](/home/tk/Desktop/Penguin-TPU/docs/specs/scalar-functional-subset.md)
  - [architecture-spec.md](/home/tk/Desktop/Penguin-TPU/docs/specs/architecture-spec.md)
  - [microarchitecture-spec.md](/home/tk/Desktop/Penguin-TPU/docs/specs/microarchitecture-spec.md)
  - [memory-organization-spec.md](/home/tk/Desktop/Penguin-TPU/docs/specs/memory-organization-spec.md)
  - [configuration-parameters-spec.md](/home/tk/Desktop/Penguin-TPU/docs/specs/configuration-parameters-spec.md)
  - [upstream-npu-spec-merge-review.md](/home/tk/Desktop/Penguin-TPU/docs/reviews/upstream-npu-spec-merge-review.md)

What does not exist yet:

- direct PyTorch-to-Penguin export logic
- manifest-level runtime mapping for `constants.bin`
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
- initial floating-point elementwise view is BF16 over the `64 x 16` tensor-register
  interpretation
- first opcode floor:
  - `vadd`
  - `vmul`
  - `vmax`
  - `vmin`
  - `vrelu`
  - `vmov`
- pipelineable elementwise operations use a 2-cycle latency class
- future non-pipelineable operations such as division use an 8-cycle latency class

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

Current regression status after the verification/documentation sweep:

- `uv run pytest` passes with 235 tests

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
- first tensor-side model slice now exists in `penguin-model`:
  - architectural tensor register storage for `m0..m63`
  - architectural MXU weight-slot storage for `mxu0/1.w0/1`
  - blocking `vload` / `vstore` VMEM transfers
  - blocking `mxu.push.mxu0/1` VMEM -> weight-slot transfers
  - `matmul.mxu0/1` and `matmul.add.mxu0/1` functional semantics
  - assembly parsing support for `m*` and `w*` operands
  - FP8 activation/weight interpretation and BF16 result storage
- tensor-side verification is now active in the main pytest suite:
  - MXU tests no longer sit behind a stale registry gate
  - parser coverage now includes tensor-memory and MXU operand forms
  - tensor-memory tests cover `vload` / `vstore` round-trip behavior and alignment faults
  - MXU tests cover weight-slot isolation, cross-MXU isolation, explicit partial sums,
    VMEM sourcing, and perf-counter / latency accounting for tensor ops
- the public `INSTRUCTION_LATENCY` view now includes tensor instructions in addition to
  the scalar and DMA subset
- runnable tensor examples now include both single-tile demos and larger DMA-backed
  stripmined workloads backed by checked-in `.S` programs under
  `tests/vectors/programs/tensor/examples/`
- the functional/perf model now carries explicit bandwidth-based transfer timing knobs
  and systematic verification for:
  - delay-slot execution order and target-start timing
  - younger control-transfer replacement inside delay slots
  - DMA wait ordering in traces
  - intermediate tensor-register and DRAM tile state during larger workloads
- current Python verification status at this checkpoint:
  - `uv run pytest` passes with 144 tests
- runnable tensor examples now exist for:
  - single-tile `matmul`
  - tiled `linear` with bias over a 128x32 input batch and 32 output features
  - DMA-backed tiled `linear` with bias over a 192x64 input batch and 48 output features
- the tensor examples load checked-in assembly from `tests/vectors/programs/tensor/examples/`,
  emit Perfetto-compatible JSON traces, and verify simulator output against a PyTorch
  BF16-accumulation golden reference
- trace and performance-model fixes for correct observability:
  - **PC in trace**: the logged PC records the value at the instruction fetch stage so that
    the instruction in the IFU matches the logged PC (fetch address). PC is emitted when
    an instruction enters the IFU, not at retire.
  - **DMA transfer bar**: the DMA transfer interval on the trace is now logged entirely in
    trace (pipeline) time. The transfer start is logged when the load/store issues; the
    transfer end is logged when the matching `dma.wait.chN` completes. Previously the
    transfer end used cycle-based time, which made it look like dependent instructions
    (e.g. `vload`) could start before the fence resolved.
  - **Delay slots**: control-flow redirects still retire two delay-slot instructions before
    the next fetch uses the target; only the PC log semantics were adjusted to reflect
    fetch-stage PC.

Not yet implemented:

- VPU execution
- XLU execution
- executable package
- compiler lowering

## Tensor Modeling Notes

The tensor-side implementation intentionally made a few conservative choices where the
spec is not fully frozen yet.

- The current assembly parser accepts:
  - `vload mD, imm(xN)`
  - `vstore mS, imm(xN)`
  - `mxu.push.mxu0 w0, imm(xN)` and analogous forms
  - `matmul.mxu0 mD, mS, w0`
  - `matmul.add.mxu0 mD, mS, w0, mP`
- This memory-operand syntax is a model choice, not a fully frozen ISA encoding.
- The model uses `torch.float8_e4m3fn` as the practical stand-in for the spec’s
  `FP8_e4m3` contract. Confirm later whether this is acceptable or whether a stricter
  custom FP8 implementation is required.
- The model computes MXU results into BF16 tensor-register images and does not yet
  implement optional BF16-to-FP8 writeback, because the writeback-mode instruction forms
  are not frozen.
- The model does not yet implement workload-level scalar output scaling for matmul,
  because that control path is specified at the workload level but not yet frozen as an
  instruction or CSR interface.
- Tensor instruction latency is currently modeled with deterministic placeholders:
  - `vload`: 64 cycles
  - `vstore`: 64 cycles
  - `mxu.push.*`: 32 cycles
  - `matmul.*`: 64 cycles
- Tensor operations are currently modeled as blocking and atomic at instruction
  retirement. This preserves deterministic correctness, but it does not yet model the
  full spec intent that long-chime MXU work may overlap with younger scalar issue when
  hazards permit.

## DMA Modeling Notes

The DRAM <-> VMEM DMA path already existed in the scalar model. The current refactor
keeps that unit-stride functionality and tightens the architectural-state contract
around it.

- The model now carries an explicit host-programmable `mem_base` field in `ArchState`.
- For the current baseline, the model interprets `mem_base` as the shared high-address
  extension for memory-like instructions:
  - effective address = `(mem_base << 32) | low32`
  - `low32` comes from the scalar-register-indirect address calculation
- This choice preserves the current absolute 32-bit-address programs while still making
  it possible to model memory above 4 GiB later.
- DMA issue and wait remain channel-specific and unit-stride only.
- DMA still snapshots the source byte payload at issue time and makes the destination
  bytes architecturally visible only when the matching `dma.wait.chN` completes.
- DMA completion timing is no longer a fixed placeholder:
  - off-chip DRAM link timing now models a 32-bit serialized interface at half core
    frequency
  - the off-chip timing includes 2 serialized overhead words per DMA operation
  - VMEM-side timing now models a 128-bit on-chip system bus
  - DMA completion uses the slower of the off-chip and VMEM transfer paths
- `vload` / `vstore` and `mxu.push.*` now derive their default latency from the VMEM
  bus width instead of using row-count placeholders.
- the Python model now has one hierarchical `PenguinCoreConfig` entry point that owns:
  - memory-map constants
  - DMA alignment/channel-count parameters
  - tensor register and weight-slot geometry
  - VPU timing parameters
  - off-chip and VMEM bandwidth parameters
  - trace-timing and scalar delay-slot parameters
- module-level constants such as `DRAM_BASE`, `MREG_BYTES`, and `DMA_CHANNEL_COUNT`
  remain as aliases of the default core configuration for compatibility, but the active
  runtime behavior now flows through `ArchState.config`
- `PenguinCore`, `ArchState`, the example workloads, and the shared scalar testbench
  helper now instantiate through `PenguinCoreConfig`
- the VPU model now executes BF16 whole-register elementwise operations directly from
  and to the tensor register file
- initial VPU latency is parameterized through `config.vpu`
- architecturally unspecified DRAM/VMEM/register contents now default to deterministic
  pseudo-random data in the Python model via `config.initialization`
- checked-in startup programs that depend on clean scalar state now include an explicit
  scalar-register scrub prologue

Open question to confirm later:

- whether the intended `MEM_BASE` behavior is exactly a high-bits extension
  `(mem_base << 32) | low32`, or some other shared offset encoding
- whether the architected first VPU data view should remain BF16-only, or whether FP8
  elementwise views are also intended in the initial ISA

## Immediate Next Steps

1. Start using the executable-package manifest and symbol table from actual compiler
   export paths instead of example-only/test-only flows.
2. Add formal tensor layout/packing spec, especially padding and tail-handling rules.
3. Add a first binary/text assembly encoding spec for 32-bit instructions.
4. Define the host control and CSR map for launch, halt, done, and error reporting.
5. Refine tensor execution timing and overlap semantics for long-chime MXU/VPU work.
6. Implement the next tensor-side functional unit: XLU transpose.

## Remaining Open Items

Only a small set of important questions remain:

- exact DMA instruction shapes
- exact `vload` / `vstore` encodings
- exact host-side CSR-region base address and field layout
- exact DMA instruction shapes for tensor-era programs
- exact `MEM_BASE` address-extension encoding
- exact XLU transpose instruction encoding and sequencing
- whether VMEM is logically unified or internally partitioned by traffic class
- final tensor ISA encoding details
- halt behavior while long-chime tensor operations are in flight

## Checkpoint Note

The current document set should be treated as the first coherent baseline for the tensor
architecture and memory organization. Future changes should update the formal specs and
then reflect the delta here, rather than letting `SOUL.md` become a second competing
specification.

## RTL Bring-Up Start

- `rtl/penguin_tpu/` now includes a minimal FPGA hello-world path:
  - vendored `uart.v`, `uart_tx.v`, and `uart_rx.v` from the referenced
    `alexforencich/verilog-uart` project under the upstream MIT license
  - a new `penguin_uart_hello_top.v` top level that sends `Hello World\r\n`
    once per second over UART
  - that top level now matches the checked-in Nexys Video constraint naming:
    `sys_clk_i`, `cpu_resetn`, `uart_tx_in`, and `uart_rx_out`
  - internal RTL module interfaces now use the repository clock/reset naming
    convention: `clock` and `reset`
- the top-level UART path is intentionally simple and parameterized only by
  `CLK_FREQ_HZ` and `BAUD_RATE`; it is meant for first-board bring-up before
  any Penguin core integration
- `tests/cocotb/` now contains a first RTL regression for that bring-up path:
  - `tb_uart_hello.py` drives the top-level clock/reset, decodes the serial
    waveform, and checks both the `Hello World\r\n` payload and 1 Hz cadence
  - `test_uart_hello.py` runs the cocotb test through Verilator from pytest
- current cocotb dependency is intentionally pinned to `>=1.8,<1.9` because the
  installed Verilator version is `5.020`, which is older than the minimum
  version expected by newer cocotb Verilator integrations
- GitHub Actions CI now runs RTL regressions in a dedicated `rtl-tests` job that
  installs Verilator on `ubuntu-latest` and executes `pytest tests/cocotb`
- on March 15, 2026, the checked-in Vivado TCL flow successfully built
  `penguin_uart_hello_top.bit`, programmed the connected Nexys Video board, and
  produced `Hello World` over the enumerated USB UART device `/dev/ttyUSB0`
- added `scripts/vivado/read_uart_hello.py` plus
  `docs/flows/nexys-video-hello-world-bringup.md` to make FPGA build, program,
  and UART validation repeatable
- refined the board bring-up path with
  `scripts/vivado/run_hello_world_bringup.sh`, which cleans the Vivado project,
  runs the TCL flow in order, retries FPGA programming, and validates UART
  output through `uv run python`
- reran the full wrapper flow successfully on March 15, 2026; clean build,
  program, and UART validation all passed

Open follow-up for the next FPGA step:

- board-specific reset polarity, clock frequency, pin constraints, and Vivado
  project flow still need to be supplied before bitstream generation

## Scalar RTL Planning

- added `docs/plans/scalar-core-encoding-and-rtl-plan.md` to capture the
  proposed scalar binary encoding baseline plus a step-by-step RTL bring-up plan
- current planning direction is:
  - keep scalar binary encodings RV32I-compatible for the first RTL slice
  - keep Penguin-specific `s*` mnemonics at the assembly/spec layer
  - reserve RISC-V custom major opcodes for future DMA/tensor instruction
    families instead of spending them on scalar bring-up
- the formal spec set now reflects that same direction:
  - `architecture-spec.md` defines the scalar binary baseline and reserved
    custom-opcode space
  - `microarchitecture-spec.md` defines the scalar decode baseline and first
    scalar RTL slice structure
  - `scalar-functional-subset.md` no longer treats the scalar binary layer as
    undefined

## Scalar RTL First Implementation

- implemented the first scalar RTL subtree under `rtl/penguin_tpu/scalar/`:
  - `penguin_scalar_defs.vh`
  - `penguin_scalar_decoder.v`
  - `penguin_scalar_regfile.v`
  - `penguin_scalar_alu.v`
  - `penguin_scalar_branch_unit.v`
  - `penguin_scalar_lsu.v`
  - `penguin_scalar_controller.v`
  - `penguin_scalar_core.v`
- added `penguin_model.scalar_encoding.encode_scalar_instruction()` so the
  software side can produce RV32I-compatible scalar machine words for RTL tests
- validated the implementation incrementally:
  - Python unit tests for scalar encoding
  - cocotb regression for the standalone decoder
  - cocotb regressions for regfile, ALU, branch unit, and LSU
  - cocotb regression for the integrated scalar core covering arithmetic,
    load/store, 2-delay-slot jumps, younger control-transfer override, and
    misaligned-load halt behavior
- added a first scalar-core UART-MMIO top level:
  - `rtl/penguin_tpu/penguin_scalar_uart_hello_top.v`
  - this instantiates the scalar core, embeds a checked-in program ROM, exposes
    a tiny UART MMIO block, and prints `hello, this is penguin`
- added the corresponding checked-in assembly source:
  - `tests/vectors/programs/scalar/rtl/uart_mmio_hello.S`
- added a generated ROM init include derived from that assembly:
  - `rtl/penguin_tpu/scalar/penguin_scalar_uart_hello_program_init.vh`
- end-to-end cocotb regression now proves that the scalar core can write to the
  UART through MMIO and emit the target string
- adjusted the checked-in UART-MMIO hello program so the scalar-core FPGA target
  loops forever instead of printing once and halting; this makes post-program
  serial attachment reliable during board validation

## Multi-Target FPGA Bring-Up

- generalized the Vivado bring-up wrapper to
  `scripts/vivado/run_fpga_bringup.sh`
- the wrapper now supports:
  - `--target uart_hello`
  - `--target scalar_core`
  - `sclar_core` as a compatibility alias for `scalar_core`
- `scripts/vivado/2_add_files.tcl` now selects the synthesis top from
  `PENGUIN_VIVADO_TARGET` and includes both board targets plus the scalar-core
  RTL source set
- `scripts/vivado/4_program_device.tcl` now derives the bitstream path from the
  active project top instead of assuming a single fixed design
- `scripts/vivado/run_hello_world_bringup.sh` now acts as a compatibility
  wrapper around the new multi-target script
- reran the full board flow for `scalar_core` on March 16, 2026:
  - first hardware-programming attempt failed at `open_hw_target` with Vivado's
    intermittent "No devices detected" error
  - the wrapper retry path succeeded on the second programming attempt without
    changing the board setup
  - UART validation on `/dev/ttyUSB0` succeeded and captured repeated
    `hello, this is penguin`
  - the routed bitstream is therefore functionally usable on the board
- current scalar-core FPGA caveat:
  - Vivado 2024.2 still reports negative routed timing on
    `penguin_scalar_uart_hello_top`
  - routed timing summary from the successful board-tested build:
    - `WNS=-2.787 ns`
    - `TNS=-2266.791 ns`
  - functional board validation passed despite the timing violation, so timing
    closure remains the next hardware-quality issue rather than basic
    functionality
