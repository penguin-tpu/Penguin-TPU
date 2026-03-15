# Upstream NPU Spec Merge Review

## 1. Purpose

This document records what was learned from reviewing the upstream
`ucb-ee194-tapeout/npu_model/npu_spec` documents and which points were merged into the
Penguin spec set.

It also captures the places where Penguin intentionally diverges, where the upstream
design would force a new architectural choice, and where neither document set yet closes
an important risk.

## 2. Upstream Documents Reviewed

The review covered:

- `npu_spec/00_preface/README.md`
- `npu_spec/01_introduction/README.md`
- `npu_spec/02_system_parameters/README.md`
- `npu_spec/03_registers_and_execution_state/README.md`
- `npu_spec/04_functional_units/README.md`
- `npu_spec/05_memory_model/README.md`
- `npu_spec/06_instruction_set/README.md`

The upstream repository currently has some section-label drift, so the review was based on
actual document contents rather than folder names alone.

Related reference material also worth keeping in view:

- [The Saturn Microarchitecture Manual](https://saturn-vectors.org/)

Saturn is not a TPU-like tensor-register machine, so it is not a direct architectural
template for Penguin. It is still valuable for future work on host integration,
frontend/fault behavior, memory disambiguation, and disciplined microarchitectural
interface design.

## 3. What Was Merged Into Penguin

The upstream spec was most useful in the places where Penguin had design intent but not
yet a sharp contract.

Merged improvements:

- explicit architectural-state inventory
- explicit host launch / completion model
- explicit reset expectations
- explicit DMA busy/completion semantics
- explicit statement that DMA completion order across channels is not ordered by issue
- explicit rule that `dma.wait.chN` returns immediately if the channel is already idle
- explicit DMA alignment and transfer-granularity contract
- explicit fixed transfer sizes for `vload`, `vstore`, and `mxu.push.*`
- explicit hazard-ordering rules around DMA versus VMEM-facing operations
- explicit all-or-nothing completion rule for structural conflicts
- explicit shared configuration-parameter document

These were the main missing contracts in Penguin before this merge pass.

## 4. Deliberate Divergences From Upstream

These are not mistakes. They are current Penguin choices.

### 4.1 Scalar width

Upstream uses 64-bit scalar control registers. Penguin remains 32-bit in the baseline and
extends memory reach through one shared base-offset CSR.

Implication:

- Penguin instruction encoding stays tighter
- software address generation is simpler in the short term
- very large absolute addresses require the base-offset mechanism to be well specified

### 4.2 Tensor-register accessibility

Upstream allows DMA paths that target matrix registers or weight buffers more directly.
Penguin intentionally forces DRAM -> VMEM -> `m` / `w*`.

Implication:

- Penguin keeps the async boundary narrow and explicit
- on-chip transfers stay deterministic
- there is an extra explicit staging step in software

### 4.3 Structural conflicts

Upstream describes a bank-conflict model where younger instructions can win and older
instructions can become partially written. Penguin should not adopt that behavior.

Implication:

- Penguin stalls or arbitrates internally instead
- architectural results remain all-or-nothing
- compiler reasoning and verification remain much cleaner

### 4.4 Dedicated transpose unit

Upstream includes an XLU for transpose. Penguin now also plans a dedicated XLU in the
baseline direction, and the current design direction is to reuse the upstream XLU
structure as a starting point.

Frozen interpretation:

- XLU is baseline hardware, not a deferred option
- XLU scope is transpose only
- Penguin is not currently adopting a broader TPU-style cross-lane function set

### 4.5 Datatype breadth

Upstream keeps multiple floating-point formats in play. Penguin intentionally narrows the
first tensor contract to `FP8_e4m3` multiplicands and BF16 accumulation.

Implication:

- the numeric contract is simpler and easier to verify
- future broadening should be justified by workload evidence rather than added preemptively

## 5. Resolved User Decisions

### 5.1 Host CSR map

Resolved direction:

- the architecture now keeps a required list of host-visible CSRs
- those CSRs are laid out consecutively in memory
- the CSR-region base address remains deferred until SoC integration

### 5.2 Shared memory-base CSR access path

Resolved direction:

- host-only programming is sufficient for early bring-up
- future Penguin code may access the CSR region through MMIO-style loads/stores or direct
  hardware connections where needed

### 5.3 DMA channel reuse semantics

Resolved direction:

- each DMA channel allows only one outstanding operation
- software is responsible for not reusing a busy channel
- the functional/performance model should check this and fail explicitly on violation

### 5.4 VMEM alignment for tensor-facing operations

Resolved direction:

- `vload`, `vstore`, and `mxu.push.*` require 32-byte alignment for now
- this is still allowed to change in a later revision if implementation pressure justifies
  it

### 5.5 Exact XLU contract

Resolved direction:

- XLU only needs to handle transpose in the current Penguin baseline
- broader TPU-style cross-lane functions are intentionally out of scope for now

## 6. Risks And Pitfalls Still Not Fully Covered

These are important even though they are not all ready to freeze yet.

### 6.1 Tail handling and padding

Whole-register execution means non-multiple problem shapes must be handled by padding,
masked software scheduling, or layout choices.

Resolved direction:

- software is responsible for placing memory in the correct tiled layout
- software applies zero padding at tile granularity
- OpenXLA tiled-layout material is a useful non-normative reference

### 6.2 Floating-point corner semantics

Resolved direction:

- subnormals are treated as zero
- infinities are not supported as a normal architectural outcome
- overflow clamps to the largest-magnitude representable value
- FP8/BF16 conversion uses round-to-nearest
- NaNs are expected to be absent in valid software-generated workloads, but propagate if
  encountered

### 6.3 Error handling model

Resolved direction:

- Penguin does not support general trap or interrupt handling
- on any architecturally detected error, execution halts and the error state is reported
  via CSR/status
- Penguin does not attempt recovery or exception-handler dispatch

### 6.4 Same-register simultaneous access

Penguin now rules out partial architectural completion, but it still needs a sharper rule
for same-cycle read/write or write/write targeting the same tensor register by different
units.

Resolved direction:

- multiple in-flight tensor instructions may not access the same architectural tensor
  register
- software is responsible for avoiding those cases
- future perf modeling should check this rule explicitly

### 6.5 Executable-package schema

The machine now has enough frozen structure that the package format should be defined
soon.

Still deferred:

- IMEM image format
- assembly/binary relationship
- manifest schema
- tensor blob addressing and layout

## 7. Recommended Follow-On Spec Work

The next useful documents or revisions should be:

1. tensor ISA and encoding spec
2. executable-package spec
3. tensor layout / padding / packing spec
4. host control and CSR map spec
