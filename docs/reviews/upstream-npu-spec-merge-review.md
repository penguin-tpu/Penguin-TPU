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

Upstream includes an XLU for transpose. Penguin currently has no dedicated transpose unit
in the frozen baseline.

Implication:

- transpose, layout reshaping, or similar data movement must currently be handled through
  VMEM layout choices, VPU support, or later hardware additions
- if workload analysis proves transpose is hot, this may deserve revisiting

### 4.5 Datatype breadth

Upstream keeps multiple floating-point formats in play. Penguin intentionally narrows the
first tensor contract to `FP8_e4m3` multiplicands and BF16 accumulation.

Implication:

- the numeric contract is simpler and easier to verify
- future broadening should be justified by workload evidence rather than added preemptively

## 5. Questions Or Choice Points Still Worth User Review

These are the remaining places where a real design decision may still be needed.

### 5.1 Exact host CSR map

Penguin now defines the logical control/status state it needs, but not the exact MMIO
layout.

Decision still needed:

- whether to freeze host-visible CSR addresses now or wait for SoC integration

### 5.2 Shared memory-base CSR access path

Penguin architecture requires a shared `mem_base` CSR, but the current scalar subset does
not yet define a full CSR-manipulation instruction path.

Decision still needed:

- host-programmed only for early bring-up
- or add Penguin scalar CSR instructions earlier

### 5.3 DMA channel reuse semantics

The spec now defines busy state and waits, but not whether a channel supports internal
queuing or exactly one outstanding operation.

Decision still needed:

- single outstanding operation per channel
- or queued operations behind each channel

### 5.4 VMEM alignment for tensor-facing operations

DMA alignment is now frozen at 32 bytes. VMEM alignment for `vload`, `vstore`, and
`mxu.push.*` is still open.

Decision still needed:

- likely choose between 32-byte alignment and stronger whole-object alignment

### 5.5 Dedicated transpose support

The upstream XLU makes the cost of transpose explicit. Penguin currently leaves this out.

Decision still needed:

- keep transpose out of hardware for the first slice
- or add a narrow transpose/move primitive if compiler analysis shows it is unavoidable

## 6. Risks And Pitfalls Still Not Fully Covered

These are important even though they are not all ready to freeze yet.

### 6.1 Tail handling and padding

Whole-register execution means non-multiple problem shapes must be handled by padding,
masked software scheduling, or layout choices.

Missing contract:

- exact zero-fill or padding rules in the executable package and compiler output

### 6.2 Floating-point corner semantics

The spec now covers output rounding and saturation, but it still does not fully define:

- NaN behavior
- infinity behavior
- subnormal handling
- FP8/BF16 conversion corner cases

### 6.3 Trap behavior with in-flight long-chime operations

The scalar subset has trap conditions, but the tensor-side spec does not yet say what
happens if a trap, halt, or host stop occurs while MXU/VPU work is in flight.

Missing contract:

- drain before halt
- immediate stop
- or precise/imprecise trap policy

Saturn is a useful reference here because it is explicit about commit boundaries and
precise-fault handling. Penguin will almost certainly want a much simpler rule, but it
should still be a consciously chosen rule.

### 6.4 Same-register simultaneous access

Penguin now rules out partial architectural completion, but it still needs a sharper rule
for same-cycle read/write or write/write targeting the same tensor register by different
units.

Missing contract:

- whether issue must prevent those cases
- or whether arbitration plus a deterministic priority rule is allowed

### 6.5 Executable-package schema

The machine now has enough frozen structure that the package format should be defined
soon.

Missing contract:

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
5. floating-point corner-case behavior note
