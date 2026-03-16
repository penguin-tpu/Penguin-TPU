# Penguin-TPU TODO

This file is the single source of truth for active pending work, undecided items, and
repo-level TODO tracking. Keep historical context and completed-change notes in
`SOUL.md`, but keep open work here.

## Repo-Wide

- Keep `TODO.md` as the only place for active TODOs, pending tasks, and to-be-decided
  items. Other docs may reference this file, but should not maintain their own open-work
  lists.
- Define the formal tensor layout and packing specification, including padding and
  tail-handling rules.
- Define the 32-bit binary/text assembly encoding specification used across compiler,
  model, and RTL.
- Define the host control and CSR map for launch, halt, done, and error reporting.
- Finalize tensor-era ISA decisions that affect multiple stacks:
  - exact DMA instruction shapes
  - exact `vload` / `vstore` encodings
  - exact XLU transpose instruction encoding and sequencing
  - whether VMEM is logically unified or internally partitioned by traffic class
  - halt behavior while long-chime tensor operations are in flight

## Compiler

- Implement the direct PyTorch-to-Penguin export path for the intended fixed model flow.
- Move executable-package manifest and symbol-table usage from example/test-only flows to
  actual compiler export paths.
- Define and stabilize the compiler-side lowering boundary for the current staged
  Gemma-style examples.
- Extend the executable-package contract so `constants.bin` has a defined runtime memory
  mapping instead of being bundle collateral only.

## Modeling

- Add manifest-driven runtime staging for `constants.bin` once the bundle contract defines
  its memory mapping.
- Refine long-chime tensor execution timing and overlap semantics for MXU, VPU, and XLU
  work.
- Decide whether `mem_base` is architecturally a high-bits extension
  `(mem_base << 32) | low32` or some other shared address-offset encoding.
- Decide whether the first architected VPU data view remains BF16-only or also includes
  FP8 elementwise views.
- Decide whether the first XLU architecture cut stays BF16-only for transpose or grows a
  separate FP8 transpose path.

## RTL

- Build the RTL execution/testbench flow around the same executable package artifacts used
  by the software stack.
- Connect compiler- or vector-generated assembly/bundle artifacts into cocotb-based RTL
  regression runs.
- Move from FPGA hello-world bring-up into actual Penguin-core bring-up and validation.
- Reuse the same executable-package artifacts for on-hardware FPGA validation after RTL
  execution is in place.
